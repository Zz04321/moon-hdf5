///|
test "v4 single chunk indexes support filters masks and unallocated fills" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/layout_v4.h5"))
  for name in ["raw", "filtered"] {
    let dataset = file.open_dataset("/single/" + name)
    assert_eq(dataset.layout.version, 4)
    assert_true(dataset.layout.chunk_index is Single(_, _))
    let values = integer_values(dataset)
    assert_eq(values.length(), 12)
    for index in 0..<12 {
      assert_eq(values[index], index.to_int64())
    }
  }
  assert_eq(
    integer_values(file.open_dataset("/single/unallocated")),
    Array::make(12, -7L),
  )
  assert_eq(integer_values(file.open_dataset("/single/skipped")), [
    1L, 2L, 3L, 4L,
  ])
}

///|
test "v4 implicit indexes map chunk grids using maximum dimensions" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/layout_v4.h5"))
  for (name, count) in [("edge", 35), ("resizable", 12)] {
    let dataset = file.open_dataset("/implicit/" + name)
    assert_true(dataset.layout.chunk_index is Implicit)
    let values = integer_values(dataset)
    assert_eq(values.length(), count)
    for index in 0..<count {
      assert_eq(values[index], index.to_int64())
    }
  }
  assert_eq(integer_values(file.open_dataset("/contiguous")), [10L, 20L])
  assert_eq(integer_values(file.open_dataset("/compact")), [30L, 40L])
}

///|
test "v4 fixed array and direct chunk payload bounds" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/layout_v4.h5"))
  assert_eq(integer_values(file.open_dataset("/fixed_array")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L,
  ])
  for path in ["/single/raw", "/implicit/edge"] {
    let dataset = file.open_dataset(path)
    let error = try {
      read_direct_chunks(
        b"",
        dataset.layout,
        dataset.dataspace,
        dataset.datatype,
        file.superblock.base_address,
        [],
        FixedArray::make(140, b'\x00'),
      )
      None
    } catch {
      error => Some(error)
    }
    assert_true(
      error is Some(InvalidField(_)) || error is Some(UnexpectedEof(_)),
    )
  }
}

///|
test "v4 dimension widths and truncated metadata are validated" {
  for width in [1, 2, 3, 4, 5, 6, 7, 8] {
    let buffer = @buffer.Buffer::Buffer(size_hint=40)
    for byte in b"\x04\x02\x00\x02" {
      buffer.write_byte(byte)
    }
    buffer.write_byte(width.to_byte())
    for size in [3, 4] {
      buffer.write_byte(size.to_byte())
      for _ in 1..<width {
        buffer.write_byte(b'\x00')
      }
    }
    buffer.write_byte(b'\x01')
    buffer.write_uint64_le(100UL)
    let data = buffer.to_bytes()
    let layout = parse_data_layout(
      { msg_type: msg_data_layout, flags: b'\x00', data_offset: 0, data, },
      8,
      8,
    )
    assert_eq(layout.chunk_dimensions, [3U])
    assert_eq(layout.element_size, 4)
    assert_eq(layout.data_address, 100UL)
    for length in 0..<data.length() {
      let truncated = data.get_view(start=0, end=length).unwrap().to_owned()
      let error = try {
        parse_data_layout(
          {
            msg_type: msg_data_layout,
            flags: b'\x00',
            data_offset: 0,
            data: truncated,
          },
          8,
          8,
        )
        |> ignore
        None
      } catch {
        error => Some(error)
      }
      assert_true(error is Some(UnexpectedEof(_)))
    }
  }
}
