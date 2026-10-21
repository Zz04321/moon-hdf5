///|
test "extensible arrays read growing grids and swizzled unlimited dimensions" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/extensible_array.h5"),
  )
  for group in ["raw", "filtered"] {
    for (name, count) in [("grown", 420), ("swizzled", 210)] {
      let dataset = file.read_dataset("/" + group + "/" + name)
      assert_true(dataset.layout.chunk_index is ExtensibleArray(_))
      let values = integer_values(dataset)
      assert_eq(values.length(), count)
      for index in 0..<count {
        assert_eq(values[index], index.to_int64())
      }
    }
  }
  assert_eq(integer_values(file.read_dataset("/other/skipped")), [
    -3L, -3L, -3L, -3L, 0L, 1L, 2L, 3L, -3L, -3L, -3L, -3L,
  ])
  assert_eq(
    integer_values(file.read_dataset("/other/unallocated")),
    Array::make(8, -5L),
  )
  assert_eq(integer_values(file.read_dataset("/other/empty")), [])
}

///|
test "extensible arrays traverse direct blocks super blocks and sparse pages" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/extensible_array.h5"),
  )
  let written = [
    0, 3, 4, 19, 20, 51, 52, 115, 116, 243, 244, 499, 500, 1011, 1012, 65523, 65524,
    131060, 133108, 199999,
  ]
  for group in ["raw", "filtered"] {
    let values = integer_values(file.read_dataset("/" + group + "/paged"))
    assert_eq(values.length(), 200000)
    for index in 0..<values.length() {
      assert_eq(
        values[index],
        if written.contains(index) {
          index.to_int64()
        } else {
          -7L
        },
      )
    }
    assert_eq(values[131059], -7L)
  }
}

///|
test "extensible arrays reject damaged headers blocks and truncated pages" {
  let bytes = @fs.read_file_to_bytes("testdata/extensible_array.h5")
  let file = Hdf5File::parse(bytes)
  let dataset = file.read_dataset("/raw/paged")
  let header = dataset.layout.data_address.to_int()
  let index = Reader::at(bytes, header + 60).read_u64_le().to_int()
  let direct = Reader::at(bytes, index + 46).read_u64_le().to_int()
  let super_block = Reader::at(bytes, index + 94 + 9 * 8).read_u64_le().to_int()
  let paged_block = Reader::at(bytes, super_block + 18 + 64)
    .read_u64_le()
    .to_int()
  for
    position in [
      header,
      header + 4,
      header + 5,
      header + 6,
      header + 7,
      header + 8,
      header + 9,
      header + 10,
      header + 11,
      index,
      index + 4,
      index + 5,
      index + 6,
      direct,
      direct + 14,
      super_block,
      super_block + 14,
      paged_block,
      paged_block + 14,
    ] {
    let corrupted = bytes.to_array()
    corrupted[position] = corrupted[position] ^ b'\x01'
    let error = try {
      _read_chunked_payload(
        Bytes::from_array(corrupted),
        dataset.layout,
        dataset.dataspace,
        dataset.datatype,
        8,
        0UL,
      )
      |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(InvalidField(_)))
  }
  for
    end in [
      header + 71,
      index + 14,
      direct + 18,
      super_block + 18,
      paged_block + 22 + 1024 * 8 + 3,
    ] {
    let error = try {
      _read_chunked_payload(
        bytes.get_view(start=0, end~).unwrap().to_owned(),
        dataset.layout,
        dataset.dataspace,
        dataset.datatype,
        8,
        0UL,
      )
      |> ignore
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
test "extensible arrays respect four byte file addresses" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/extensible_array_32.h5"),
  )
  assert_eq(file.superblock.offset_size, 4)
  assert_eq(file.superblock.length_size, 8)
  for name in ["raw", "filtered"] {
    let values = integer_values(file.read_dataset("/" + name))
    assert_eq(values.length(), 1500)
    for index in 0..<1500 {
      assert_eq(values[index], index.to_int64())
    }
  }
}

///|
test "extensible layout configuration rejects unsafe shifts and invalid powers" {
  for
    params in [
      b"\x00\x04\x04\x10\x0a", b"\x41\x04\x04\x10\x0a", b"\x20\x00\x04\x10\x0a",
      b"\x20\x04\x03\x10\x0a", b"\x20\x04\x04\x0f\x0a", b"\x20\x04\x04\x10\x20",
    ] {
    let buffer = @buffer.Buffer::Buffer(size_hint=21)
    for byte in b"\x04\x02\x00\x02\x01\x01\x04\x04" {
      buffer.write_byte(byte)
    }
    for byte in params {
      buffer.write_byte(byte)
    }
    buffer.write_uint64_le(100UL)
    let error = try {
      parse_data_layout(
        { msg_type: msg_data_layout, flags: b'\x00', data: buffer.to_bytes(), },
        8,
        8,
      )
      |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(InvalidField(_)))
  }
}
