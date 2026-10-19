///|
test "fixed arrays read filtered edge chunks and bounded maximum strides" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/fixed_array.h5"))
  for (name, count) in [("raw", 35), ("filtered", 35), ("resizable", 12)] {
    let dataset = file.open_dataset("/" + name)
    assert_true(dataset.layout.chunk_index is FixedArray(10))
    let values = integer_values(dataset)
    assert_eq(values.length(), count)
    for index in 0..<count {
      assert_eq(values[index], index.to_int64())
    }
  }
  assert_eq(integer_values(file.open_dataset("/skipped")), [
    0L, 1L, 2L, 3L, -3L, -3L, -3L, -3L,
  ])
  assert_eq(
    integer_values(file.open_dataset("/unallocated")),
    Array::make(8, -5L),
  )
}

///|
test "paged fixed arrays preserve missing chunks and uninitialized pages" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/fixed_array.h5"))
  for name in ["paged", "paged_filtered"] {
    let values = integer_values(file.open_dataset("/" + name))
    assert_eq(values.length(), 4099)
    for index in 0..<4099 {
      assert_eq(
        values[index],
        if index == 0 || index == 1023 || index == 2048 || index == 4098 {
          index.to_int64()
        } else {
          -7L
        },
      )
    }
  }
}

///|
test "fixed array rejects corrupt metadata and truncated entries" {
  let bytes = @fs.read_file_to_bytes("testdata/fixed_array.h5")
  let file = Hdf5File::parse(bytes)
  let dataset = file.open_dataset("/raw")
  let header = dataset.layout.data_address.to_int()
  let block = Reader::at(bytes, header + 16).read_u64_le().to_int()
  for
    position in [
      header,
      header + 4,
      header + 5,
      header + 6,
      header + 7,
      header + 8,
      block,
      block + 4,
      block + 5,
      block + 6,
    ] {
    let corrupted = bytes.to_array()
    corrupted[position] = corrupted[position] ^ b'\x01'
    let error = try {
      read_chunked_payload(
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
  for end in [header + 27, block + 13, block + 14 + 9 * 8 + 3] {
    let error = try {
      read_chunked_payload(
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
