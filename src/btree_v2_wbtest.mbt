///|
test "v2 B-tree chunks support filters multiple unlimited dimensions and fills" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/btree_v2.h5"))
  for group in ["raw", "filtered"] {
    let dataset = file.open_dataset("/" + group + "/edge")
    assert_true(dataset.layout.chunk_index is BTreeV2(_))
    let values = integer_values(dataset)
    assert_eq(values.length(), 35)
    for index in 0..<35 {
      assert_eq(values[index], index.to_int64())
    }
    let sparse = integer_values(file.open_dataset("/" + group + "/sparse"))
    assert_eq(sparse.length(), 63)
    for index in 0..<63 {
      assert_eq(
        sparse[index],
        if index == 11 {
          42L
        } else if index == 62 {
          99L
        } else {
          -7L
        },
      )
    }
  }
  assert_eq(integer_values(file.open_dataset("/other/skipped")), [
    0L, 1L, 2L, 3L, -3L, -3L, -3L, -3L,
  ])
  assert_eq(
    integer_values(file.open_dataset("/other/unallocated")),
    Array::make(8, -5L),
  )
  assert_eq(integer_values(file.open_dataset("/other/empty")), [])
}

///|
test "v2 B-tree internal records and two levels of children are read" {
  let bytes = @fs.read_file_to_bytes("testdata/btree_v2.h5")
  let file = Hdf5File::parse(bytes)
  for group in ["raw", "filtered"] {
    let dataset = file.open_dataset("/" + group + "/deep")
    let header = dataset.layout.data_address.to_int()
    assert_true(Reader::at(bytes, header + 12).read_u16_le().to_int() >= 2)
    let values = integer_values(dataset)
    assert_eq(values.length(), 10000)
    for index in 0..<10000 {
      assert_eq(values[index], index.to_int64())
    }
  }
}

///|
test "v2 B-tree rejects corrupt metadata counts and truncated nodes" {
  let bytes = @fs.read_file_to_bytes("testdata/btree_v2.h5")
  let file = Hdf5File::parse(bytes)
  let dataset = file.open_dataset("/raw/deep")
  let header = dataset.layout.data_address.to_int()
  let root = Reader::at(bytes, header + 16).read_u64_le().to_int()
  let count = Reader::at(bytes, header + 24).read_u16_le().to_int()
  let records_size = Reader::at(bytes, header + 10).read_u16_le().to_int()
  let pointers = root + 6 + count * records_size
  for
    position in [
      header,
      header + 4,
      header + 5,
      header + 6,
      header + 10,
      header + 12,
      header + 14,
      header + 15,
      header + 24,
      header + 26,
      root,
      root + 4,
      root + 5,
      pointers + 8,
      pointers + 9,
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
  let node_size = Reader::at(bytes, header + 6)
    .read_u32_le()
    .reinterpret_as_int()
  for end in [header + 37, root + node_size - 1] {
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
  let cyclic = bytes.to_array()
  for byte_index in 0..<8 {
    cyclic[pointers + byte_index] = bytes[header + 16 + byte_index]
  }
  let error = try {
    read_chunked_payload(
      Bytes::from_array(cyclic),
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

///|
test "v2 B-tree rejects duplicate and overflowing chunk coordinates" {
  let bytes = @fs.read_file_to_bytes("testdata/btree_v2.h5")
  let file = Hdf5File::parse(bytes)
  let dataset = file.open_dataset("/raw/edge")
  let header = dataset.layout.data_address.to_int()
  let root = Reader::at(bytes, header + 16).read_u64_le().to_int()
  let record_size = Reader::at(bytes, header + 10).read_u16_le().to_int()
  for overflow in [false, true] {
    let corrupted = bytes.to_array()
    for byte_index in 0..<16 {
      corrupted[root + 6 + record_size + 8 + byte_index] = if overflow {
        b'\xff'
      } else {
        bytes[root + 6 + 8 + byte_index]
      }
    }
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
}
