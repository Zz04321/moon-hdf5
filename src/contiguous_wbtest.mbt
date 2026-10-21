///|
test "unallocated contiguous datasets return typed fill values" {
  for
    path in ["testdata/contiguous_fill_v2.h5", "testdata/contiguous_fill_v3.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    assert_eq(integer_values(file.read_dataset("/signed")), [-7L, -7L, -7L])
    assert_eq(integer_values(file.read_dataset("/big_endian")), [
      -1234L, -1234L, -1234L,
    ])
    assert_eq(
      unsigned_values(file.read_dataset("/unsigned")),
      Array::make(3, 9223372036854775809UL),
    )
    assert_eq(float_values(file.read_dataset("/floating")), [1.5, 1.5, 1.5])
    assert_eq(string_values(file.read_dataset("/text")), [
      "none", "none", "none",
    ])
    assert_eq(integer_values(file.read_dataset("/default")), [0L, 0L, 0L])
    assert_eq(integer_values(file.read_dataset("/other/scalar")), [-2L])
    assert_eq(integer_values(file.read_dataset("/other/allocated")), [
      -7L, 42L, -7L,
    ])
  }
}

///|
test "unallocated storage does not hide external data or disabled fills" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/contiguous_fill_v3.h5"),
  )
  let original = file.object_header_at(file.resolve("/signed").address)
  for external in [false, true] {
    let messages = original.messages.filter(message => {
      message.msg_type != msg_fill_value
    })
    messages.push({
      msg_type: if external {
        msg_external_data_files
      } else {
        msg_fill_value
      },
      flags: b'\x00',
      data: b"\x03\x06",
    })
    let header : ObjectHeader = { _version: original._version, messages, }
    let error = try {
      read_dataset(
        file.bytes,
        header,
        file.superblock.offset_size,
        file.superblock.length_size,
        file.superblock.base_address,
      )
      |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(UnsupportedFeature(_)))
  }
}

///|
test "empty and null contiguous datasets do not dereference undefined addresses" {
  for
    path in ["testdata/contiguous_fill_v2.h5", "testdata/contiguous_fill_v3.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    let empty = file.read_dataset("/other/empty")
    assert_eq(empty.dataspace.dimensions, [0UL, 3UL])
    assert_eq(integer_values(empty), [])
    let null = file.read_dataset("/other/null")
    assert_eq(null.dataspace.kind, Null)
    assert_eq(integer_values(null), [])
  }
}
