///|
test "dataspace products reject overflow instead of wrapping" {
  for
    dimensions in [
      [1073741824UL, 1073741824UL, 16UL],
      [2147483647UL, 2147483647UL, 3UL],
    ] {
    let space : Dataspace = {
      kind: Simple,
      dimensions,
      maximum_dimensions: [],
    }
    let error = try {
      space.element_count() |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(InvalidField(_)))
  }
}

///|
test "empty dataspaces remain empty regardless of dimension order" {
  for
    dimensions in [
      [0UL, 1073741824UL, 1073741824UL, 16UL],
      [1073741824UL, 1073741824UL, 16UL, 0UL],
    ] {
    let space : Dataspace = {
      kind: Simple,
      dimensions,
      maximum_dimensions: [],
    }
    assert_eq(space.element_count(), 0L)
  }
  let space : Dataspace = {
    kind: Simple,
    dimensions: [2147483647UL, 2147483647UL],
    maximum_dimensions: [],
  }
  assert_eq(space.element_count(), 4611686014132420609L)
}

///|
test "chunk products reject overflow before narrowing to Int32" {
  for dimensions in [[1073741824L, 1073741824L, 16L], [65536L, 65536L]] {
    let error = try {
      product(dimensions) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(InvalidField(_)))
  }
  assert_eq(product([2147483647L]), 2147483647L)
}

///|
test "selection products are bounded before generating indices" {
  let error = try {
    hyperslab_indices(
      [2147483647UL, 2147483647UL],
      [0UL, 0UL],
      [2147483647UL, 2147483647UL],
      [1UL, 1UL],
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    error,
    Some(InvalidArgument("selection contains too many elements")),
  )
}

///|
test "chunk byte products cannot wrap to bypass the payload check" {
  let error = try {
    copy_chunk(
      b"",
      0,
      [0UL, 0UL],
      [65536U, 65536U],
      [1UL, 1UL],
      1073741824,
      FixedArray::make(0, b'\x00'),
      None,
    )
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(InvalidField(_)))
}

///|
test "dataset byte size is checked before reading or allocating storage" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/basic_v0.h5"))
  let node = file.resolve("/group/values")
  let original = file.object_header_at(node.address)
  let buffer = @buffer.Buffer::Buffer(size_hint=24)
  for byte in b"\x02\x02\x00\x01" {
    buffer.write_byte(byte)
  }
  buffer.write_uint64_le(1073741824UL)
  buffer.write_uint64_le(1073741824UL)
  let messages = original.messages.map(message => {
    if message.msg_type == msg_dataspace {
      { msg_type: msg_dataspace, flags: b'\x00', data: buffer.to_bytes(), }
    } else {
      message
    }
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
  assert_eq(error, Some(InvalidField("dataset byte size is too large")))
}

///|
test "chunk destination bounds are checked before byte offset multiplication" {
  let error = try {
    copy_chunk(
      b"\x01\x00\x00\x00",
      4,
      [1073741824UL],
      [1U],
      [2147483647UL],
      4,
      FixedArray::make(4, b'\x00'),
      None,
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    error,
    Some(InvalidField("chunk destination exceeds output buffer")),
  )
}
