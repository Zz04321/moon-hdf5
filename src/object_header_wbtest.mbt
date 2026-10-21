///|
test "modern object header continuations preserve all root links" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/header_continuations.h5"),
  )
  let nodes = file.walk()
  assert_eq(nodes.length(), 7)
  for
    name in [
      "signed", "big_endian", "floating", "text", "unallocated", "default", "partial_default",
    ] {
    assert_true(nodes.any(node => node.path == "/" + name))
    assert_eq(integer_values(file.read_dataset("/" + name)), [42L])
  }
}

///|
fn continuation_fixture(gap : Int32, ordered : Bool) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=80)
  let prefix = if ordered { 6 } else { 4 }
  let child_address = 7 + prefix + 16 + 4
  for byte in b"OHDR" {
    buffer.write_byte(byte)
  }
  buffer.write_byte(b'\x02')
  buffer.write_byte(if ordered { b'\x04' } else { b'\x00' })
  buffer.write_byte((prefix + 16).to_byte())
  buffer.write_byte(b'\x10')
  buffer.write_uint16_le(16)
  buffer.write_byte(b'\x00')
  if ordered {
    buffer.write_uint16_le(0)
  }
  buffer.write_uint64_le(child_address.to_uint64())
  buffer.write_uint64_le((8 + prefix + gap).to_uint64())
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  let continuation_start = buffer.length()
  for byte in b"OCHK" {
    buffer.write_byte(byte)
  }
  buffer.write_byte(b'\x00')
  buffer.write_uint16_le(0)
  buffer.write_byte(b'\x00')
  if ordered {
    buffer.write_uint16_le(1)
  }
  for _ in 0..<gap {
    buffer.write_byte(b'\xff')
  }
  buffer.write_uint_le(
    metadata_checksum(
      buffer.to_bytes(),
      continuation_start,
      buffer.length() - continuation_start,
    ),
  )
  buffer.to_bytes()
}

///|
test "continuation checksum and short gaps are excluded from messages" {
  for ordered in [false, true] {
    let prefix = if ordered { 6 } else { 4 }
    for gap in 0..<prefix {
      let bytes = continuation_fixture(gap, ordered)
      let header = parse_object_header(bytes, 0UL, 8, 8, 0UL)
      assert_eq(header.messages.length(), 2)
      assert_eq(header.messages[1].msg_type, _msg_nil)
      assert_eq(header.messages[1].data.length(), 0)
      let truncated = bytes
        .get_view(start=0, end=bytes.length() - 1)
        .unwrap()
        .to_owned()
      let error = try {
        parse_object_header(truncated, 0UL, 8, 8, 0UL) |> ignore
        None
      } catch {
        error => Some(error)
      }
      assert_true(error is Some(InvalidField(_)))
    }
  }
}

///|
test "initial chunk includes a complete final empty message" {
  let buffer = @buffer.Buffer::Buffer(size_hint=16)
  buffer.write_bytes(b"OHDR\x02\x00\x04\x00\x00\x00\x00")
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  let bytes = buffer.to_bytes()
  let header = parse_object_header(bytes, 0UL, 8, 8, 0UL)
  assert_eq(header.messages.length(), 1)
  assert_eq(header.messages[0].msg_type, _msg_nil)
  let error = try {
    let truncated = bytes
      .get_view(start=0, end=bytes.length() - 1)
      .unwrap()
      .to_owned()
    parse_object_header(truncated, 0UL, 8, 8, 0UL) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(UnexpectedEof(_)))
}

///|
test "object header checksum corruption is rejected" {
  let buffer = @buffer.Buffer::Buffer(size_hint=16)
  buffer.write_bytes(b"OHDR\x02\x00\x04\x00\x00\x00\x00")
  buffer.write_uint_le(
    metadata_checksum(buffer.to_bytes(), 0, buffer.length()) + 1U,
  )
  let error = try {
    parse_object_header(buffer.to_bytes(), 0UL, 8, 8, 0UL) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidField("object header checksum mismatch")))
}

///|
test "64 bit header chunk size is checked before narrowing" {
  let error = try {
    parse_object_header(
      b"OHDR\x02\x03\x00\x00\x00\x00\x01\x00\x00\x00", 0UL, 8, 8, 0UL,
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidField("object header chunk size is too large")))
}
