///|
test "lookup3 matches Bob Jenkins reference vectors" {
  assert_eq(metadata_checksum(b"a", 0, 1), 0x58D68708U)
  assert_eq(metadata_checksum(b"abc", 0, 3), 0x0E397631U)
  assert_eq(metadata_checksum(b"123456789", 0, 9), 0x845D9A96U)
  assert_eq(metadata_checksum(b"123456789012", 0, 12), 0xB578F5B0U)
  assert_eq(metadata_checksum(b"1234567890123", 0, 13), 0xDE86D9F5U)
}

///|
test "metadata checksum accepts a matching stored value" {
  let payload = b"payload"
  let checksum = metadata_checksum(payload, 0, payload.length())
  let buffer = @buffer.Buffer::Buffer(size_hint=16)
  for byte in payload {
    buffer.write_byte(byte)
  }
  buffer.write_uint_le(checksum)
  verify_metadata_checksum(buffer.to_bytes(), 0, payload.length(), "test block")
}

///|
test "metadata checksum reports mismatches" {
  let payload = b"payload"
  let checksum = metadata_checksum(payload, 0, payload.length())
  let buffer = @buffer.Buffer::Buffer(size_hint=16)
  for byte in payload {
    buffer.write_byte(byte)
  }
  buffer.write_uint_le(checksum + 1U)
  let error = try {
    verify_metadata_checksum(
      buffer.to_bytes(),
      0,
      payload.length(),
      "test block",
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidField("test block checksum mismatch")))
}

///|
test "zeroed checksum fields are excluded from the computation" {
  let buffer = @buffer.Buffer::Buffer(size_hint=24)
  buffer.write_bytes(b"FHDB\x00\x08\x00\x00\x00\x00\x00\x00\x00")
  buffer.write_uint_le(0U)
  buffer.write_bytes(b"object")
  let bytes = buffer.to_bytes()
  let checksum_offset = 9
  let checksum = checksum_lookup3(
    bytes,
    0,
    bytes.length(),
    checksum_offset,
    4,
    0U,
  )
  let patched = @buffer.Buffer::Buffer(size_hint=24)
  patched.write_bytes(bytes.get_view(start=0, end=checksum_offset).unwrap())
  patched.write_uint_le(checksum)
  patched.write_bytes(
    bytes.get_view(start=checksum_offset + 4, end=bytes.length()).unwrap(),
  )
  verify_metadata_checksum_zeroed(
    patched.to_bytes(),
    0,
    bytes.length(),
    checksum_offset,
    "direct block",
  )
}

///|
test "lookup3 matches a stored superblock checksum" {
  let bytes = @fs.read_file_to_bytes("testdata/latest_v3.h5")
  let superblock = parse_superblock(bytes)
  let covered = superblock.superblock_end -
    superblock._superblock_offset -
    metadata_checksum_size
  assert_eq(
    superblock.superblock_checksum,
    metadata_checksum(bytes, superblock._superblock_offset, covered),
  )
}
