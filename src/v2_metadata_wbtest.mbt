///|
fn write_test_bytes(buffer : @buffer.Buffer, data : Bytes) -> Unit {
  for byte in data {
    buffer.write_byte(byte)
  }
}

///|
fn write_test_zeroes(buffer : @buffer.Buffer, count : Int32) -> Unit {
  for _ in 0..<count {
    buffer.write_byte(b'\x00')
  }
}

///|
fn write_test_checksum(buffer : @buffer.Buffer, start : Int32) -> Unit {
  buffer.write_uint_le(
    metadata_checksum(buffer.to_bytes(), start, buffer.length() - start),
  )
}

///|
fn synthetic_group_btree(header_type : Int32, total : UInt64) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=400)
  write_test_bytes(buffer, b"BTHD")
  buffer.write_byte(b'\x00')
  buffer.write_byte(header_type.to_byte())
  write_test_bytes(buffer, b"\x80\x00\x00\x00")
  buffer.write_uint16_le(5)
  buffer.write_uint16_le(1)
  buffer.write_byte(b'\x64')
  buffer.write_byte(b'\x28')
  buffer.write_uint64_le(128UL)
  buffer.write_uint16_le(1)
  buffer.write_uint64_le(total)
  write_test_checksum(buffer, 0)
  write_test_zeroes(buffer, 90)

  write_test_bytes(buffer, b"BTIN")
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x05')
  write_test_bytes(buffer, b"\x02\x00\x00\x00")
  buffer.write_byte(b'\x42')
  buffer.write_uint64_le(256UL)
  buffer.write_byte(b'\x01')
  buffer.write_uint64_le(384UL)
  buffer.write_byte(b'\x01')
  write_test_checksum(buffer, 128)
  write_test_zeroes(buffer, 95)

  write_test_bytes(buffer, b"BTLF")
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x05')
  write_test_bytes(buffer, b"\x00\x00\x00\x00")
  buffer.write_byte(b'\x41')
  write_test_checksum(buffer, 256)
  write_test_zeroes(buffer, 113)

  write_test_bytes(buffer, b"BTLF")
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x05')
  write_test_bytes(buffer, b"\x03\x00\x00\x00")
  buffer.write_byte(b'\x43')
  write_test_checksum(buffer, 384)
  write_test_zeroes(buffer, 113)
  buffer.to_bytes()
}

///|
test "metadata v2 B-tree traverses internal nodes in record order" {
  let records = read_metadata_btree_records(
    synthetic_group_btree(5, 3UL),
    0UL,
    8,
    8,
    0UL,
    [5, 6],
  )
  assert_eq(records.length(), 3)
  assert_eq(records[0].heap_id, b"\x41")
  assert_eq(records[1].heap_id, b"\x42")
  assert_eq(records[2].heap_id, b"\x43")
}

///|
test "metadata v2 B-tree rejects mismatched types and subtree counts" {
  let wrong_type = try {
    read_metadata_btree_records(synthetic_group_btree(8, 3UL), 0UL, 8, 8, 0UL, [
      5, 6,
    ])
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_type,
    Some(InvalidField("metadata v2 B-tree type does not match its index")),
  )

  let wrong_count = try {
    read_metadata_btree_records(synthetic_group_btree(5, 2UL), 0UL, 8, 8, 0UL, [
      5, 6,
    ])
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_count,
    Some(InvalidField("invalid metadata v2 B-tree subtree count")),
  )
}

///|
test "metadata v2 B-tree verifies header and node checksums" {
  let bytes = synthetic_group_btree(5, 3UL)
  let header_corrupt = bytes.to_array()
  header_corrupt[34] = header_corrupt[34] ^ b'\x01'
  let header_error = try {
    read_metadata_btree_records(
      Bytes::from_array(header_corrupt),
      0UL,
      8,
      8,
      0UL,
      [5, 6],
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    header_error,
    Some(InvalidField("metadata v2 B-tree header checksum mismatch")),
  )

  let node_corrupt = bytes.to_array()
  node_corrupt[128 + 6] = node_corrupt[128 + 6] ^ b'\x01'
  let node_error = try {
    read_metadata_btree_records(
      Bytes::from_array(node_corrupt),
      0UL,
      8,
      8,
      0UL,
      [5, 6],
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    node_error,
    Some(InvalidField("metadata v2 B-tree node checksum mismatch")),
  )
}
