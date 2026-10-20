///|
fn test_fractal_heap(
  bytes : Bytes,
  id_length : Int32,
  managed_offset_size : Int32,
  managed_length_size : Int32,
) -> FractalHeap {
  {
    bytes,
    source: None,
    header_address: 0UL,
    offset_size: 8,
    length_size: 8,
    base_address: 0UL,
    id_length,
    flags: 0,
    table_width: 4,
    start_block_size: 512UL,
    managed_offset_size,
    managed_length_size,
    block_offset_size: 2,
    max_direct_rows: 4,
    indirect_rows_base: 6,
    managed_space_size: 4096UL,
    huge_btree_address: 0xFFFFFFFFFFFFFFFFUL,
    filter_len: 0,
    filters: [],
    huge_id_size: if id_length - 1 < 8 {
      id_length - 1
    } else {
      8
    },
    huge_ids_direct: id_length - 1 >= 8,
    blocks: [],
  }
}

///|
test "fractal heap IDs decode tiny and direct huge objects" {
  let tiny = test_fractal_heap(b"", 7, 2, 1)
  assert_eq(tiny.get_data(b"\x22abc\x00\x00\x00"), b"abc")

  let huge = test_fractal_heap(b"payload", 17, 2, 2)
  let buffer = @buffer.Buffer::Buffer(size_hint=17)
  buffer.write_byte(b'\x10')
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(7UL)
  assert_eq(huge.get_data(buffer.to_bytes()), b"payload")
}

///|
test "fractal heap IDs reject missing huge indexes and out-of-block objects" {
  let heap = test_fractal_heap(b"", 8, 2, 1)
  let indirect_huge = try {
    let _ = heap.get_data(b"\x10\x00\x00\x00\x00\x00\x00\x00")
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    indirect_huge,
    Some(InvalidField("fractal heap has no huge object B-tree")),
  )

  let managed = try {
    let _ = test_fractal_heap(b"", 4, 2, 1).get_data(b"\x00\x00\x00\x01")
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    managed,
    Some(InvalidField("fractal heap object offset is outside blocks")),
  )

  let truncated_huge = try {
    let _ = test_fractal_heap(b"", 17, 2, 1).get_data(
      b"\x30\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(truncated_huge, Some(InvalidField("invalid fractal heap ID type")))
}

///|
fn test_fractal_heap_with_checksums(
  bytes : Bytes,
  flags : Int32,
) -> FractalHeap {
  {
    bytes,
    source: None,
    header_address: 0UL,
    offset_size: 8,
    length_size: 8,
    base_address: 0UL,
    id_length: 8,
    flags,
    table_width: 4,
    start_block_size: 32UL,
    managed_offset_size: 2,
    managed_length_size: 1,
    block_offset_size: 2,
    max_direct_rows: 4,
    indirect_rows_base: 6,
    managed_space_size: 4096UL,
    huge_btree_address: 0xFFFFFFFFFFFFFFFFUL,
    filter_len: 0,
    filters: [],
    huge_id_size: 7,
    huge_ids_direct: false,
    blocks: [],
  }
}

///|
fn direct_block_fixture(corrupt : Bool) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=32)
  buffer.write_bytes(b"FHDB\x00")
  buffer.write_uint64_le(0UL)
  buffer.write_uint16_le(0)
  let checksum_offset = buffer.length()
  buffer.write_uint_le(0U)
  buffer.write_bytes(b"payload")
  for _ in 0..<(32 - buffer.length()) {
    buffer.write_byte(b'\x00')
  }
  let raw = buffer.to_bytes()
  let checksum = checksum_lookup3(raw, 0, raw.length(), checksum_offset, 4, 0U)
  let patched = @buffer.Buffer::Buffer(size_hint=32)
  patched.write_bytes(raw.get_view(start=0, end=checksum_offset).unwrap())
  patched.write_uint_le(if corrupt { checksum + 1U } else { checksum })
  patched.write_bytes(
    raw.get_view(start=checksum_offset + 4, end=raw.length()).unwrap(),
  )
  patched.to_bytes()
}

///|
test "fractal heap verifies direct block checksums" {
  let heap = test_fractal_heap_with_checksums(direct_block_fixture(false), 2)
  heap.add_direct_block(0UL, 32UL, 32UL, 0U, 0)
  assert_eq(heap.blocks.length(), 1)

  let damaged = test_fractal_heap_with_checksums(direct_block_fixture(true), 2)
  let error = try {
    damaged.add_direct_block(0UL, 32UL, 32UL, 0U, 0)
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    error,
    Some(InvalidField("fractal heap direct block checksum mismatch")),
  )
}

///|
fn indirect_block_fixture(corrupt : Bool) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=64)
  buffer.write_bytes(b"FHIB\x00")
  buffer.write_uint64_le(0UL)
  buffer.write_uint16_le(0)
  for _ in 0..<4 {
    buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  }
  let checksum_offset = buffer.length()
  buffer.write_uint_le(0U)
  let raw = buffer.to_bytes()
  let checksum = metadata_checksum(raw, 0, checksum_offset)
  let patched = @buffer.Buffer::Buffer(size_hint=64)
  patched.write_bytes(raw.get_view(start=0, end=checksum_offset).unwrap())
  patched.write_uint_le(if corrupt { checksum + 1U } else { checksum })
  patched.write_bytes(
    raw.get_view(start=checksum_offset + 4, end=raw.length()).unwrap(),
  )
  patched.to_bytes()
}

///|
test "fractal heap always verifies indirect block checksums" {
  let heap = test_fractal_heap_with_checksums(indirect_block_fixture(false), 0)
  heap.add_indirect_block(0UL, 1, 0, [])

  let damaged = test_fractal_heap_with_checksums(
    indirect_block_fixture(true),
    0,
  )
  let error = try {
    damaged.add_indirect_block(0UL, 1, 0, [])
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    error,
    Some(InvalidField("fractal heap indirect block checksum mismatch")),
  )
}

///|
test "fractal heap header checksum corruption is rejected" {
  let bytes = @fs.read_file_to_bytes("testdata/attribute_storage.h5")
  let mut offset = 0
  while offset + 4 <= bytes.length() &&
        !bytes
        .get_view(start=offset, end=offset + 4)
        .unwrap()
        .equal_to_bytes(b"FRHP") {
    offset = offset + 1
  }
  let checksum_offset = offset + 4 + 1 + 2 + 2 + 1 + 4 + 12 * 8 + 3 * 8 + 8
  let corrupted = bytes.to_array()
  corrupted[checksum_offset] = corrupted[checksum_offset] ^ b'\x01'
  let error = try {
    parse_fractal_heap(
      Bytes::from_array(corrupted),
      offset.to_uint64(),
      8,
      8,
      0UL,
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidField("fractal heap header checksum mismatch")))
}

///|
fn test_filtered_fractal_heap(bytes : Bytes) -> FractalHeap {
  {
    bytes,
    source: None,
    header_address: 0UL,
    offset_size: 8,
    length_size: 8,
    base_address: 0UL,
    id_length: 8,
    flags: 2,
    table_width: 4,
    start_block_size: 32UL,
    managed_offset_size: 2,
    managed_length_size: 1,
    block_offset_size: 2,
    max_direct_rows: 4,
    indirect_rows_base: 6,
    managed_space_size: 4096UL,
    huge_btree_address: 0xFFFFFFFFFFFFFFFFUL,
    filter_len: 12,
    filters: [{ id: 1, parameters: [6U], }],
    huge_id_size: 7,
    huge_ids_direct: false,
    blocks: [],
  }
}

///|
test "filtered fractal heap decodes and checksums direct blocks" {
  let logical = direct_block_fixture(false)
  let compressed = @zlib.zlib(logical[:])
  let heap = test_filtered_fractal_heap(compressed)
  heap.add_direct_block(0UL, 32UL, compressed.length().to_uint64(), 0U, 0)
  assert_eq(heap.blocks.length(), 1)
  assert_eq(heap.get_data(b"\x00\x13\x00\x07\x00\x00\x00\x00"), b"payload")

  let damaged = test_filtered_fractal_heap(
    @zlib.zlib(direct_block_fixture(true)[:]),
  )
  let error = try {
    damaged.add_direct_block(
      0UL,
      32UL,
      damaged.bytes.length().to_uint64(),
      0U,
      0,
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    error,
    Some(InvalidField("fractal heap direct block checksum mismatch")),
  )
}

///|
test "filtered fractal heaps and huge objects resolve real fixtures" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/fractal_heap_advanced.h5"),
  )
  let filtered = file.resolve("/filtered")
  let filtered_links = file.links(file.object_header_at(filtered.address))
  assert_eq(filtered_links.length(), 1201)
  assert_eq(filtered_links[0].name, "item_0000")
  assert_true(filtered_links[1200].name.has_prefix("long_"))
  assert_eq(
    file.resolve("/filtered/" + filtered_links[1200].name).kind,
    Dataset,
  )

  let raw = file.resolve("/raw")
  let raw_links = file.links(file.object_header_at(raw.address))
  assert_eq(raw_links.length(), 13)
  assert_true(raw_links[12].name.has_prefix("long_"))
  assert_eq(file.resolve("/raw/" + raw_links[12].name).kind, Dataset)
}

///|
test "fractal heap decodes directly accessed filtered huge objects" {
  let compressed = @zlib.zlib(b"payload"[:])
  let heap : FractalHeap = {
    bytes: compressed,
    source: None,
    header_address: 0UL,
    offset_size: 8,
    length_size: 8,
    base_address: 0UL,
    id_length: 29,
    flags: 0,
    table_width: 4,
    start_block_size: 32UL,
    managed_offset_size: 2,
    managed_length_size: 1,
    block_offset_size: 2,
    max_direct_rows: 4,
    indirect_rows_base: 6,
    managed_space_size: 4096UL,
    huge_btree_address: 0xFFFFFFFFFFFFFFFFUL,
    filter_len: 12,
    filters: [{ id: 1, parameters: [6U], }],
    huge_id_size: 24,
    huge_ids_direct: true,
    blocks: [],
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=29)
  buffer.write_byte(b'\x10')
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(compressed.length().to_uint64())
  buffer.write_uint_le(0U)
  buffer.write_uint64_le(7UL)
  assert_eq(heap.get_data(buffer.to_bytes()), b"payload")
}
