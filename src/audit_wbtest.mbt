///|
test "oversized vlen string descriptors raise an error before narrowing" {
  let heap = @buffer.Buffer::Buffer(size_hint=40)
  for byte in b"GCOL\x01\x00\x00\x00" {
    heap.write_byte(byte)
  }
  heap.write_uint64_le(40UL)
  heap.write_uint16_le(1)
  heap.write_uint16_le(1)
  heap.write_uint_le(0U)
  heap.write_uint64_le(1UL)
  for byte in b"a\x00\x00\x00\x00\x00\x00\x00" {
    heap.write_byte(byte)
  }
  let datatype : Datatype = {
    class: 9,
    version: 1,
    size: 16,
    signed: false,
    little_endian: true,
    vlen_is_string: true,
    enum_members: [],
    compound_members: [],
    base_types: [],
    array_dimensions: [],
    bit_offset: 0,
    bit_precision: 0,
    string_padding: 0,
    string_charset: 0,
    opaque_tag: "",
    reference_kind: 0,
  }
  for length in [1U, 0x80000000U, 0xFFFFFFFFU] {
    let descriptor = @buffer.Buffer::Buffer(size_hint=16)
    descriptor.write_uint_le(length)
    descriptor.write_uint64_le(0UL)
    descriptor.write_uint_le(1U)
    if length == 1U {
      assert_eq(
        values_strings(
          decode_values(
            heap.to_bytes(),
            descriptor.to_bytes(),
            1L,
            datatype,
            8,
            8,
            0UL,
          ),
        ),
        ["a"],
      )
    } else {
      let error = try {
        decode_values(
          heap.to_bytes(),
          descriptor.to_bytes(),
          1L,
          datatype,
          8,
          8,
          0UL,
        )
        |> ignore
        None
      } catch {
        error => Some(error)
      }
      assert_eq(
        error,
        Some(InvalidField("variable-length string is too large")),
      )
    }
  }
}

///|
test "local heap size cannot truncate or read beyond the input" {
  for size in [64UL, 0x100000008UL] {
    let buffer = @buffer.Buffer::Buffer(size_hint=40)
    for byte in b"HEAP\x00\x00\x00\x00" {
      buffer.write_byte(byte)
    }
    buffer.write_uint64_le(size)
    buffer.write_uint64_le(0UL)
    buffer.write_uint64_le(32UL)
    buffer.write_uint64_le(0UL)
    let error = try {
      parse_local_heap(buffer.to_bytes(), 0UL, 8, 8, 0UL) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(InvalidField(_)))
  }
  for (address, size) in [(0, 4), (2147483647, 4), (-1, 2)] {
    let heap : LocalHeap = {
      bytes: b"abc",
      data_address: address,
      data_size: size,
      offset_to_free_list: 0UL,
    }
    let error = try {
      heap.get_string(0UL) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(InvalidField(_)))
  }
}

///|
test "public B-tree configuration cannot bypass threshold checks by overflow" {
  let error = try {
    validate_btree_v2_config({
      node_size: 2048,
      split_percent: 100,
      merge_percent: 2147483647,
    })
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(InvalidField(_)))
}

///|
test "global heap reads require the declared collection and object padding" {
  for
    (collection_size, payload) in [
      (40UL, b"a\x00\x00\x00\x00\x00\x00\x00"),
      (80UL, b"a"),
      (33UL, b"a"),
    ] {
    let buffer = @buffer.Buffer::Buffer(size_hint=40)
    for byte in b"GCOL\x01\x00\x00\x00" {
      buffer.write_byte(byte)
    }
    buffer.write_uint64_le(collection_size)
    buffer.write_uint16_le(1)
    buffer.write_uint16_le(1)
    buffer.write_uint_le(0U)
    buffer.write_uint64_le(1UL)
    for byte in payload {
      buffer.write_byte(byte)
    }
    if collection_size == 40UL {
      assert_eq(
        read_global_heap_object(buffer.to_bytes(), 0UL, 1, 8, 8, 0UL),
        b"a",
      )
    } else {
      let error = try {
        read_global_heap_object(buffer.to_bytes(), 0UL, 1, 8, 8, 0UL) |> ignore
        None
      } catch {
        error => Some(error)
      }
      assert_true(error is Some(InvalidField(_)))
    }
  }
}

///|
test "global heaps honor four-byte length fields" {
  let buffer = @buffer.Buffer::Buffer(size_hint=32)
  for byte in b"GCOL\x01\x00\x00\x00" {
    buffer.write_byte(byte)
  }
  buffer.write_uint_le(32U)
  buffer.write_uint16_le(1)
  buffer.write_uint16_le(1)
  buffer.write_uint_le(0U)
  buffer.write_uint_le(1U)
  buffer.write_byte(b'a')
  for _ in 0..<7 {
    buffer.write_byte(b'\x00')
  }
  assert_eq(read_global_heap_object(buffer.to_bytes(), 0UL, 1, 4, 4, 0UL), b"a")
}
