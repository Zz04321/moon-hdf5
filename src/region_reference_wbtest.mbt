///|
fn region_test_dataspace(dimensions : Array[UInt64]) -> Dataspace {
  { kind: Simple, dimensions, maximum_dimensions: dimensions.copy(), }
}

///|
test "version-2 regular and irregular hyperslabs decode" {
  let regular = @buffer.Buffer::Buffer(size_hint=80)
  regular.write_uint_le(2U)
  regular.write_uint_le(2U)
  regular.write_byte(b'\x01')
  regular.write_uint_le(0U)
  regular.write_uint_le(2U)
  for value in [0UL, 2UL, 2UL, 1UL, 0UL, 3UL, 2UL, 1UL] {
    regular.write_uint64_le(value)
  }
  assert_eq(
    parse_region_selection(
      Reader::new(regular.to_bytes()),
      region_test_dataspace([10UL, 10UL]),
    ),
    RegionHyperslab([0UL, 0UL], [2UL, 3UL], [2UL, 2UL], [1UL, 1UL]),
  )

  let irregular = @buffer.Buffer::Buffer(size_hint=64)
  irregular.write_uint_le(2U)
  irregular.write_uint_le(2U)
  irregular.write_byte(b'\x00')
  irregular.write_uint_le(0U)
  irregular.write_uint_le(1U)
  irregular.write_uint64_le(2UL)
  for value in [0UL, 1UL, 4UL, 4UL] {
    irregular.write_uint64_le(value)
  }
  assert_eq(
    parse_region_selection(
      Reader::new(irregular.to_bytes()),
      region_test_dataspace([6UL]),
    ),
    RegionBlocks([{ start: [0UL], end: [1UL], }, { start: [4UL], end: [4UL], }]),
  )
}

///|
test "region selections reject unsupported versions flags and trailing data" {
  let unsupported = @buffer.Buffer::Buffer(size_hint=16)
  unsupported.write_uint_le(2U)
  unsupported.write_uint_le(4U)
  let unsupported_error = try {
    parse_region_selection(
      Reader::new(unsupported.to_bytes()),
      region_test_dataspace([6UL]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    unsupported_error,
    Some(UnsupportedFeature("unsupported region hyperslab version")),
  )

  let bad_flags = @buffer.Buffer::Buffer(size_hint=32)
  bad_flags.write_uint_le(2U)
  bad_flags.write_uint_le(2U)
  bad_flags.write_byte(b'\x02')
  bad_flags.write_uint_le(0U)
  bad_flags.write_uint_le(1U)
  let flags_error = try {
    parse_region_selection(
      Reader::new(bad_flags.to_bytes()),
      region_test_dataspace([6UL]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    flags_error,
    Some(InvalidField("unsupported region hyperslab flags")),
  )

  let trailing = @buffer.Buffer::Buffer(size_hint=16)
  trailing.write_uint_le(3U)
  trailing.write_uint_le(1U)
  trailing.write_uint_le(0U)
  trailing.write_uint_le(0U)
  trailing.write_byte(b'\x00')
  let trailing_error = try {
    parse_region_selection(
      Reader::new(trailing.to_bytes()),
      region_test_dataspace([6UL]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    trailing_error,
    Some(InvalidField("region selection has trailing bytes")),
  )
}

///|
test "region selections validate rank and dataset bounds" {
  let wrong_rank = @buffer.Buffer::Buffer(size_hint=80)
  wrong_rank.write_uint_le(2U)
  wrong_rank.write_uint_le(2U)
  wrong_rank.write_byte(b'\x01')
  wrong_rank.write_uint_le(0U)
  wrong_rank.write_uint_le(2U)
  for value in [0UL, 0UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL] {
    wrong_rank.write_uint64_le(value)
  }
  let rank_error = try {
    parse_region_selection(
      Reader::new(wrong_rank.to_bytes()),
      region_test_dataspace([6UL]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    rank_error,
    Some(InvalidField("region hyperslab rank does not match the dataset")),
  )

  let out_of_bounds = @buffer.Buffer::Buffer(size_hint=80)
  out_of_bounds.write_uint_le(2U)
  out_of_bounds.write_uint_le(2U)
  out_of_bounds.write_byte(b'\x01')
  out_of_bounds.write_uint_le(0U)
  out_of_bounds.write_uint_le(1U)
  for value in [5UL, 1UL, 1UL, 2UL] {
    out_of_bounds.write_uint64_le(value)
  }
  let bounds_error = try {
    parse_region_selection(
      Reader::new(out_of_bounds.to_bytes()),
      region_test_dataspace([6UL]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    bounds_error,
    Some(InvalidField("region hyperslab extends outside the dataset")),
  )
}

///|
test "region selections reject malformed points ranks and blocks" {
  let unsupported_points = @buffer.Buffer::Buffer(size_hint=8)
  unsupported_points.write_uint_le(1U)
  unsupported_points.write_uint_le(3U)
  let points_error = try {
    parse_region_selection(
      Reader::new(unsupported_points.to_bytes()),
      region_test_dataspace([6UL]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    points_error,
    Some(UnsupportedFeature("unsupported region point-selection version")),
  )

  let huge_rank = @buffer.Buffer::Buffer(size_hint=16)
  huge_rank.write_uint_le(2U)
  huge_rank.write_uint_le(3U)
  huge_rank.write_byte(b'\x01')
  huge_rank.write_byte(b'\x08')
  huge_rank.write_uint_le(0xFFFFFFFFU)
  let rank_error = try {
    parse_region_selection(
      Reader::new(huge_rank.to_bytes()),
      region_test_dataspace([6UL]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    rank_error,
    Some(InvalidField("region selection rank is outside 0..32")),
  )

  let point_oob = @buffer.Buffer::Buffer(size_hint=32)
  point_oob.write_uint_le(1U)
  point_oob.write_uint_le(1U)
  point_oob.write_uint_le(0U)
  point_oob.write_uint_le(0U)
  point_oob.write_uint_le(1U)
  point_oob.write_uint_le(1U)
  point_oob.write_uint_le(6U)
  let point_error = try {
    parse_region_selection(
      Reader::new(point_oob.to_bytes()),
      region_test_dataspace([6UL]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    point_error,
    Some(InvalidField("region point is outside the dataset")),
  )

  let block_oob = @buffer.Buffer::Buffer(size_hint=32)
  block_oob.write_uint_le(2U)
  block_oob.write_uint_le(1U)
  block_oob.write_uint_le(0U)
  block_oob.write_uint_le(24U)
  block_oob.write_uint_le(1U)
  block_oob.write_uint_le(1U)
  block_oob.write_uint_le(0U)
  block_oob.write_uint_le(6U)
  let block_error = try {
    parse_region_selection(
      Reader::new(block_oob.to_bytes()),
      region_test_dataspace([6UL]),
    )
    |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    block_error,
    Some(InvalidField("region block is outside the dataset")),
  )
}

///|
test "region reference inputs reject wrong lengths and heap identifiers" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/references.h5"))
  let wrong_length = try {
    file.resolve_region_reference(b"\x00\x00\x00") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(wrong_length is Some(InvalidField(_)))

  let zero : Array[Byte] = []
  for _ in 0..<12 {
    zero.push(b'\x00')
  }
  assert_true(file.resolve_region_reference(Bytes::from_array(zero)) is None)

  let invalid_heap = @buffer.Buffer::Buffer(size_hint=12)
  invalid_heap.write_uint64_le(1UL)
  invalid_heap.write_uint_le(1U)
  let heap_error = try {
    file.resolve_region_reference(invalid_heap.to_bytes()) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(heap_error is Some(InvalidField(_)))
}

///|
test "region references reject a truncated global heap collection" {
  let bytes = @fs.read_file_to_bytes("testdata/references.h5")
  let file = Hdf5File::parse(bytes)
  let references = reference_value_bytes(file.read_dataset("/region_refs"))
  let reference_reader = Reader::new(references[0])
  let heap_address = reference_reader.read_uint_le(file.superblock.offset_size)
  let end = heap_address.to_int() + 16
  assert_true(end < bytes.length())
  let truncated : Array[Byte] = []
  for index in 0..<end {
    truncated.push(bytes[index])
  }
  let truncated_file = Hdf5File::parse(Bytes::from_array(truncated))
  let error = try {
    truncated_file.resolve_region_reference(references[0]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidField("invalid global heap collection size")))
}
