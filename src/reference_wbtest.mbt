///|
fn reference_value_bytes(dataset : Dataset) -> Array[Bytes] raise Hdf5Error {
  let references : Array[Bytes] = []
  match dataset.values {
    Structured(values) =>
      for value in values {
        match value {
          Reference(bytes) => references.push(bytes)
          _ => raise InvalidField("expected reference values")
        }
      }
    _ => raise InvalidField("expected a structured reference dataset")
  }
  references
}

///|
fn check_reference_fixture(file : Hdf5File) -> Unit raise Error {
  let references = reference_value_bytes(file.read_dataset("/object_refs"))
  assert_eq(references.length(), 6)
  let target = file.resolve("/target")
  assert_eq(
    file.resolve_object_reference(references[0]),
    Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
  )
  let group = file.resolve("/group")
  assert_eq(
    file.resolve_object_reference(references[1]),
    Some({ address: group.address, kind: Group, path: Some("/group"), }),
  )
  let child = file.resolve("/group/child")
  assert_eq(
    file.resolve_object_reference(references[2]),
    Some({ address: child.address, kind: Dataset, path: Some("/group/child"), }),
  )
  let root = file.resolve("/")
  assert_eq(
    file.resolve_object_reference(references[3]),
    Some({ address: root.address, kind: Group, path: Some("/"), }),
  )
  let datatype = file.resolve("/named_i4")
  assert_eq(datatype.kind, Datatype)
  assert_eq(
    file.resolve_object_reference(references[4]),
    Some({ address: datatype.address, kind: Datatype, path: Some("/named_i4"), }),
  )
  assert_eq(file.resolve_object_reference(references[5]), None)

  let attributes = file.attributes("/target")
  let child_ref = match
    attributes.search_by(attribute => attribute.name == "child_ref") {
    Some(index) => attributes[index]
    None => raise InvalidField("target.child_ref attribute is missing")
  }
  match child_ref.values {
    Some(Structured([Reference(bytes)])) =>
      assert_eq(
        file.resolve_object_reference(bytes),
        Some({
          address: child.address,
          kind: Dataset,
          path: Some("/group/child"),
        }),
      )
    _ => raise InvalidField("target.child_ref has an unexpected value")
  }
}

///|
fn selection_result_integer_values(
  result : DatasetSelectionResult,
) -> Array[Int64] raise Error {
  match result.values {
    Integers(values) => values
    _ => fail("expected integer region-reference values")
  }
}

///|
fn check_region_values(
  file : Hdf5File,
  reference : Bytes,
  dimensions : Array[UInt64],
  expected : Array[Int64],
) -> Unit raise Error {
  let result = file.read_region_reference(reference) catch {
    error => fail("failed to read region reference: " + error.message())
  }
  let result = match result {
    Some(result) => result
    None => fail("expected a non-null region reference")
  }
  assert_eq(result.dimensions, dimensions)
  assert_eq(selection_result_integer_values(result), expected)
}

///|
fn resolve_region_or_fail(
  file : Hdf5File,
  reference : Bytes,
  label : String,
) -> Hdf5RegionReferenceTarget raise Error {
  let result = file.resolve_region_reference(reference) catch {
    error => fail("failed to resolve " + label + ": " + error.message())
  }
  match result {
    Some(target) => target
    None => fail(label + " region reference is unexpectedly null")
  }
}

///|
fn assert_region_hyperslab(
  selection : Hdf5RegionSelection,
  start : Array[UInt64],
  stride : Array[UInt64],
  count : Array[UInt64],
  block : Array[UInt64],
) -> Unit raise Error {
  match selection {
    RegionHyperslab(actual_start, actual_stride, actual_count, actual_block) => {
      assert_eq(actual_start, start)
      assert_eq(actual_stride, stride)
      assert_eq(actual_count, count)
      assert_eq(actual_block, block)
    }
    _ => fail("expected a regular region hyperslab")
  }
}

///|
fn assert_region_blocks(
  selection : Hdf5RegionSelection,
  expected : Array[Hdf5RegionBlock],
) -> Unit raise Error {
  match selection {
    RegionBlocks(blocks) => assert_eq(blocks, expected)
    _ => fail("expected an irregular region hyperslab")
  }
}

///|
fn check_region_reference_fixture(
  file : Hdf5File,
  v108 : Bool,
) -> Unit raise Error {
  let references = reference_value_bytes(file.read_dataset("/region_refs"))
  assert_eq(references.length(), 11)
  let target = file.resolve("/target")
  let target_dataspace = file.read_dataset_metadata("/target").dataspace

  let slice = resolve_region_or_fail(file, references[0], "slice")
  assert_eq(slice.address, target.address)
  assert_eq(slice.kind, Dataset)
  assert_eq(slice.path, Some("/target"))
  assert_eq(slice.dataspace.dimensions, [6UL])
  if v108 {
    assert_region_blocks(slice.selection, [{ start: [1UL], end: [2UL], }])
  } else {
    assert_region_hyperslab(slice.selection, [1UL], [1UL], [1UL], [2UL])
  }
  check_region_values(file, references[0], [2UL], [1L, 2L])

  let regular = resolve_region_or_fail(file, references[1], "regular")
  if v108 {
    assert_region_blocks(regular.selection, [
      { start: [0UL], end: [1UL], },
      { start: [3UL], end: [4UL], },
    ])
  } else {
    assert_region_hyperslab(regular.selection, [0UL], [3UL], [2UL], [2UL])
  }
  check_region_values(file, references[1], [4UL], [0L, 1L, 3L, 4L])

  let points = resolve_region_or_fail(file, references[2], "point")
  assert_eq(points.selection, RegionPoints([[4UL], [0UL], [3UL]]))
  check_region_values(file, references[2], [3UL], [4L, 0L, 3L])

  let blocks = resolve_region_or_fail(file, references[3], "block")
  assert_region_blocks(blocks.selection, [
    { start: [0UL], end: [1UL], },
    { start: [4UL], end: [4UL], },
  ])
  check_region_values(file, references[3], [3UL], [0L, 1L, 4L])

  let all = resolve_region_or_fail(file, references[4], "all")
  assert_eq(all.selection, RegionAll)
  check_region_values(file, references[4], [6UL], [0L, 1L, 2L, 3L, 4L, 5L])

  let empty = resolve_region_or_fail(file, references[5], "empty")
  assert_eq(empty.selection, RegionEmpty)
  check_region_values(file, references[5], [0UL], [])

  let scalar = resolve_region_or_fail(file, references[6], "scalar")
  assert_eq(scalar.dataspace.kind, Scalar)
  assert_eq(scalar.dataspace.dimensions, [])
  assert_eq(scalar.selection, RegionAll)
  check_region_values(file, references[6], [], [7L])

  let matrix = resolve_region_or_fail(file, references[7], "matrix")
  assert_eq(matrix.dataspace.dimensions, [6UL, 5UL])
  if v108 {
    assert_region_blocks(matrix.selection, [
      { start: [0UL, 0UL], end: [3UL, 0UL], },
      { start: [0UL, 3UL], end: [3UL, 3UL], },
    ])
  } else {
    assert_region_hyperslab(
      matrix.selection,
      [0UL, 0UL],
      [1UL, 3UL],
      [1UL, 2UL],
      [4UL, 1UL],
    )
  }
  check_region_values(
    file,
    references[7],
    if v108 {
      [8UL]
    } else {
      [4UL, 2UL]
    },
    [0L, 3L, 5L, 8L, 10L, 13L, 15L, 18L],
  )

  let matrix_blocks = resolve_region_or_fail(
    file,
    references[8],
    "matrix block",
  )
  assert_region_blocks(matrix_blocks.selection, [
    { start: [0UL, 0UL], end: [0UL, 2UL], },
    { start: [3UL, 1UL], end: [3UL, 2UL], },
  ])
  check_region_values(file, references[8], [5UL], [0L, 1L, 2L, 16L, 17L])

  assert_true(file.resolve_region_reference(references[9]) is None)
  let dangling_error = try {
    file.resolve_region_reference(references[10]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(dangling_error is Some(InvalidField(_)))
  assert_eq(target_dataspace.dimensions, [6UL])
}

///|
fn check_region_layout_fixture(file : Hdf5File) -> Unit raise Error {
  assert_eq(file.superblock.offset_size, 4)
  assert_eq(file.superblock.length_size, 8)
  let compact = file.resolve("/compact")
  let compact_ref = reference_value_bytes(
      file.read_dataset("/object_ref_compact"),
    )[0]
  assert_eq(compact_ref.length(), 4)
  assert_eq(
    file.resolve_object_reference(compact_ref),
    Some({ address: compact.address, kind: Dataset, path: Some("/compact"), }),
  )
  let group = file.resolve("/group")
  let group_ref = reference_value_bytes(file.read_dataset("/object_ref_group"))[0]
  assert_eq(group_ref.length(), 4)
  assert_eq(
    file.resolve_object_reference(group_ref),
    Some({ address: group.address, kind: Group, path: Some("/group"), }),
  )
  let chunked = file.resolve("/chunked")
  let chunked_ref = reference_value_bytes(
      file.read_dataset("/object_ref_chunked"),
    )[0]
  assert_eq(chunked_ref.length(), 4)
  assert_eq(
    file.resolve_object_reference(chunked_ref),
    Some({ address: chunked.address, kind: Dataset, path: Some("/chunked"), }),
  )
  let null_ref = reference_value_bytes(file.read_dataset("/object_ref_null"))[0]
  assert_eq(null_ref.length(), 4)
  assert_eq(file.resolve_object_reference(null_ref), None)

  let attributes = file.attributes("/compact")
  let sibling_ref = match
    attributes.search_by(attribute => attribute.name == "sibling_ref") {
    Some(index) => attributes[index]
    None => raise InvalidField("compact.sibling_ref attribute is missing")
  }
  let contiguous = file.resolve("/contiguous")
  match sibling_ref.values {
    Some(Structured([Reference(bytes)])) =>
      assert_eq(
        file.resolve_object_reference(bytes),
        Some({
          address: contiguous.address,
          kind: Dataset,
          path: Some("/contiguous"),
        }),
      )
    _ => raise InvalidField("compact.sibling_ref has an unexpected value")
  }

  assert_eq(file.read_dataset_metadata("/compact").layout.kind(), Compact)
  assert_eq(file.read_dataset_metadata("/contiguous").layout.kind(), Contiguous)
  assert_eq(file.read_dataset_metadata("/chunked").layout.kind(), Chunked)

  let compact = reference_value_bytes(file.read_dataset("/ref_compact"))
  let contiguous = reference_value_bytes(file.read_dataset("/ref_contiguous"))
  let chunked = reference_value_bytes(file.read_dataset("/ref_chunked"))
  assert_eq(compact.length(), 1)
  assert_eq(contiguous.length(), 1)
  assert_eq(chunked.length(), 1)
  assert_eq(compact[0].length(), 8)
  assert_eq(contiguous[0].length(), 8)
  assert_eq(chunked[0].length(), 8)

  let compact_target = resolve_region_or_fail(file, compact[0], "compact")
  assert_eq(compact_target.path, Some("/compact"))
  assert_eq(
    compact_target.selection,
    RegionHyperslab([1UL], [2UL], [2UL], [1UL]),
  )
  check_region_values(file, compact[0], [2UL], [1L, 3L])

  let contiguous_target = resolve_region_or_fail(
    file,
    contiguous[0],
    "contiguous",
  )
  assert_eq(contiguous_target.path, Some("/contiguous"))
  assert_eq(contiguous_target.selection, RegionPoints([[4UL], [0UL], [2UL]]))
  check_region_values(file, contiguous[0], [3UL], [4L, 0L, 2L])

  let chunked_target = resolve_region_or_fail(file, chunked[0], "chunked")
  assert_eq(chunked_target.path, Some("/chunked"))
  assert_eq(
    chunked_target.selection,
    RegionHyperslab([0UL], [3UL], [2UL], [2UL]),
  )
  check_region_values(file, chunked[0], [4UL], [0L, 1L, 3L, 4L])
}

///|
test "object references resolve from in-memory files" {
  for path in ["testdata/references.h5", "testdata/references_v108.h5"] {
    check_reference_fixture(Hdf5File::parse(@fs.read_file_to_bytes(path)))
  }
}

///|
test "region references resolve and read from in-memory files" {
  let latest = Hdf5File::parse(@fs.read_file_to_bytes("testdata/references.h5"))
  check_region_reference_fixture(latest, false)
  let v108 = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/references_v108.h5"),
  )
  check_region_reference_fixture(v108, true)
}

///|
test "region references resolve and read from random-access sources" {
  let latest_bytes = @fs.read_file_to_bytes("testdata/references.h5")
  let latest_sources : Array[&Hdf5Source] = [latest_bytes]
  check_region_reference_fixture(
    Hdf5File::parse_source(latest_sources[0]),
    false,
  )
  let v108_bytes = @fs.read_file_to_bytes("testdata/references_v108.h5")
  let v108_sources : Array[&Hdf5Source] = [v108_bytes]
  check_region_reference_fixture(Hdf5File::parse_source(v108_sources[0]), true)
}

///|
test "four-byte region references read compact contiguous and filtered data" {
  let bytes = @fs.read_file_to_bytes("testdata/references_layouts_32.h5")
  check_region_layout_fixture(Hdf5File::parse(bytes))
  let sources : Array[&Hdf5Source] = [bytes]
  check_region_layout_fixture(Hdf5File::parse_source(sources[0]))
}

///|
test "object references resolve from random-access sources" {
  for path in ["testdata/references.h5", "testdata/references_v108.h5"] {
    let bytes = @fs.read_file_to_bytes(path)
    let sources : Array[&Hdf5Source] = [bytes]
    check_reference_fixture(Hdf5File::parse_source(sources[0]))
  }
}

///|
test "null malformed dangling and region references are distinguished" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/references.h5"))
  let null_reference = b"\x00\x00\x00\x00\x00\x00\x00\x00"
  assert_eq(file.resolve_object_reference(null_reference), None)

  let short_error = try {
    file.resolve_object_reference(b"\x00\x00\x00\x00\x00\x00\x00") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(short_error is Some(InvalidField(_)))

  let dangling = reference_value_bytes(file.read_dataset("/dangling_refs"))
  let dangling_error = try {
    file.resolve_object_reference(dangling[0]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(dangling_error is Some(InvalidField(_)))

  let region = reference_value_bytes(file.read_dataset("/region_refs"))
  assert_eq(region[0].length(), 12)
  let region_error = try {
    file.resolve_object_reference(region[0]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    region_error,
    Some(InvalidArgument("region references require resolve_region_reference")),
  )
}
