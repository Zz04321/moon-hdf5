///|
test "public high-level API reads a file" {
  let file = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/basic_v0.h5"),
  )
  let node = file.resolve("/group/values")
  assert_eq(node.kind, @moon-hdf5.Dataset)
  let metadata = file.open_dataset_metadata("/group/values")
  assert_eq(metadata.dataspace.dimensions, [10UL])
  assert_eq(metadata.datatype.size, 4)
  assert_eq(metadata.layout.layout_class, 1)
  let dataset = file.open_dataset("/group/values")
  match dataset.values {
    Integers(values) => assert_eq(values.length(), 10)
    _ => abort("expected integer values")
  }
  let attributes = file.attributes("/")
  assert_eq(attributes.length(), 1)
  assert_eq(attributes[0].name, "title")
}

///|
test "public API reads dataset metadata without values" {
  let file = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/basic_v0.h5"),
  )
  let metadata = file.open_dataset_metadata("/matrix")
  assert_eq(metadata.dataspace.dimensions, [2UL, 3UL])
  assert_eq(metadata.datatype.size, 4)
  assert_eq(metadata.layout.layout_class, 1)
  let error = try {
    file.open_dataset_metadata("/group") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(
    error is Some(@moon-hdf5.InvalidField("object is not a dataset: /group")),
  )
}

///|
test "public API reads hyperslabs and points from contiguous datasets" {
  let file = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/basic_v0.h5"),
  )
  let slab = file.read_dataset_selection(
    "/matrix",
    Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 2UL]),
  )
  assert_eq(slab.dimensions, [2UL, 2UL])
  match slab.values {
    Floats(values) => assert_eq(values, [0.0, 2.0, 3.0, 5.0])
    _ => abort("expected floating-point hyperslab")
  }
  let points = file.read_dataset_selection(
    "/matrix",
    Points([[1UL, 2UL], [0UL, 1UL], [1UL, 2UL]]),
  )
  assert_eq(points.dimensions, [3UL])
  match points.values {
    Floats(values) => assert_eq(values, [5.0, 1.0, 5.0])
    _ => abort("expected floating-point points")
  }
}

///|
test "dataset selections validate bounds and allow empty results" {
  let file = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/basic_v0.h5"),
  )
  let empty = file.read_dataset_selection(
    "/matrix",
    Hyperslab([2UL, 0UL], [0UL, 2UL], [1UL, 1UL]),
  )
  assert_eq(empty.dimensions, [0UL, 2UL])
  match empty.values {
    Floats(values) => assert_eq(values, [])
    _ => abort("expected empty floating-point selection")
  }
  let invalid : Array[@moon-hdf5.DatasetSelection] = [
    Hyperslab([0UL], [1UL], [1UL]),
    Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 0UL]),
    Hyperslab([0UL, 1UL], [2UL, 2UL], [1UL, 2UL]),
    Points([[2UL, 0UL]]),
  ]
  for selection in invalid {
    let error = try {
      file.read_dataset_selection("/matrix", selection) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(@moon-hdf5.InvalidField(_)))
  }
}

///|
test "dataset selections handle scalar empty and null dataspaces" {
  let file = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/contiguous_fill_v3.h5"),
  )
  let scalar = file.read_dataset_selection("/other/scalar", Points([[]]))
  match scalar.values {
    Integers(values) => assert_eq(values, [-2L])
    _ => abort("expected scalar integer selection")
  }
  let empty = file.read_dataset_selection(
    "/other/empty",
    Hyperslab([0UL, 0UL], [0UL, 2UL], [1UL, 1UL]),
  )
  assert_eq(empty.dimensions, [0UL, 2UL])
  match empty.values {
    Integers(values) => assert_eq(values, [])
    _ => abort("expected empty integer selection")
  }
  let null = file.read_dataset_selection("/other/null", Points([]))
  assert_eq(null.dimensions, [0UL])
  match null.values {
    Integers(values) => assert_eq(values, [])
    _ => abort("expected null integer selection")
  }
  let error = try {
    file.read_dataset_selection("/other/null", Points([[]])) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(@moon-hdf5.InvalidField(_)))
}

///|
test "dataset selections cover compact and contiguous fills" {
  let compact_file = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/interop_earliest.h5"),
  )
  let compact = compact_file.read_dataset_selection(
    "/data/compact",
    Hyperslab([1UL], [1UL], [1UL]),
  )
  match compact.values {
    Integers(values) => assert_eq(values, [40L])
    _ => abort("expected compact integer selection")
  }
  let fill_file = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/contiguous_fill_v3.h5"),
  )
  let fill = fill_file.read_dataset_selection("/signed", Points([[2UL], [0UL]]))
  match fill.values {
    Integers(values) => assert_eq(values, [-7L, -7L])
    _ => abort("expected filled integer selection")
  }
}

///|
test "chunked selections cover every index filters fills and point order" {
  let earliest = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/interop_earliest.h5"),
  )
  let slab = earliest.read_dataset_selection(
    "/data/chunked_raw",
    Hyperslab([0UL, 1UL], [3UL, 3UL], [2UL, 2UL]),
  )
  assert_eq(slab.dimensions, [3UL, 3UL])
  match slab.values {
    Integers(values) =>
      assert_eq(values, [1L, 3L, 5L, 15L, 17L, 19L, 29L, 31L, 33L])
    _ => abort("expected v1 B-tree integer selection")
  }
  let modern = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/interop_v110.h5"),
  )
  let cases : Array[(String, Array[Array[UInt64]], Array[Int64])] = [
    (
      "/data/chunked_filtered",
      [[0UL, 0UL], [4UL, 6UL], [2UL, 2UL], [0UL, 0UL]],
      [100L, 134L, 116L, 100L],
    ),
    (
      "/data/unlimited_one_filtered",
      [[4UL, 6UL], [0UL, 0UL], [2UL, 3UL], [4UL, 6UL]],
      [34L, 0L, 17L, 34L],
    ),
    (
      "/data/unlimited_two_filtered",
      [[9UL, 9UL], [0UL, 0UL], [4UL, 7UL], [9UL, 9UL]],
      [99L, 0L, 47L, 99L],
    ),
    (
      "/data/single_filtered",
      [[2UL, 3UL], [0UL, 0UL], [1UL, 2UL], [2UL, 3UL]],
      [11L, 0L, 6L, 11L],
    ),
    (
      "/data/chunked_sparse",
      [[0UL, 0UL], [1UL, 2UL], [6UL, 8UL], [3UL, 4UL], [1UL, 2UL]],
      [-7L, 42L, 99L, -7L, 42L],
    ),
  ]
  for (path, points, expected) in cases {
    let selected = modern.read_dataset_selection(path, Points(points))
    assert_eq(selected.dimensions, [expected.length().to_uint64()])
    match selected.values {
      Integers(values) => assert_eq(values, expected)
      _ => abort("expected chunked integer selection")
    }
  }
  let filtered = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/filters_v2.h5"),
  )
  let empty = filtered.read_dataset_selection(
    "/special/corrupt",
    Hyperslab([0UL], [0UL], [1UL]),
  )
  match empty.values {
    Integers(values) => assert_eq(values, [])
    _ => abort("expected empty filtered selection")
  }
}
