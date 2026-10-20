///|
test "writer round-trips groups datasets and attributes" {
  let writer = Hdf5Writer::new()
  writer.add_group("/science/results")
  writer.add_dataset(
    "/science/results/counts",
    [2UL, 2UL],
    Int64Values([1L, -2L, 3L, 4L]),
  )
  writer.add_dataset("/weight", [], Float64Values([1.5]))
  writer.add_dataset("/labels", [2UL], FixedStringValues(["moon", "数据"], 8))
  writer.add_dataset("/empty", [0UL, 3UL], UInt64Values([]))
  writer.add_attribute("/", "title", [], FixedStringValues(["writer demo"], 12))
  writer.add_attribute("/science", "version", [], UInt64Values([1UL]))
  writer.add_attribute(
    "/science/results/counts",
    "unit",
    [],
    FixedStringValues(["items"], 8),
  )

  let file = Hdf5File::parse(writer.encode())
  assert_eq(file.superblock.version, 3)
  assert_eq(file.walk().map(node => node.path), [
    "/science", "/science/results", "/science/results/counts", "/weight", "/labels",
    "/empty",
  ])
  assert_eq(integer_values(file.open_dataset("/science/results/counts")), [
    1L, -2L, 3L, 4L,
  ])
  assert_eq(float_values(file.open_dataset("/weight")), [1.5])
  assert_eq(string_values(file.open_dataset("/labels")), ["moon", "数据"])
  assert_eq(unsigned_values(file.open_dataset("/empty")), [])
  match file.attributes("/")[0].values {
    Some(Strings(values)) => assert_eq(values, ["writer demo"])
    _ => abort("expected root string attribute")
  }
  match file.attributes("/science")[0].values {
    Some(UnsignedIntegers(values)) => assert_eq(values, [1UL])
    _ => abort("expected group integer attribute")
  }
  match file.attributes("/science/results/counts")[0].values {
    Some(Strings(values)) => assert_eq(values, ["items"])
    _ => abort("expected dataset string attribute")
  }
}

///|
test "writer rejects invalid paths shapes strings and duplicates" {
  let writer = Hdf5Writer::new()
  let relative = try {
    writer.add_group("relative")
    None
  } catch {
    error => Some(error)
  }
  assert_eq(relative, Some(InvalidField("writer paths must be absolute")))

  let shape = try {
    writer.add_dataset("/bad", [2UL], Int64Values([1L]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    shape,
    Some(InvalidField("writer value count does not match dataspace")),
  )

  let text = try {
    writer.add_dataset("/bad", [1UL], FixedStringValues(["long"], 3))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(text, Some(InvalidField("fixed string value exceeds its width")))

  let dimension = try {
    writer.add_dataset("/bad", [0UL, 0x80000000UL], UInt64Values([]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(dimension, Some(InvalidField("writer dimension is too large")))

  writer.add_dataset("/ok", [], Int64Values([1L]))
  let duplicate = try {
    writer.add_dataset("/ok", [], Int64Values([2L]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(duplicate, Some(InvalidField("writer object path already exists")))
}

///|
test "writer snapshots caller-owned value and dimension arrays" {
  let writer = Hdf5Writer::new()
  let dimensions = [2UL]
  let values = [10L, 20L]
  writer.add_dataset("/values", dimensions, Int64Values(values))
  dimensions[0] = 1UL
  values[0] = 99L
  let file = Hdf5File::parse(writer.encode())
  let dataset = file.open_dataset("/values")
  assert_eq(dataset.dataspace.dimensions, [2UL])
  assert_eq(integer_values(dataset), [10L, 20L])
}

///|
test "writer emits multi-chunk fills and filter pipelines" {
  let writer = Hdf5Writer::new()
  writer.add_chunked_dataset(
    "/raw",
    [3UL, 5UL],
    [2U, 3U],
    UInt64Values([
      0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 10UL, 11UL, 12UL, 13UL, 14UL,
    ]),
  )
  let sparse = Array::make(35, -7L)
  sparse[0] = 42L
  sparse[17] = 99L
  sparse[34] = -2L
  writer.add_chunked_dataset(
    "/filtered",
    [5UL, 7UL],
    [2U, 3U],
    Int64Values(sparse),
    fill_value=Int64Values([-7L]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  let file = Hdf5File::parse(writer.encode())
  let raw = file.open_dataset("/raw")
  assert_true(raw.layout.chunk_index is FixedArray(_))
  assert_eq(unsigned_values(raw), [
    0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 10UL, 11UL, 12UL, 13UL, 14UL,
  ])
  let filtered = file.open_dataset("/filtered")
  assert_true(filtered.layout.chunk_index is FixedArray(_))
  assert_eq(integer_values(filtered), sparse)
  let selected = file.read_dataset_selection(
    "/filtered",
    Points([[4UL, 6UL], [0UL, 0UL], [2UL, 3UL], [1UL, 1UL]]),
  )
  match selected.values {
    Integers(values) => assert_eq(values, [-2L, 42L, 99L, -7L])
    _ => abort("expected filtered integer selection")
  }
}

///|
test "writer validates chunk metadata filters and fill types" {
  let writer = Hdf5Writer::new()
  let cases : Array[Hdf5Error?] = [
    try {
      writer.add_chunked_dataset(
        "/rank",
        [2UL],
        [1U, 1U],
        Int64Values([1L, 2L]),
      )
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset("/empty", [0UL], [1U], Int64Values([]))
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset(
        "/fill",
        [1UL],
        [1U],
        Int64Values([1L]),
        fill_value=UInt64Values([0UL]),
      )
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset("/filters", [1UL], [1U], Int64Values([1L]), filters=[
        Deflate(6),
        Shuffle,
      ])
      None
    } catch {
      error => Some(error)
    },
    try {
      writer.add_chunked_dataset(
        "/many",
        [1025UL],
        [1U],
        Int64Values(Array::make(1025, 0L)),
      )
      None
    } catch {
      error => Some(error)
    },
  ]
  assert_eq(cases, [
    Some(InvalidField("chunk dimensions must match a simple dataspace")),
    Some(InvalidField("chunk dimension is invalid")),
    Some(InvalidField("fill value must be one value of the dataset type")),
    Some(InvalidField("shuffle must be the first writer filter")),
    Some(InvalidField("writer fixed-array index exceeds 1024 chunks")),
  ])
}
