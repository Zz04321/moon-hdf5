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
  assert_eq(integer_values(file.read_dataset("/science/results/counts")), [
    1L, -2L, 3L, 4L,
  ])
  assert_eq(float_values(file.read_dataset("/weight")), [1.5])
  assert_eq(string_values(file.read_dataset("/labels")), ["moon", "数据"])
  assert_eq(unsigned_values(file.read_dataset("/empty")), [])
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
test "writer round-trips absolute relative and dangling soft links" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/experiment/counts", [3UL], Int64Values([1L, 2L, 3L]))
  writer.add_group("/experiment/archive")
  writer.add_soft_link("/experiment", "latest", "/experiment/counts")
  writer.add_soft_link("/experiment", "relative", "counts")
  writer.add_soft_link("/experiment", "archive_link", "archive")
  writer.add_soft_link("/experiment", "missing", "/does/not/exist")

  let file = Hdf5File::parse(writer.encode())
  assert_eq(file.resolve("/experiment/latest").path, "/experiment/counts")
  assert_eq(file.resolve("/experiment/latest").kind, Dataset)
  assert_eq(integer_values(file.read_dataset("/experiment/latest")), [
    1L, 2L, 3L,
  ])
  assert_eq(file.resolve("/experiment/relative").path, "/experiment/counts")
  assert_eq(
    file.resolve("/experiment/archive_link").path,
    "/experiment/archive",
  )
  assert_eq(file.resolve("/experiment/archive_link").kind, Group)
  let dangling = try {
    file.resolve("/experiment/missing") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(dangling, Some(InvalidField("path component not found: does")))
}

///|
test "writer round-trips external links" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/experiment/counts", [3UL], Int64Values([1L, 2L, 3L]))
  writer.add_external_link(
    "/experiment", "external", "external-target.h5", "/experiment/counts",
  )
  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  let header = file.object_header_at(file.resolve("/experiment").address)
  let links = file.links(header)
  let index = links.search_by(link => link.name == "external").unwrap()
  assert_eq(links[index].cache_type, 64U)
  match links[index].link_target {
    Some(target) => assert_eq(target, "external-target.h5")
    None => abort("expected external link target file")
  }
  match links[index].external_path {
    Some(path) => assert_eq(path, "/experiment/counts")
    None => abort("expected external link target path")
  }
  assert_true(
    file
    .walk()
    .any(node => node.path == "/experiment/external" && node.kind == Unknown),
  )
  let unresolved = try {
    file.resolve("/experiment/external") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    unresolved,
    Some(
      UnsupportedFeature(
        "external links are not resolved: external-target.h5:/experiment/counts",
      ),
    ),
  )
}

///|
test "writer rejects invalid and conflicting external links" {
  let writer = Hdf5Writer::new()
  let empty_name = try {
    writer.add_external_link("/", "", "target.h5", "/target")
    None
  } catch {
    error => Some(error)
  }
  let slash_name = try {
    writer.add_external_link("/", "bad/name", "target.h5", "/target")
    None
  } catch {
    error => Some(error)
  }
  let empty_file = try {
    writer.add_external_link("/", "link", "", "/target")
    None
  } catch {
    error => Some(error)
  }
  let nul_file = try {
    writer.add_external_link("/", "link", "bad\u{0}file", "/target")
    None
  } catch {
    error => Some(error)
  }
  let empty_path = try {
    writer.add_external_link("/", "link", "target.h5", "")
    None
  } catch {
    error => Some(error)
  }
  let nul_path = try {
    writer.add_external_link("/", "link", "target.h5", "/bad\u{0}path")
    None
  } catch {
    error => Some(error)
  }
  let too_long = try {
    writer.add_external_link("/", "link", "target.h5", "p".repeat(65521))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(empty_name, Some(InvalidArgument("external link name is empty")))
  assert_eq(
    slash_name,
    Some(InvalidArgument("external link names cannot contain '/'")),
  )
  assert_eq(
    empty_file,
    Some(InvalidArgument("external link target file is empty")),
  )
  assert_eq(
    nul_file,
    Some(InvalidArgument("external link target file cannot contain NUL")),
  )
  assert_eq(
    empty_path,
    Some(InvalidArgument("external link target path is empty")),
  )
  assert_eq(
    nul_path,
    Some(InvalidArgument("external link target path cannot contain NUL")),
  )
  assert_eq(
    too_long,
    Some(InvalidArgument("external link message is too long")),
  )

  writer.add_dataset("/real", [], Int64Values([1L]))
  let occupied = try {
    writer.add_external_link("/", "real", "target.h5", "/target")
    None
  } catch {
    error => Some(error)
  }
  writer.add_external_link("/", "link", "target.h5", "/target")
  let duplicate = try {
    writer.add_external_link("/", "link", "other.h5", "/other")
    None
  } catch {
    error => Some(error)
  }
  let group_conflict = try {
    writer.add_group("/link/child")
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    occupied,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    duplicate,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    group_conflict,
    Some(InvalidArgument("writer path conflicts with an external link")),
  )
}

///|
test "writer rejects invalid and conflicting soft links" {
  let writer = Hdf5Writer::new()
  let empty_name = try {
    writer.add_soft_link("/", "", "/target")
    None
  } catch {
    error => Some(error)
  }
  let slash_name = try {
    writer.add_soft_link("/", "bad/name", "/target")
    None
  } catch {
    error => Some(error)
  }
  let dot_name = try {
    writer.add_soft_link("/", ".", "/target")
    None
  } catch {
    error => Some(error)
  }
  let nul_name = try {
    writer.add_soft_link("/", "bad\u{0}name", "/target")
    None
  } catch {
    error => Some(error)
  }
  let empty_target = try {
    writer.add_soft_link("/", "link", "")
    None
  } catch {
    error => Some(error)
  }
  let nul_target = try {
    writer.add_soft_link("/", "link", "/bad\u{0}target")
    None
  } catch {
    error => Some(error)
  }
  let too_long_target = try {
    writer.add_soft_link("/", "link", "t".repeat(65524))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(empty_name, Some(InvalidArgument("soft link name is empty")))
  assert_eq(
    slash_name,
    Some(InvalidArgument("soft link names cannot contain '/'")),
  )
  assert_eq(
    dot_name,
    Some(InvalidArgument("soft link names cannot be dot segments")),
  )
  assert_eq(
    nul_name,
    Some(InvalidArgument("soft link names cannot contain NUL")),
  )
  assert_eq(empty_target, Some(InvalidArgument("soft link target is empty")))
  assert_eq(
    nul_target,
    Some(InvalidArgument("soft link targets cannot contain NUL")),
  )
  assert_eq(
    too_long_target,
    Some(InvalidArgument("soft link message is too long")),
  )

  writer.add_dataset("/real", [], Int64Values([1L]))
  let occupied = try {
    writer.add_soft_link("/", "real", "/other")
    None
  } catch {
    error => Some(error)
  }
  writer.add_soft_link("/", "link", "/real")
  let duplicate = try {
    writer.add_soft_link("/", "link", "/real")
    None
  } catch {
    error => Some(error)
  }
  let dataset_conflict = try {
    writer.add_dataset("/link", [], Int64Values([2L]))
    None
  } catch {
    error => Some(error)
  }
  let group_conflict = try {
    writer.add_group("/link/child")
    None
  } catch {
    error => Some(error)
  }
  let parent_conflict = try {
    writer.add_soft_link("/link", "child", "/real")
    None
  } catch {
    error => Some(error)
  }
  let dataset_parent = try {
    writer.add_soft_link("/real", "child", "/real")
    None
  } catch {
    error => Some(error)
  }
  let attribute_target = try {
    writer.add_attribute("/link", "unit", [], UInt64Values([1UL]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    occupied,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    duplicate,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    dataset_conflict,
    Some(InvalidArgument("writer object path already exists")),
  )
  assert_eq(
    group_conflict,
    Some(InvalidArgument("writer path conflicts with a soft link")),
  )
  assert_eq(
    parent_conflict,
    Some(InvalidArgument("writer path conflicts with a soft link")),
  )
  assert_eq(
    dataset_parent,
    Some(InvalidArgument("writer path conflicts with a dataset")),
  )
  assert_eq(
    attribute_target,
    Some(InvalidArgument("attribute target does not exist")),
  )
}

///|
test "writer continues attribute and link object headers" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/many_attributes", [], UInt64Values([0UL]))
  for index in 0..<2500 {
    writer.add_attribute(
      "/many_attributes",
      "attr_" + index.to_string(),
      [],
      UInt64Values([index.to_uint64()]),
    )
  }
  writer.add_group("/many_links")
  for index in 0..<2500 {
    writer.add_soft_link(
      "/many_links",
      "link_" + index.to_string(),
      "/many_attributes",
    )
  }
  let bytes = writer.encode()
  let file = Hdf5File::parse(bytes)
  let attributes = file.attributes("/many_attributes")
  assert_eq(attributes.length(), 2500)
  match attributes[0].values {
    Some(UnsignedIntegers(values)) => assert_eq(values, [0UL])
    _ => abort("expected first continued attribute")
  }
  match attributes[2499].values {
    Some(UnsignedIntegers(values)) => assert_eq(values, [2499UL])
    _ => abort("expected last continued attribute")
  }
  let header = parse_object_header(
    bytes,
    file.resolve("/many_attributes").address,
    8,
    8,
    0UL,
  )
  assert_true(
    header.messages.any(message => message.msg_type == msg_object_continuation),
  )
  assert_eq(file.resolve("/many_links/link_2499").path, "/many_attributes")
  let links_header = parse_object_header(
    bytes,
    file.resolve("/many_links").address,
    8,
    8,
    0UL,
  )
  assert_true(
    links_header.messages.any(message => {
      message.msg_type == msg_object_continuation
    }),
  )
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
  assert_eq(relative, Some(InvalidArgument("writer paths must be absolute")))

  let shape = try {
    writer.add_dataset("/bad", [2UL], Int64Values([1L]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    shape,
    Some(InvalidArgument("writer value count does not match dataspace")),
  )

  let text = try {
    writer.add_dataset("/bad", [1UL], FixedStringValues(["long"], 3))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(text, Some(InvalidArgument("fixed string value exceeds its width")))

  let dimension = try {
    writer.add_dataset("/bad", [0UL, 0x80000000UL], UInt64Values([]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(dimension, Some(InvalidArgument("writer dimension is too large")))

  writer.add_dataset("/ok", [], Int64Values([1L]))
  let duplicate = try {
    writer.add_dataset("/ok", [], Int64Values([2L]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    duplicate,
    Some(InvalidArgument("writer object path already exists")),
  )
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
  let dataset = file.read_dataset("/values")
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
  let raw = file.read_dataset("/raw")
  assert_true(raw.layout.chunk_index is FixedArray(_))
  assert_eq(unsigned_values(raw), [
    0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 10UL, 11UL, 12UL, 13UL, 14UL,
  ])
  let filtered = file.read_dataset("/filtered")
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
  ]
  assert_eq(cases, [
    Some(InvalidArgument("chunk dimensions must match a simple dataspace")),
    Some(InvalidArgument("chunk dimension is invalid")),
    Some(InvalidArgument("fill value must be one value of the dataset type")),
    Some(InvalidArgument("shuffle must be the first writer filter")),
  ])
}

///|
test "writer fixed arrays handle the page boundary" {
  let writer = Hdf5Writer::new()
  for (path, count) in [("/exact", 1024), ("/many", 1025)] {
    let values = Array::make(count, 0L)
    values[0] = 7L
    values[count - 1] = 9L
    writer.add_chunked_dataset(
      path,
      [count.to_uint64()],
      [1U],
      Int64Values(values),
    )
  }
  let file = Hdf5File::parse(writer.encode())
  for (path, count) in [("/exact", 1024), ("/many", 1025)] {
    let selected = file.read_dataset_selection(
      path,
      Points([[count.to_uint64() - 1UL], [512UL], [0UL]]),
    )
    match selected.values {
      Integers(items) => assert_eq(items, [9L, 0L, 7L])
      _ => fail("expected fixed-array values across the page boundary")
    }
  }
}

///|
test "writer appends resizes and fills extensible datasets" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/series",
    [0UL, 3UL],
    [None, Some(4UL)],
    [2U, 2U],
    Int64Values([]),
    fill_value=Int64Values([-1L]),
  )
  writer.append_dataset(
    "/series",
    [2UL, 3UL],
    Int64Values([1L, 2L, 3L, 4L, 5L, 6L]),
  )
  writer.append_dataset("/series", [1UL, 3UL], Int64Values([7L, 8L, 9L]))
  writer.resize_dataset("/series", [5UL, 3UL])
  let grown_file = Hdf5File::parse(writer.encode())
  let grown = grown_file.read_dataset("/series")
  assert_eq(grown.dataspace.dimensions, [5UL, 3UL])
  assert_eq(grown.dataspace.maximum_dimensions, [0xFFFFFFFFFFFFFFFFUL, 4UL])
  assert_true(grown.layout.chunk_index is ExtensibleArray(_))
  assert_eq(integer_values(grown), [
    1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, -1L, -1L, -1L, -1L, -1L, -1L,
  ])

  writer.resize_dataset("/series", [2UL, 3UL])
  let shrunk = Hdf5File::parse(writer.encode()).read_dataset("/series")
  assert_eq(integer_values(shrunk), [1L, 2L, 3L, 4L, 5L, 6L])
}

///|
test "writer extensible arrays use direct data blocks" {
  let values = Array::make(300, 0L)
  values[0] = 1L
  values[127] = 2L
  values[128] = 3L
  values[299] = 4L
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/many",
    [300UL],
    [None],
    [1U],
    Int64Values(values),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  let file = Hdf5File::parse(writer.encode())
  let selected = file.read_dataset_selection(
    "/many",
    Points([[299UL], [128UL], [127UL], [1UL], [0UL]]),
  )
  match selected.values {
    Integers(items) => assert_eq(items, [4L, 3L, 2L, 0L, 1L])
    _ => fail("expected extensible-array direct block values")
  }
}

///|
test "writer extensible arrays use super blocks and paged data blocks" {
  let values = Array::make(134200, 0U)
  values[0] = 1U
  values[243] = 2U
  values[244] = 3U
  values[65523] = 8U
  values[65524] = 9U
  values[131059] = 4U
  values[131060] = 5U
  values[132100] = 6U
  values[134199] = 7U
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/paged",
    [134200UL],
    [None],
    [1U],
    UInt8Values(values),
    filters=[Deflate(6), Fletcher32],
  )
  let file = Hdf5File::parse(writer.encode())
  let selected = file.read_dataset_selection(
    "/paged",
    Points([
      [134199UL],
      [132100UL],
      [131061UL],
      [131060UL],
      [131059UL],
      [65524UL],
      [65523UL],
      [244UL],
      [243UL],
      [0UL],
    ]),
  )
  match selected.values {
    UnsignedIntegers(items) =>
      assert_eq(items, [7UL, 6UL, 0UL, 5UL, 4UL, 9UL, 8UL, 3UL, 2UL, 1UL])
    _ => fail("expected extensible-array super-block values")
  }
}

///|
test "writer rejects chunk grids beyond supported capacity" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/large",
    [0UL, 50000UL],
    [None, Some(50000UL)],
    [1U, 1U],
    Int64Values([]),
  )
  let error = try {
    writer.resize_dataset("/large", [50000UL, 50000UL])
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidArgument("writer chunk grid is too large")))
}

///|
test "writer extensible indexes swizzle a nonleading unlimited dimension" {
  let writer = Hdf5Writer::new()
  writer.add_extensible_dataset(
    "/filtered",
    [2UL, 0UL],
    [Some(2UL), None],
    [1U, 2U],
    UInt64Values([]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  writer.append_dataset(
    "/filtered",
    [2UL, 3UL],
    UInt64Values([1UL, 2UL, 3UL, 4UL, 5UL, 6UL]),
  )
  writer.append_dataset("/filtered", [2UL, 1UL], UInt64Values([7UL, 8UL]))
  let dataset = Hdf5File::parse(writer.encode()).read_dataset("/filtered")
  assert_eq(dataset.dataspace.dimensions, [2UL, 4UL])
  assert_eq(unsigned_values(dataset), [1UL, 2UL, 3UL, 7UL, 4UL, 5UL, 6UL, 8UL])
}

///|
test "writer validates extensible shapes and mutations" {
  let writer = Hdf5Writer::new()
  let missing_unlimited = try {
    writer.add_extensible_dataset(
      "/fixed",
      [2UL],
      [Some(4UL)],
      [2U],
      Int64Values([1L, 2L]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    missing_unlimited,
    Some(InvalidArgument("extensible dataset requires one unlimited dimension")),
  )
  let oversized_chunk = try {
    writer.add_extensible_dataset(
      "/oversized_chunk",
      [0UL, 1UL],
      [None, Some(1UL)],
      [0x7FFFFFFFU, 2U],
      Int64Values([]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(oversized_chunk, Some(InvalidArgument("writer chunk is too large")))
  writer.add_extensible_dataset(
    "/values",
    [1UL, 2UL],
    [None, Some(2UL)],
    [1U, 2U],
    Int64Values([1L, 2L]),
  )
  let maximum = try {
    writer.resize_dataset("/values", [2UL, 3UL])
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    maximum,
    Some(InvalidArgument("resized dataset exceeds its maximum dimensions")),
  )
  let append_shape = try {
    writer.append_dataset("/values", [1UL, 1UL], Int64Values([3L]))
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    append_shape,
    Some(InvalidArgument("appended block must match fixed dataset dimensions")),
  )
}

///|
test "writer round-trips narrow integers and float32 across layouts" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/i8", [3UL], Int8Values([-128, 0, 127]))
  writer.add_dataset("/i16", [3UL], Int16Values([-32768, 0, 32767]))
  writer.add_dataset("/i32", [3UL], Int32Values([-2147483647, 0, 2147483647]))
  writer.add_dataset("/u8", [3UL], UInt8Values([0U, 1U, 255U]))
  writer.add_dataset("/u16", [3UL], UInt16Values([0U, 1U, 65535U]))
  writer.add_dataset("/u32", [3UL], UInt32Values([0U, 1U, 0xFFFFFFFFU]))
  writer.add_dataset(
    "/f32",
    [3UL],
    Float32Values([Float(-1.5), Float(0.0), Float(2.25)]),
  )
  writer.add_chunked_dataset(
    "/chunked_i16",
    [5UL],
    [2U],
    Int16Values([-7, 2, -7, 4, -7]),
    fill_value=Int16Values([-7]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  writer.add_extensible_dataset(
    "/growing_f32",
    [0UL],
    [None],
    [2U],
    Float32Values([]),
    fill_value=Float32Values([Float(-1.0)]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  writer.append_dataset(
    "/growing_f32",
    [3UL],
    Float32Values([Float(1.25), Float(2.5), Float(3.75)]),
  )
  writer.resize_dataset("/growing_f32", [4UL])

  let file = Hdf5File::parse(writer.encode())
  assert_eq(integer_values(file.read_dataset("/i8")), [-128L, 0L, 127L])
  assert_eq(integer_values(file.read_dataset("/i16")), [-32768L, 0L, 32767L])
  assert_eq(integer_values(file.read_dataset("/i32")), [
    -2147483647L, 0L, 2147483647L,
  ])
  assert_eq(unsigned_values(file.read_dataset("/u8")), [0UL, 1UL, 255UL])
  assert_eq(unsigned_values(file.read_dataset("/u16")), [0UL, 1UL, 65535UL])
  assert_eq(unsigned_values(file.read_dataset("/u32")), [0UL, 1UL, 0xFFFFFFFFUL])
  assert_eq(float_values(file.read_dataset("/f32")), [-1.5, 0.0, 2.25])
  assert_eq(integer_values(file.read_dataset("/chunked_i16")), [
    -7L, 2L, -7L, 4L, -7L,
  ])
  assert_eq(float_values(file.read_dataset("/growing_f32")), [
    1.25, 2.5, 3.75, -1.0,
  ])
}

///|
test "writer rejects narrow integer values outside their datatype" {
  let writer = Hdf5Writer::new()
  let signed = try {
    writer.add_dataset("/signed", [1UL], Int8Values([128]))
    None
  } catch {
    error => Some(error)
  }
  let unsigned = try {
    writer.add_dataset("/unsigned", [1UL], UInt16Values([65536U]))
    None
  } catch {
    error => Some(error)
  }
  let fill = try {
    writer.add_chunked_dataset(
      "/fill",
      [1UL],
      [1U],
      Int8Values([1]),
      fill_value=Int8Values([-129]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(signed, Some(InvalidArgument("Int8 writer value is out of range")))
  assert_eq(
    unsigned,
    Some(InvalidArgument("UInt16 writer value is out of range")),
  )
  assert_eq(fill, Some(InvalidArgument("Int8 writer value is out of range")))
}

///|
fn writer_record_type() -> Datatype raise Hdf5Error {
  let status = Datatype::write_enum(Datatype::write_integer(false, 1), [
    { name: "MISSING", value: EnumValue::Unsigned(0UL), },
    { name: "READY", value: EnumValue::Unsigned(1UL), },
    { name: "DONE", value: EnumValue::Unsigned(2UL), },
  ])
  let samples = Datatype::write_array(Datatype::write_integer(true, 2), [
    2UL, 2UL,
  ])
  Datatype::write_compound([
    ("id", Datatype::write_integer(true, 4)),
    ("status", status),
    ("samples", samples),
    ("label", Datatype::write_fixed_string(6)),
  ])
}

///|
fn writer_record(
  id : Int64,
  status : String,
  samples : Array[Int64],
  label : String,
) -> DatasetValue {
  Compound([
    { name: "id", value: SignedInteger(id), },
    { name: "status", value: Enumeration(status), },
    {
      name: "samples",
      value: Elements(samples.map(value => SignedInteger(value))),
    },
    { name: "label", value: Text(label), },
  ])
}

///|
test "writer encodes compound enum and array values recursively" {
  let writer = Hdf5Writer::new()
  let record_type = writer_record_type()
  let missing = writer_record(0L, "MISSING", [0L, 0L, 0L, 0L], "")
  let first = writer_record(7L, "READY", [1L, 2L, 3L, 4L], "alpha")
  let second = writer_record(-2L, "DONE", [-1L, -2L, -3L, -4L], "beta")
  writer.add_chunked_dataset(
    "/records",
    [3UL],
    [2U],
    StructuredValues(record_type, [first, missing, second]),
    fill_value=StructuredValues(record_type, [missing]),
    filters=[Shuffle, Deflate(6), Fletcher32],
  )
  let array_type = Datatype::write_array(Datatype::write_float(4), [2UL])
  writer.add_dataset(
    "/vectors",
    [2UL],
    StructuredValues(array_type, [
      Elements([FloatingPoint(1.5), FloatingPoint(-2.0)]),
      Elements([FloatingPoint(3.25), FloatingPoint(4.5)]),
    ]),
  )
  writer.add_attribute(
    "/records",
    "prototype",
    [],
    StructuredValues(record_type, [first]),
  )

  let file = Hdf5File::parse(writer.encode())
  let records = file.read_dataset("/records")
  assert_eq(records.datatype.class, 6)
  assert_eq(records.datatype.compound_members.length(), 4)
  assert_eq(structured_values(records), [first, missing, second])
  assert_eq(structured_values(file.read_dataset("/vectors")), [
    Elements([FloatingPoint(1.5), FloatingPoint(-2.0)]),
    Elements([FloatingPoint(3.25), FloatingPoint(4.5)]),
  ])
  match file.attributes("/records")[0].values {
    Some(Structured(values)) => assert_eq(values, [first])
    _ => abort("expected compound attribute")
  }
}

///|
test "writer appends and fills extensible enums" {
  let writer = Hdf5Writer::new()
  let state = Datatype::write_enum(Datatype::write_integer(false, 1), [
    { name: "UNKNOWN", value: EnumValue::Unsigned(0UL), },
    { name: "OPEN", value: EnumValue::Unsigned(1UL), },
    { name: "CLOSED", value: EnumValue::Unsigned(2UL), },
  ])
  writer.add_extensible_dataset(
    "/states",
    [0UL],
    [None],
    [2U],
    StructuredValues(state, []),
    fill_value=StructuredValues(state, [Enumeration("UNKNOWN")]),
  )
  writer.append_dataset(
    "/states",
    [2UL],
    StructuredValues(state, [Enumeration("OPEN"), Enumeration("CLOSED")]),
  )
  writer.resize_dataset("/states", [4UL])
  match Hdf5File::parse(writer.encode()).read_dataset("/states").values {
    Enums(values) => assert_eq(values, ["OPEN", "CLOSED", "UNKNOWN", "UNKNOWN"])
    _ => abort("expected enum values")
  }
}

///|
test "writer validates structured values before encoding" {
  let writer = Hdf5Writer::new()
  let record_type = writer_record_type()
  let wrong_fields = try {
    writer.add_dataset(
      "/bad",
      [1UL],
      StructuredValues(record_type, [
        Compound([
          { name: "status", value: Enumeration("READY"), },
          { name: "id", value: SignedInteger(1L), },
          {
            name: "samples",
            value: Elements([
              SignedInteger(1L),
              SignedInteger(2L),
              SignedInteger(3L),
              SignedInteger(4L),
            ]),
          },
          { name: "label", value: Text("bad"), },
        ]),
      ]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    wrong_fields,
    Some(InvalidArgument("compound value fields are out of order")),
  )
  let oversized_enum = try {
    let invalid = Datatype::write_enum(Datatype::write_integer(false, 1), [
      { name: "TOO_LARGE", value: EnumValue::Unsigned(256UL), },
    ])
    writer.add_dataset(
      "/enum",
      [1UL],
      StructuredValues(invalid, [Enumeration("TOO_LARGE")]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    oversized_enum,
    Some(InvalidArgument("structured unsigned integer is out of range")),
  )
}

///|
test "writer round-trips variable-length UTF-8 strings" {
  let writer = Hdf5Writer::new()
  writer.add_dataset(
    "/names",
    [4UL],
    VariableStringValues(["月亮", "HDF5", "", "月亮"]),
  )
  writer.add_chunked_dataset(
    "/chunked_names",
    [5UL],
    [2U],
    VariableStringValues(["alpha", "", "数据", "alpha", ""]),
  )
  writer.add_extensible_dataset(
    "/bounded_names",
    [1UL, 2UL],
    [None, Some(4UL)],
    [1U, 2U],
    VariableStringValues(["a", "b"]),
  )
  writer.add_extensible_dataset(
    "/events",
    [0UL],
    [None],
    [2U],
    VariableStringValues([]),
  )
  writer.append_dataset(
    "/events",
    [2UL],
    VariableStringValues(["start", "结束"]),
  )
  writer.resize_dataset("/events", [3UL])
  writer.add_attribute(
    "/names",
    "description",
    [],
    VariableStringValues(["UTF-8 标签"]),
  )

  let file = Hdf5File::parse(writer.encode())
  let names = file.read_dataset("/names")
  assert_eq(names.datatype.class, 9)
  assert_eq(names.datatype.version, 1)
  assert_eq(names.datatype.base_types.length(), 1)
  assert_eq(names.datatype.base_types[0].class, 0)
  assert_eq(names.datatype.base_types[0].size, 1)
  assert_eq(string_values(file.read_dataset("/names")), [
    "月亮", "HDF5", "", "月亮",
  ])
  assert_eq(string_values(file.read_dataset("/chunked_names")), [
    "alpha", "", "数据", "alpha", "",
  ])
  assert_eq(string_values(file.read_dataset("/bounded_names")), ["a", "b"])
  assert_eq(string_values(file.read_dataset("/events")), ["start", "结束", ""])
  match file.attributes("/names")[0].values {
    Some(Strings(values)) => assert_eq(values, ["UTF-8 标签"])
    _ => abort("expected variable-length string attribute")
  }
}

///|
test "writer rejects filters for variable-length datasets" {
  let chunked = try {
    Hdf5Writer::new().add_chunked_dataset(
      "/strings",
      [2UL],
      [1U],
      VariableStringValues(["alpha", ""]),
      filters=[Deflate(6), Fletcher32],
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    chunked,
    Some(
      UnsupportedFeature(
        "filters are not supported for variable-length datasets",
      ),
    ),
  )
  let extensible = try {
    Hdf5Writer::new().add_extensible_dataset(
      "/strings",
      [0UL],
      [None],
      [1U],
      VariableStringValues([]),
      filters=[Shuffle],
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    extensible,
    Some(
      UnsupportedFeature(
        "filters are not supported for variable-length datasets",
      ),
    ),
  )
}

///|
test "writer rejects custom variable-length string fills" {
  let writer = Hdf5Writer::new()
  let chunked = try {
    writer.add_chunked_dataset(
      "/strings",
      [1UL],
      [1U],
      VariableStringValues(["value"]),
      fill_value=VariableStringValues(["missing"]),
    )
    None
  } catch {
    error => Some(error)
  }
  let extensible = try {
    writer.add_extensible_dataset(
      "/extensible_strings",
      [0UL],
      [None],
      [1U],
      VariableStringValues([]),
      fill_value=VariableStringValues(["missing"]),
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    chunked,
    Some(
      UnsupportedFeature(
        "custom variable-length fill values are not supported yet",
      ),
    ),
  )
  assert_eq(
    extensible,
    Some(
      UnsupportedFeature(
        "custom variable-length fill values are not supported yet",
      ),
    ),
  )
}
