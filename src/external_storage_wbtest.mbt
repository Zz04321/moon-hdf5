///|
fn external_selection_integers(result : DatasetSelectionResult) -> Array[Int64] {
  match result.values {
    Integers(values) => values
    _ => abort("expected external selection integers")
  }
}

///|
fn external_chunk_integers(chunk : DatasetChunk) -> Array[Int64] {
  match chunk.values {
    Integers(values) => values
    _ => abort("expected external chunk integers")
  }
}

///|
test "external data file names normalize local URIs and reject remotes" {
  assert_eq(external_storage_file_name("data.bin"), "data.bin")
  assert_eq(external_storage_file_name("file:data.bin"), "data.bin")
  assert_eq(external_storage_file_name("file:///tmp/data.bin"), "/tmp/data.bin")
  assert_eq(
    external_storage_file_name("file://localhost/tmp/data.bin"),
    "/tmp/data.bin",
  )
  let remote = try {
    external_storage_file_name("file://server/tmp/data.bin") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(
    remote
    is Some(
      UnsupportedFeature("remote external data file hosts are not supported")
    ),
  )
  let scheme = try {
    external_storage_file_name("https://example.com/data.bin") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(
    scheme
    is Some(UnsupportedFeature("external data URL schemes are not supported")),
  )
}

///|
test "resolver reads external raw data segments and selections" {
  let bytes = @fs.read_file_to_bytes("testdata/external_storage.h5")
  let resolver = Hdf5Resolver::new(bytes)
  resolver.add_source(
    "external_storage_0.bin",
    @fs.read_file_to_bytes("testdata/external_storage_0.bin"),
  )
  resolver.add_source(
    "external_storage_1.bin",
    @fs.read_file_to_bytes("testdata/external_storage_1.bin"),
  )
  resolver.add_source(
    "external_storage_offset.bin",
    @fs.read_file_to_bytes("testdata/external_storage_offset.bin"),
  )
  resolver.add_source(
    "external_storage_unaligned_0.bin",
    @fs.read_file_to_bytes("testdata/external_storage_unaligned_0.bin"),
  )
  resolver.add_source(
    "external_storage_unaligned_1.bin",
    @fs.read_file_to_bytes("testdata/external_storage_unaligned_1.bin"),
  )

  assert_eq(integer_values(resolver.read_dataset("/multi")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L,
  ])
  let metadata = resolver.read_dataset_metadata("/multi")
  assert_eq(metadata.dataspace.dimensions, [8UL])
  assert_eq(metadata.layout.kind(), Contiguous)

  let slab = resolver.read_dataset_selection(
    "/multi",
    Hyperslab([1UL], [4UL], [2UL]),
  )
  assert_eq(external_selection_integers(slab), [1L, 3L, 5L, 7L])
  let points = resolver.read_dataset_selection(
    "/multi",
    Points([[6UL], [0UL], [4UL]]),
  )
  assert_eq(external_selection_integers(points), [6L, 0L, 4L])
  assert_eq(integer_values(resolver.read_dataset("/offset")), [
    101L, 102L, 103L, 104L,
  ])
  assert_eq(integer_values(resolver.read_dataset("/unaligned")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L,
  ])
  assert_eq(
    external_selection_integers(
      resolver.read_dataset_selection("/unaligned", Points([[2UL], [5UL]])),
    ),
    [2L, 5L],
  )
}

///|
test "resolver reads external raw data with four-byte addresses" {
  let bytes = @fs.read_file_to_bytes("testdata/external_storage_32.h5")
  let resolver = Hdf5Resolver::new(bytes)
  resolver.add_source(
    "external_storage_32.bin",
    @fs.read_file_to_bytes("testdata/external_storage_32.bin"),
  )
  assert_eq(integer_values(resolver.read_dataset("/legacy")), [7L, 8L, 9L])
  assert_eq(
    external_selection_integers(
      resolver.read_dataset_selection("/legacy", Points([[2UL], [0UL]])),
    ),
    [9L, 7L],
  )
}

///|
test "external raw data requires registered sources and the resolver API" {
  let bytes = @fs.read_file_to_bytes("testdata/external_storage.h5")
  let direct = Hdf5File::parse(bytes)
  let direct_error = try {
    direct.read_dataset("/multi") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(
    direct_error
    is Some(
      UnsupportedFeature("external dataset storage requires Hdf5Resolver")
    ),
  )

  let missing = Hdf5Resolver::new(bytes)
  let missing_error = try {
    missing.read_dataset("/multi") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(
    missing_error
    is Some(
      InvalidField(
        "external target file is not registered: external_storage_0.bin"
      )
    ),
  )

  let truncated = Hdf5Resolver::new(bytes)
  truncated.add_source(
    "external_storage_0.bin",
    @fs.read_file_to_bytes("testdata/external_storage_0.bin")
    .get_view(start=0, end=8)
    .unwrap()
    .to_owned(),
  )
  truncated.add_source(
    "external_storage_1.bin",
    @fs.read_file_to_bytes("testdata/external_storage_1.bin"),
  )
  let truncated_values = Some(integer_values(truncated.read_dataset("/multi"))) catch {
    _ => None
  }
  assert_eq(truncated_values, Some([0L, 1L, 0L, 3L, 4L, 5L, 6L, 7L]))

  let resolver = Hdf5Resolver::new(bytes)
  resolver.add_source(
    "external_storage_0.bin",
    @fs.read_file_to_bytes("testdata/external_storage_0.bin"),
  )
  resolver.add_source(
    "external_storage_1.bin",
    @fs.read_file_to_bytes("testdata/external_storage_1.bin"),
  )
  let chunks = resolver.chunk_iterator("/multi", [5UL])
  let first = chunks.next().unwrap()
  assert_eq(first.offset, [0UL])
  assert_eq(first.dimensions, [5UL])
  assert_eq(external_chunk_integers(first), [0L, 1L, 2L, 3L, 4L])
  let second = chunks.next().unwrap()
  assert_eq(second.offset, [5UL])
  assert_eq(second.dimensions, [3UL])
  assert_eq(external_chunk_integers(second), [5L, 6L, 7L])
  assert_true(chunks.next() is None)
}
