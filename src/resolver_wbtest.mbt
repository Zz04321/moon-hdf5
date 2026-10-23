///|
test "resolver follows chained external and relative links" {
  let sibling_writer = Hdf5Writer::new()
  sibling_writer.add_dataset("/remote", [2UL], Int64Values([41L, 42L]))
  let target_writer = Hdf5Writer::new()
  target_writer.add_dataset("/value", [2UL], Int64Values([7L, 8L]))
  target_writer.add_soft_link("/", "latest", "/value")
  target_writer.add_external_link("/", "next", "sibling.h5", "/remote")
  target_writer.add_attribute(
    "/value",
    "unit",
    [],
    FixedStringValues(["item"], 4),
  )
  let root_writer = Hdf5Writer::new()
  root_writer.add_external_link("/", "chain", "data/target.h5", "/next")
  root_writer.add_external_link("/", "soft", "data/target.h5", "/latest")

  let root_bytes = root_writer.encode()
  let target_bytes = target_writer.encode()
  let sibling_bytes = sibling_writer.encode()
  let resolver = Hdf5Resolver::new(root_bytes)
  resolver.add_source("data/target.h5", target_bytes)
  resolver.add_source("data/sibling.h5", sibling_bytes)
  let node = resolver.resolve("/chain")
  assert_eq(node.file, "data/sibling.h5")
  assert_eq(node.path, "/remote")
  assert_eq(node.kind, Dataset)
  assert_eq(integer_values(resolver.read_dataset("/chain")), [41L, 42L])
  let metadata = resolver.read_dataset_metadata("/chain")
  assert_eq(metadata.dataspace.dimensions, [2UL])
  let selected = resolver.read_dataset_selection(
    "/chain",
    Hyperslab([0UL], [1UL], [1UL]),
  )
  match selected.values {
    Integers(values) => assert_eq(values, [41L])
    _ => abort("expected resolver selection integers")
  }
  let chunks = resolver.chunk_iterator("/chain", [1UL])
  let first = chunks.next().unwrap()
  assert_eq(first.offset, [0UL])
  assert_eq(first.dimensions, [1UL])
  assert_eq(chunk_integer_values(first), [41L])

  let direct = Hdf5Resolver::new(root_bytes)
  direct.add_source("data/target.h5", target_bytes)
  assert_eq(integer_values(direct.read_dataset("/soft")), [7L, 8L])
  match direct.attributes("/soft")[0].values {
    Some(Strings(values)) => assert_eq(values, ["item"])
    _ => abort("expected resolver string attribute")
  }
}

///|
test "resolver reports missing targets and link cycles" {
  let missing_writer = Hdf5Writer::new()
  missing_writer.add_external_link("/", "missing", "missing.h5", "/value")
  let missing_bytes = missing_writer.encode()
  let missing = Hdf5Resolver::new(missing_bytes)
  let missing_error = try {
    missing.resolve("/missing") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    missing_error,
    Some(
      InvalidField("external link target file is not registered: missing.h5"),
    ),
  )

  let first_writer = Hdf5Writer::new()
  first_writer.add_external_link("/", "to_second", "second.h5", "/to_first")
  let second_writer = Hdf5Writer::new()
  second_writer.add_external_link("/", "to_first", "first.h5", "/to_second")
  let first_bytes = first_writer.encode()
  let second_bytes = second_writer.encode()
  let cyclic = Hdf5Resolver::new(first_bytes)
  cyclic.add_source("first.h5", first_bytes)
  cyclic.add_source("second.h5", second_bytes)
  let cycle_error = try {
    cyclic.resolve("/to_second") |> ignore
    None
  } catch {
    error => Some(error)
  }
  match cycle_error {
    Some(InvalidField(message)) =>
      assert_true(
        message.has_prefix("soft or external link cycle detected at "),
      )
    _ => abort("expected resolver cycle error")
  }
}
