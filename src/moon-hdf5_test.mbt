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
