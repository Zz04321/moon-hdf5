///|
test "public high-level API reads a file" {
  let file = @moon-hdf5.Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/basic_v0.h5"),
  )
  let node = file.resolve("/group/values")
  assert_eq(node.kind, @moon-hdf5.Dataset)
  let dataset = file.open_dataset("/group/values")
  match dataset.values {
    Integers(values) => assert_eq(values.length(), 10)
    _ => abort("expected integer values")
  }
  let attributes = file.attributes("/")
  assert_eq(attributes.length(), 1)
  assert_eq(attributes[0].name, "title")
}
