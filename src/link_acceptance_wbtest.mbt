///|
test "walk expands hard-linked groups and stops ancestor cycles" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/hard_links.h5"))
  let paths = file.walk().map(node => node.path)
  assert_eq(paths.length(), 6)
  for path in ["/a", "/a/loop", "/a/values", "/b", "/b/loop", "/b/values"] {
    assert_true(paths.contains(path))
  }
  for path in ["/a/values", "/b/values", "/a/loop/values", "/b/loop/values"] {
    let values = match file.read_dataset(path).values {
      Integers(values) => values
      _ => abort("expected hard-linked integer dataset")
    }
    assert_eq(values, [1L, 2L, 3L])
  }
}
