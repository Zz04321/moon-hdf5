///|
test "inspector exit status distinguishes success errors and usage" {
  assert_eq(run(["hdf5"]), 2)
  assert_eq(run(["hdf5", "one", "two"]), 2)
  assert_eq(run(["hdf5", "testdata/does-not-exist.h5"]), 1)
  assert_eq(run(["hdf5", "moon.mod"]), 1)
  assert_eq(run(["hdf5", "testdata/basic_v0.h5"]), 0)
  assert_eq(run(["hdf5", "testdata/attribute_storage.h5"]), 1)
  assert_eq(run(["hdf5", "testdata/numeric_layouts.h5"]), 1)
}
