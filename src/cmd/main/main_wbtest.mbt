///|
test "inspector exit status distinguishes success errors and usage" {
  assert_eq(run(["hdf5"]), 2)
  assert_eq(run(["hdf5", "one", "two"]), 2)
  assert_eq(run(["hdf5", "testdata/does-not-exist.h5"]), 1)
  assert_eq(run(["hdf5", "moon.mod"]), 1)
  assert_eq(run(["hdf5", "testdata/basic_v0.h5"]), 0)
  assert_eq(run(["hdf5", "testdata/attribute_storage.h5"]), 0)
  assert_eq(run(["hdf5", "testdata/numeric_layouts.h5"]), 0)
}

///|
test "inspector metadata selection and statistics commands" {
  let file = "testdata/basic_v0.h5"
  assert_eq(run(["hdf5", "inspect", file]), 0)
  assert_eq(run(["hdf5", "metadata", file, "/matrix"]), 0)
  assert_eq(run(["hdf5", "stats", file, "/matrix"]), 0)
  assert_eq(run(["hdf5", "read", file, "/matrix", "0,0", "2,2", "1,2"]), 0)
  assert_eq(run(["hdf5", "read", file, "/matrix", "--points", "1,2;0,1"]), 0)
  assert_eq(run(["hdf5", "read", file, "/matrix", "bad", "2,2", "1,1"]), 2)
  assert_eq(run(["hdf5", "metadata", file, "/group"]), 1)
}

///|
test "inspector validates unsigned coordinate syntax and overflow" {
  assert_eq(parse_uint("0"), Some(0UL))
  assert_eq(parse_uint("18446744073709551615"), Some(0xFFFFFFFFFFFFFFFFUL))
  assert_eq(parse_uint("18446744073709551616"), None)
  assert_eq(parse_uint("-1"), None)
  assert_eq(parse_uint("1x"), None)
  assert_eq(parse_coordinate("1,2,3"), Some([1UL, 2UL, 3UL]))
  assert_eq(parse_coordinate("1,,3"), None)
  assert_eq(
    parse_points("1,2;3,4;1,2"),
    Some([[1UL, 2UL], [3UL, 4UL], [1UL, 2UL]]),
  )
}
