///|
fn matrix_link_name(index : Int32) -> String {
  if index < 10 {
    "link_0" + index.to_string()
  } else {
    "link_" + index.to_string()
  }
}

///|
fn matrix_attribute_name(index : Int32) -> String {
  if index < 10 {
    "attr_0" + index.to_string()
  } else {
    "attr_" + index.to_string()
  }
}

///|
fn matrix_range(start : Int64, count : Int32) -> Array[Int64] {
  let values : Array[Int64] = []
  for index in 0..<count {
    values.push(start + index.to_int64())
  }
  values
}

///|
fn assert_matrix_attributes(
  file : Hdf5File,
  path : String,
  count : Int32,
) -> Unit raise {
  let attributes = file.attributes(path)
  assert_eq(attributes.length(), count)
  for index in 0..<count {
    assert_eq(attributes[index].name, matrix_attribute_name(index))
    match attributes[index].values {
      Some(Integers(values)) => assert_eq(values, [index.to_int64()])
      _ => abort("expected integer matrix attribute")
    }
  }
}

///|
fn assert_matrix_groups(file : Hdf5File) -> Unit raise {
  assert_eq(file.resolve("/group_compact").kind, Group)
  assert_eq(file.resolve("/group_dense").kind, Group)
  assert_eq(file.resolve("/group_dense_ordered").kind, Group)
  for index in 0..<2 {
    let values = integer_values(
      file.open_dataset("/group_compact/" + matrix_link_name(index)),
    )
    assert_eq(values, [index.to_int64(), (index + 100).to_int64()])
  }
  for group in ["/group_dense", "/group_dense_ordered"] {
    for index in 0..<12 {
      let values = integer_values(
        file.open_dataset(group + "/" + matrix_link_name(index)),
      )
      assert_eq(values, [index.to_int64(), (index + 100).to_int64()])
    }
  }
}

///|
fn assert_matrix_values(file : Hdf5File) -> Unit raise {
  assert_eq(integer_values(file.open_dataset("/data/compact")), [30L, 40L])
  assert_eq(integer_values(file.open_dataset("/data/contiguous")), [10L, 20L])
  assert_eq(
    integer_values(file.open_dataset("/data/chunked_raw")),
    matrix_range(0L, 35),
  )
  assert_eq(
    integer_values(file.open_dataset("/data/chunked_filtered")),
    matrix_range(100L, 35),
  )
  let sparse = integer_values(file.open_dataset("/data/chunked_sparse"))
  assert_eq(sparse.length(), 63)
  for index in 0..<63 {
    assert_eq(
      sparse[index],
      if index == 11 {
        42L
      } else if index == 62 {
        99L
      } else {
        -7L
      },
    )
  }
  for name in ["unlimited_one", "unlimited_one_filtered"] {
    assert_eq(
      integer_values(file.open_dataset("/data/" + name)),
      matrix_range(0L, 35),
    )
  }
  for name in ["unlimited_two", "unlimited_two_filtered"] {
    assert_eq(
      integer_values(file.open_dataset("/data/" + name)),
      matrix_range(0L, 100),
    )
  }
  for name in ["single_raw", "single_filtered"] {
    assert_eq(
      integer_values(file.open_dataset("/data/" + name)),
      matrix_range(0L, 12),
    )
  }
  assert_eq(integer_values(file.open_dataset("/data/skipped")), [
    0L, 1L, 2L, 3L, -3L, -3L, -3L, -3L,
  ])
  for
    name in [
      "compact_attrs", "compact_attrs_ordered", "dense_attrs", "dense_attrs_ordered",
    ] {
    assert_eq(integer_values(file.open_dataset("/data/" + name)), [42L])
  }
  assert_matrix_attributes(file, "/data/compact_attrs", 2)
  assert_matrix_attributes(file, "/data/compact_attrs_ordered", 2)
  assert_matrix_attributes(file, "/data/dense_attrs", 12)
  assert_matrix_attributes(file, "/data/dense_attrs_ordered", 12)
}

///|
test "h5py interop matrix reads every generated version" {
  let cases : Array[(String, Int32)] = [
    ("testdata/interop_earliest.h5", 0),
    ("testdata/interop_v108.h5", 2),
    ("testdata/interop_v110.h5", 3),
  ]
  for (path, version) in cases {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    assert_eq(file.superblock.version, version)
    assert_eq(file.walk().length(), 46)
    assert_matrix_groups(file)
    assert_matrix_values(file)
  }
}

///|
test "h5py interop matrix exercises each supported chunk index" {
  let v110 = Hdf5File::parse(@fs.read_file_to_bytes("testdata/interop_v110.h5"))
  for
    name in [
      "chunked_raw", "chunked_filtered", "unlimited_one_filtered", "unlimited_two_filtered",
    ] {
    let index = v110.open_dataset("/data/" + name).layout.chunk_index
    if name == "chunked_raw" || name == "chunked_filtered" {
      assert_true(index is FixedArray(_))
    } else if name == "unlimited_one_filtered" {
      assert_true(index is ExtensibleArray(_))
    } else {
      assert_true(index is BTreeV2(_))
    }
  }
  for name in ["single_raw", "single_filtered"] {
    let index = v110.open_dataset("/data/" + name).layout.chunk_index
    assert_true(index is Single(_, _))
  }
  for path in ["testdata/interop_earliest.h5", "testdata/interop_v108.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    for
      name in [
        "chunked_raw", "chunked_filtered", "unlimited_one_filtered", "unlimited_two_filtered",
        "single_raw",
      ] {
      let index = file.open_dataset("/data/" + name).layout.chunk_index
      assert_true(index is BTreeV1)
    }
  }
}

///|
test "h5py interop matrix reads 4-byte addresses" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/interop_32.h5"))
  assert_eq(file.superblock.version, 3)
  assert_eq(file.superblock.offset_size, 4)
  assert_eq(integer_values(file.open_dataset("/raw")), matrix_range(0L, 35))
  assert_eq(
    integer_values(file.open_dataset("/filtered")),
    matrix_range(100L, 35),
  )
}
