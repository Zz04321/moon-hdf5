///|
fn shared_integer_values(dataset : Dataset) -> Array[Int64] raise Hdf5Error {
  match dataset.values {
    Integers(values) => values
    _ => raise InvalidField("expected integer values")
  }
}

///|
fn shared_attribute_values(
  file : Hdf5File,
  path : String,
  name : String,
) -> Array[Int64] raise Hdf5Error {
  let attributes = file.attributes(path)
  let attribute = match
    attributes.search_by(attribute => attribute.name == name) {
    Some(index) => attributes[index]
    None => raise InvalidField("shared attribute not found")
  }
  match attribute.values {
    Some(Integers(values)) => values
    _ => raise InvalidField("expected integer attribute values")
  }
}

///|
fn check_shared_message_fixture(file : Hdf5File) -> Unit raise {
  let small = [7L, 42L]
  let large = [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L,
  ]
  assert_eq(shared_attribute_values(file, "/", "rootattr"), small)
  for index in 0..<12 {
    assert_eq(shared_attribute_values(file, "/d\{index}", "common"), small)
    assert_eq(
      shared_attribute_values(file, "/large\{index}", "large_common"),
      large,
    )
  }
  for index in 0..<6 {
    assert_eq(
      shared_attribute_values(file, "/component\{index}", "unique\{index}"),
      small,
    )
  }
  for index in 0..<4 {
    assert_eq(shared_integer_values(file.read_dataset("/fill\{index}")), [
      -7L, -7L, -7L, -7L,
    ])
  }

  assert_eq(file.attributes("/dense0").length(), 13)
  assert_eq(file.attributes("/dense1").length(), 13)
  for name in ["dense0", "dense1"] {
    let common = large.map(value => value + 7L)
    assert_eq(shared_attribute_values(file, "/" + name, "common"), common)
    for index in 0..<12 {
      let expected = large.map(value => value + index.to_int64())
      let suffix = if index < 10 {
        "0" + index.to_string()
      } else {
        index.to_string()
      }
      assert_eq(
        shared_attribute_values(file, "/" + name, name + "_unique" + suffix),
        expected,
      )
    }
  }

  assert_eq(shared_attribute_values(file, "/committed", "named"), small)
  assert_eq(shared_integer_values(file.read_dataset("/committed_value")), [
    4L, 5L, 6L,
  ])
  assert_eq(shared_integer_values(file.read_dataset("/chunked")).length(), 20)
  let selected = file.read_dataset_selection(
    "/chunked",
    Hyperslab([1UL, 1UL], [2UL, 3UL], [2UL, 1UL]),
  )
  match selected.values {
    Integers(values) => assert_eq(values, [6L, 7L, 8L, 16L, 17L, 18L])
    _ => raise InvalidField("expected integer chunk selection")
  }
  let chunks = file.chunk_iterator("/chunked", [2UL, 3UL])
  let first = chunks.next().unwrap()
  assert_eq(first.offset, [0UL, 0UL])
  assert_eq(first.dimensions, [2UL, 3UL])
  match first.values {
    Integers(values) => assert_eq(values, [0L, 1L, 2L, 5L, 6L, 7L])
    _ => raise InvalidField("expected integer chunk values")
  }
}

///|
test "shared messages resolve through memory and source files" {
  let bytes = @fs.read_file_to_bytes("testdata/shared_messages.h5")
  check_shared_message_fixture(Hdf5File::parse(bytes))
  let sources : Array[&Hdf5Source] = [bytes]
  check_shared_message_fixture(Hdf5File::parse_source(sources[0]))
}

///|
test "shared message heap IDs validate their index and size" {
  let context : SharedMessageContext = {
    bytes: b"",
    source: None,
    offset_size: 8,
    length_size: 8,
    base_address: 0UL,
    indexes: [],
  }
  let size_error = try {
    context.resolve_sohm_heap_id(b"\x00", msg_attribute) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    size_error,
    Some(InvalidField("shared-message heap ID has an invalid size")),
  )
  let index_error = try {
    let _ = context.resolve_sohm_heap_id(
      b"\x00\x00\x00\x00\x00\x00\x00\x00", msg_attribute,
    )
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    index_error,
    Some(UnsupportedFeature("shared message has no matching SOHM index")),
  )
}
