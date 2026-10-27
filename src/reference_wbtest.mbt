///|
fn reference_value_bytes(dataset : Dataset) -> Array[Bytes] raise Hdf5Error {
  let references : Array[Bytes] = []
  match dataset.values {
    Structured(values) =>
      for value in values {
        match value {
          Reference(bytes) => references.push(bytes)
          _ => raise InvalidField("expected reference values")
        }
      }
    _ => raise InvalidField("expected a structured reference dataset")
  }
  references
}

///|
fn check_reference_fixture(file : Hdf5File) -> Unit raise Error {
  let references = reference_value_bytes(file.read_dataset("/object_refs"))
  assert_eq(references.length(), 6)
  let target = file.resolve("/target")
  assert_eq(
    file.resolve_object_reference(references[0]),
    Some({ address: target.address, kind: Dataset, path: Some("/target"), }),
  )
  let group = file.resolve("/group")
  assert_eq(
    file.resolve_object_reference(references[1]),
    Some({ address: group.address, kind: Group, path: Some("/group"), }),
  )
  let child = file.resolve("/group/child")
  assert_eq(
    file.resolve_object_reference(references[2]),
    Some({ address: child.address, kind: Dataset, path: Some("/group/child"), }),
  )
  let root = file.resolve("/")
  assert_eq(
    file.resolve_object_reference(references[3]),
    Some({ address: root.address, kind: Group, path: Some("/"), }),
  )
  let datatype = file.resolve("/named_i4")
  assert_eq(datatype.kind, Datatype)
  assert_eq(
    file.resolve_object_reference(references[4]),
    Some({ address: datatype.address, kind: Datatype, path: Some("/named_i4"), }),
  )
  assert_eq(file.resolve_object_reference(references[5]), None)

  let attributes = file.attributes("/target")
  let child_ref = match
    attributes.search_by(attribute => attribute.name == "child_ref") {
    Some(index) => attributes[index]
    None => raise InvalidField("target.child_ref attribute is missing")
  }
  match child_ref.values {
    Some(Structured([Reference(bytes)])) =>
      assert_eq(
        file.resolve_object_reference(bytes),
        Some({
          address: child.address,
          kind: Dataset,
          path: Some("/group/child"),
        }),
      )
    _ => raise InvalidField("target.child_ref has an unexpected value")
  }
}

///|
test "object references resolve from in-memory files" {
  for path in ["testdata/references.h5", "testdata/references_v108.h5"] {
    check_reference_fixture(Hdf5File::parse(@fs.read_file_to_bytes(path)))
  }
}

///|
test "object references resolve from random-access sources" {
  for path in ["testdata/references.h5", "testdata/references_v108.h5"] {
    let bytes = @fs.read_file_to_bytes(path)
    let sources : Array[&Hdf5Source] = [bytes]
    check_reference_fixture(Hdf5File::parse_source(sources[0]))
  }
}

///|
test "null malformed dangling and region references are distinguished" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/references.h5"))
  let null_reference = b"\x00\x00\x00\x00\x00\x00\x00\x00"
  assert_eq(file.resolve_object_reference(null_reference), None)

  let short_error = try {
    file.resolve_object_reference(b"\x00\x00\x00\x00\x00\x00\x00") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(short_error is Some(InvalidField(_)))

  let dangling = reference_value_bytes(file.read_dataset("/dangling_refs"))
  let dangling_error = try {
    file.resolve_object_reference(dangling[0]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(dangling_error is Some(InvalidField(_)))

  let region = reference_value_bytes(file.read_dataset("/region_refs"))
  assert_eq(region[0].length(), 12)
  let region_error = try {
    file.resolve_object_reference(region[0]) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    region_error,
    Some(UnsupportedFeature("region reference resolution is not supported")),
  )
}
