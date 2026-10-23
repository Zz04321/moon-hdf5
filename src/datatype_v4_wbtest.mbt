///|
fn v4_record(
  id : Int64,
  state : String,
  samples : Array[Int64],
  label : String,
) -> DatasetValue {
  Compound([
    { name: "id", value: SignedInteger(id), },
    { name: "state", value: Enumeration(state), },
    {
      name: "samples",
      value: Elements(samples.map(value => SignedInteger(value))),
    },
    { name: "label", value: Text(label), },
  ])
}

///|
fn enum_names(values : DatasetValues) -> Array[String] raise Hdf5Error {
  match values {
    Enums(names) => names
    _ => raise InvalidField("expected enum values")
  }
}

///|
fn v4_sequence_record(id : Int64, value : Double) -> DatasetValue {
  Compound([
    { name: "id", value: SignedInteger(id), },
    { name: "value", value: FloatingPoint(value), },
  ])
}

///|
test "HDF5 1.14.6 default and latest datatypes decode" {
  for
    (path, expected_version) in [
      ("testdata/datatypes_v114.h5", 4),
      ("testdata/datatypes_v114_default.h5", 2),
    ] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    let records = file.read_dataset("/records")
    assert_eq(records.datatype.class, 6)
    assert_eq(records.datatype.version, expected_version)
    assert_eq(records.datatype.compound_members.length(), 4)
    assert_eq(
      records.datatype.compound_members[1].datatype.version,
      expected_version,
    )
    assert_eq(
      records.datatype.compound_members[2].datatype.version,
      expected_version,
    )
    assert_eq(structured_values(records), [
      v4_record(7L, "OPEN", [1L, 2L, 3L, 4L], "alpha"),
      v4_record(-2L, "UNKNOWN", [-1L, -2L, -3L, -4L], "xy"),
      v4_record(99L, "CLOSED", [10L, 20L, 30L, 40L], "z"),
    ])

    let states = file.read_dataset("/states")
    assert_eq(states.datatype.class, 8)
    assert_eq(
      states.datatype.version,
      if expected_version == 4 {
        4
      } else {
        1
      },
    )
    assert_eq(enum_names(states.values), ["OPEN", "CLOSED", "UNKNOWN"])

    let vectors = file.read_dataset("/vectors")
    assert_eq(vectors.datatype.class, 10)
    assert_eq(
      vectors.datatype.version,
      if expected_version == 4 {
        4
      } else {
        2
      },
    )
    assert_eq(vectors.datatype.array_dimensions, [2UL])
    assert_eq(structured_values(vectors), [
      Elements([FloatingPoint(1.5), FloatingPoint(-2.0)]),
      Elements([FloatingPoint(3.25), FloatingPoint(4.5)]),
    ])
    let sequences = file.read_dataset("/sequences")
    assert_eq(sequences.datatype.class, 9)
    assert_eq(
      sequences.datatype.version,
      if expected_version == 4 {
        4
      } else {
        1
      },
    )
    assert_eq(structured_values(sequences), [
      Sequence([v4_sequence_record(1L, 1.5), v4_sequence_record(2L, 2.5)]),
      Sequence([v4_sequence_record(3L, 3.5)]),
    ])
    assert_eq(string_values(file.read_dataset("/names")), ["月亮", "HDF5", ""])
  }
}

///|
test "HDF5 1.14.6 default and latest compound attributes decode" {
  for
    (path, expected_version) in [
      ("testdata/datatypes_v114.h5", 4),
      ("testdata/datatypes_v114_default.h5", 2),
    ] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    let attributes = file.attributes("/records")
    assert_eq(attributes.length(), 1)
    assert_eq(attributes[0].name, "prototype")
    assert_eq(attributes[0].datatype.version, expected_version)
    match attributes[0].values {
      Some(Structured(values)) =>
        assert_eq(values, [v4_record(7L, "OPEN", [1L, 2L, 3L, 4L], "alpha")])
      _ => abort("expected structured versioned attribute")
    }
  }
}
