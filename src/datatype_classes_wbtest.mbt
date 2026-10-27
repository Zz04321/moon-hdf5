///|
fn structured_values(dataset : Dataset) -> Array[DatasetValue] raise Hdf5Error {
  match dataset.values {
    Structured(values) => values
    _ => raise InvalidField("expected structured dataset")
  }
}

///|
fn compound_value(
  id : Int64,
  weight : Double,
  label : String,
  left : Int64,
  right : Int64,
  matrix : Array[UInt64],
) -> DatasetValue {
  Compound([
    { name: "id", value: SignedInteger(id), },
    { name: "weight", value: FloatingPoint(weight), },
    { name: "label", value: Text(label), },
    {
      name: "pair",
      value: Compound([
        { name: "left", value: SignedInteger(left), },
        { name: "right", value: SignedInteger(right), },
      ]),
    },
    {
      name: "matrix",
      value: Elements(matrix.map(value => UnsignedInteger(value))),
    },
  ])
}

///|
test "compound array and variable-length datatypes decode recursively" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/datatypes_v200.h5"),
  )
  let compound = file.read_dataset("/compound")
  assert_eq(compound.datatype.class, 6)
  assert_eq(compound.datatype.version, 5)
  assert_eq(compound.datatype.compound_members.length(), 5)
  assert_eq(compound.datatype.compound_members[3].datatype.class, 6)
  assert_eq(compound.datatype.compound_members[4].datatype.class, 10)
  assert_eq(compound.datatype.compound_members[4].datatype.array_dimensions, [
    2UL, 2UL,
  ])
  assert_eq(structured_values(compound), [
    compound_value(7L, 1.5, "alpha", 1L, 2L, [10UL, 11UL, 12UL, 13UL]),
    compound_value(-2L, -2.25, "xy", 3L, 4L, [20UL, 21UL, 22UL, 23UL]),
    compound_value(99L, 0.0, "z", 5L, 6L, [30UL, 31UL, 32UL, 33UL]),
  ])

  let array = file.read_dataset("/array")
  assert_eq(array.datatype.class, 10)
  assert_eq(array.datatype.version, 5)
  assert_eq(array.datatype.array_dimensions, [2UL, 3UL])
  assert_eq(structured_values(array), [
    Elements([
      SignedInteger(1L),
      SignedInteger(2L),
      SignedInteger(3L),
      SignedInteger(4L),
      SignedInteger(5L),
      SignedInteger(6L),
    ]),
    Elements([
      SignedInteger(-1L),
      SignedInteger(-2L),
      SignedInteger(-3L),
      SignedInteger(-4L),
      SignedInteger(-5L),
      SignedInteger(-6L),
    ]),
  ])

  assert_eq(structured_values(file.read_dataset("/sequences")), [
    Sequence([SignedInteger(1L), SignedInteger(-2L), SignedInteger(3L)]),
    Sequence([]),
    Sequence([SignedInteger(300L), SignedInteger(400L)]),
  ])
  assert_eq(string_values(file.read_dataset("/utf8")), ["月亮", "HDF5", ""])
}

///|
test "bitfield opaque time reference and complex datatypes decode" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/datatypes_v200.h5"),
  )
  let bitfield = file.read_dataset("/bitfield")
  assert_eq(bitfield.datatype.class, 4)
  assert_eq(bitfield.datatype.bit_precision, 16)
  assert_eq(structured_values(bitfield), [
    Bitfield(1UL),
    Bitfield(240UL),
    Bitfield(42330UL),
  ])

  let opaque_dataset = file.read_dataset("/opaque")
  assert_eq(opaque_dataset.datatype.class, 5)
  assert_eq(opaque_dataset.datatype.opaque_tag, "NUMPY:|V4")
  assert_eq(structured_values(opaque_dataset), [
    Opaque(b"\x00\x01\xfe\xff"),
    Opaque(b"MOON"),
  ])

  let time = file.read_dataset("/time")
  assert_eq(time.datatype.class, 2)
  assert_eq(structured_values(time), [Time(0UL), Time(1700000000UL)])

  let object_refs = structured_values(file.read_dataset("/object_refs"))
  assert_eq(object_refs.length(), 2)
  match (object_refs[0], object_refs[1]) {
    (Reference(first), Reference(second)) => {
      assert_eq(first.length(), 8)
      let target = file.resolve_object_reference(first).unwrap()
      assert_eq(target.kind, Dataset)
      assert_eq(target.path, Some("/target"))
      assert_eq(second, b"\x00\x00\x00\x00\x00\x00\x00\x00")
    }
    _ => abort("expected object references")
  }
  let region_refs = structured_values(file.read_dataset("/region_refs"))
  match region_refs[0] {
    Reference(value) => assert_eq(value.length(), 12)
    _ => abort("expected region reference")
  }

  assert_eq(structured_values(file.read_dataset("/complex64")), [
    Complex(1.0, 2.0),
    Complex(-3.5, 0.25),
  ])
  assert_eq(structured_values(file.read_dataset("/complex128_be")), [
    Complex(2.0, -4.0),
  ])
}

///|
test "structured values work in fills selections and attributes" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/datatypes_v200.h5"),
  )
  let filled = structured_values(file.read_dataset("/compound_fill"))
  let expected = compound_value(-2L, -2.25, "xy", 3L, 4L, [
    20UL, 21UL, 22UL, 23UL,
  ])
  assert_eq(filled, [expected, expected])

  let selected = file.read_dataset_selection(
    "/compound",
    Points([[2UL], [0UL], [2UL]]),
  )
  match selected.values {
    Structured(values) =>
      assert_eq(values, [
        compound_value(99L, 0.0, "z", 5L, 6L, [30UL, 31UL, 32UL, 33UL]),
        compound_value(7L, 1.5, "alpha", 1L, 2L, [10UL, 11UL, 12UL, 13UL]),
        compound_value(99L, 0.0, "z", 5L, 6L, [30UL, 31UL, 32UL, 33UL]),
      ])
    _ => abort("expected structured selection")
  }

  let compound_attributes = file.attributes("/compound")
  assert_eq(compound_attributes.length(), 1)
  match compound_attributes[0].values {
    Some(Structured(values)) => assert_eq(values, [expected])
    _ => abort("expected compound attribute")
  }
  let array_attributes = file.attributes("/array")
  assert_eq(array_attributes.length(), 1)
  match array_attributes[0].values {
    Some(Structured(values)) =>
      assert_eq(values, [
        Elements([
          SignedInteger(1L),
          SignedInteger(2L),
          SignedInteger(3L),
          SignedInteger(4L),
          SignedInteger(5L),
          SignedInteger(6L),
        ]),
      ])
    _ => abort("expected array attribute")
  }
}

///|
fn structured_datatype_error(data : Bytes) -> Hdf5Error? {
  try {
    parse_datatype_data(data) |> ignore
    None
  } catch {
    error => Some(error)
  }
}

///|
test "malformed structured datatype metadata is rejected" {
  assert_eq(
    structured_datatype_error(
      b"\x3a\x00\x00\x00\x04\x00\x00\x00\x01\x02\x00\x00\x00\x10\x08\x00\x00\x04\x00\x00\x00\x00\x00\x20\x00",
    ),
    Some(InvalidField("array datatype size does not match its base type")),
  )
  assert_eq(
    structured_datatype_error(
      b"\x36\x01\x00\x00\x04\x00\x00\x00A\x00\x03\x10\x08\x00\x00\x02\x00\x00\x00\x00\x00\x10\x00",
    ),
    Some(InvalidField("compound member extends beyond datatype size")),
  )
  assert_eq(
    structured_datatype_error(b"\x15\x02\x00\x00\x01\x00\x00\x00ABCDEFGH"),
    Some(InvalidField("opaque datatype tag is not terminated")),
  )
  assert_eq(
    structured_datatype_error(
      b"\x5b\x01\x00\x00\x08\x00\x00\x00\x11\x21\x3f\x00\x08\x00\x00\x00\x00\x00\x40\x00\x34\x0b\x00\x34\xff\x03\x00\x00",
    ),
    Some(InvalidField("complex datatype size does not match its base type")),
  )
}
