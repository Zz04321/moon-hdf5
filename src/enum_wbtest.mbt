///|
fn enum_values(dataset : Dataset) -> Array[String] raise Hdf5Error {
  match dataset.values {
    Enums(values) => values
    _ => raise InvalidField("expected enum dataset")
  }
}

///|
fn enum_member_value(
  datatype : Datatype,
  name : String,
) -> EnumValue raise Hdf5Error {
  for enum_member in datatype.enum_members {
    if enum_member.name == name {
      return enum_member.value
    }
  }
  raise InvalidField("enum member was not found")
}

///|
test "enum datasets decode signed unsigned and big-endian members" {
  for path in ["testdata/enums_v108.h5", "testdata/enums_v110.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    let colors = file.read_dataset("/colors")
    assert_eq(colors.datatype.class, 8)
    assert_eq(colors.datatype.enum_members.length(), 4)
    assert_eq(enum_values(colors), ["RED", "LONG_NAME_XYZ", "GREEN", "EIGHTCHR"])
    assert_eq(
      enum_member_value(colors.datatype, "GREEN"),
      EnumValue::Signed(1L),
    )
    assert_eq(
      enum_member_value(colors.datatype, "EIGHTCHR"),
      EnumValue::Signed(3L),
    )

    let unsigned = file.read_dataset("/unsigned")
    assert_eq(unsigned.datatype.signed, false)
    assert_eq(enum_values(unsigned), ["LONG_NAME_XYZ", "ZERO"])
    assert_eq(
      enum_member_value(unsigned.datatype, "ZERO"),
      EnumValue::Unsigned(0UL),
    )

    let big_endian = file.read_dataset("/big_endian")
    assert_eq(big_endian.datatype.signed, true)
    assert_eq(big_endian.datatype.little_endian, false)
    assert_eq(enum_values(big_endian), ["HIGH", "LOW"])
    assert_eq(
      enum_member_value(big_endian.datatype, "HIGH"),
      EnumValue::Signed(300L),
    )
    assert_eq(
      enum_member_value(big_endian.datatype, "LOW"),
      EnumValue::Signed(-1L),
    )

    let expected_version = if path == "testdata/enums_v108.h5" { 1 } else { 3 }
    assert_eq(colors.datatype.version, expected_version)
  }
}

///|
test "enum attributes decode names and values" {
  for path in ["testdata/enums_v108.h5", "testdata/enums_v110.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    let attributes = file.attributes("/colors")
    assert_eq(attributes.length(), 2)
    for attribute in attributes {
      match attribute.name {
        "favorite" =>
          match attribute.values {
            Some(Enums(values)) => assert_eq(values, ["GREEN"])
            _ => abort("expected enum favorite attribute")
          }
        "pair" =>
          match attribute.values {
            Some(Enums(values)) => assert_eq(values, ["LONG_NAME_XYZ", "RED"])
            _ => abort("expected enum pair attribute")
          }
        _ => abort("unexpected enum attribute")
      }
    }
  }
}

///|
fn enum_datatype_error(data : Bytes) -> Hdf5Error? {
  try {
    parse_datatype_data(data) |> ignore
    None
  } catch {
    error => Some(error)
  }
}

///|
test "version three enum names are not padded" {
  let datatype = parse_datatype_data(
    b"\x38\x01\x00\x00\x04\x00\x00\x00\x10\x08\x00\x00\x04\x00\x00\x00\x00\x00\x20\x00A\x00\x01\x00\x00\x00",
  )
  assert_eq(datatype.class, 8)
  assert_eq(datatype.enum_members.length(), 1)
  assert_eq(datatype.enum_members[0].name, "A")
  assert_eq(datatype.enum_members[0].value, EnumValue::Signed(1L))
}

///|
test "malformed enum datatypes are rejected" {
  let cases : Array[(Bytes, Hdf5Error)] = [
    (
      b"\x38\x00\x00\x00\x04\x00\x00\x00\x10\x08\x00\x00\x04\x00\x00\x00\x00\x00\x20\x00",
      InvalidField("enum datatype has no members"),
    ),
    (
      b"\x38\x01\x00\x00\x04\x00\x00\x00\x10\x08\x00\x00\x04\x00\x00\x00\x00\x00\x20\x00A",
      UnexpectedEof(20L),
    ),
    (
      b"\x38\x01\x00\x00\x04\x00\x00\x00\x10\x08\x00\x00\x04\x00\x00\x00\x00\x00\x20\x00\x00",
      InvalidField("enum member name is empty"),
    ),
    (
      b"\x38\x01\x00\x00\x08\x00\x00\x00\x10\x08\x00\x00\x04\x00\x00\x00\x00\x00\x20\x00",
      InvalidField("enum datatype size does not match its base type"),
    ),
    (
      b"\x38\x01\x00\x00\x04\x00\x00\x00\x14\x00\x00\x00\x04\x00\x00\x00",
      UnexpectedEof(16L),
    ),
    (
      b"\x48\x01\x00\x00\x04\x00\x00\x00\x10\x08\x00\x00\x04\x00\x00\x00\x00\x00\x20\x00",
      UnsupportedFeature("unsupported enum datatype version"),
    ),
  ]
  for (data, expected) in cases {
    assert_eq(enum_datatype_error(data), Some(expected))
  }
}

///|
test "unknown enum payload values are rejected" {
  let datatype : Datatype = {
    class: 8,
    version: 3,
    size: 4,
    signed: true,
    little_endian: true,
    vlen_is_string: false,
    enum_members: [{ name: "ONE", value: EnumValue::Signed(1L), }],
    compound_members: [],
    base_types: [],
    array_dimensions: [],
    bit_offset: 0,
    bit_precision: 0,
    string_padding: 0,
    string_charset: 0,
    opaque_tag: "",
    reference_kind: 0,
  }
  let error = try {
    decode_values(b"", b"\x02\x00\x00\x00", 1, datatype, 8, 8, 0UL) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidField("enum value does not match any member")))
}
