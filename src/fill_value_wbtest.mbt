///|
test "missing chunks preserve declared fill values" {
  for fixture in ["testdata/fill_v2.h5", "testdata/fill_v3.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    for (name, fill) in [("signed", -7L), ("big_endian", -1234L)] {
      let expected = Array::make(15, fill)
      expected[0] = 42L
      expected[14] = 99L
      assert_eq(integer_values(file.read_dataset("/numbers/" + name)), expected)
    }
    let floats = Array::make(15, 1.5)
    floats[0] = 42.0
    floats[14] = 99.0
    assert_eq(float_values(file.read_dataset("/numbers/floating")), floats)
    let strings = Array::make(15, "none")
    strings[0] = "yes"
    strings[14] = "end"
    assert_eq(string_values(file.read_dataset("/other/text")), strings)
    assert_eq(integer_values(file.read_dataset("/other/partial_default")), [
      5L, 0L, 0L, 0L,
    ])
  }
}

///|
test "unallocated chunk indexes return fill values" {
  for fixture in ["testdata/fill_v2.h5", "testdata/fill_v3.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(fixture))
    assert_eq(integer_values(file.read_dataset("/other/unallocated")), [
      -9L, -9L, -9L, -9L,
    ])
    assert_eq(integer_values(file.read_dataset("/other/default")), [
      0L, 0L, 0L, 0L,
    ])
  }
}

///|
test "variable-length string fills resolve from the global heap" {
  let bytes = @fs.read_file_to_bytes("testdata/vlen_fills.h5")
  let file = Hdf5File::parse(bytes)
  assert_eq(string_values(file.read_dataset("/contiguous")), [
    "missing", "missing", "missing", "missing",
  ])
  assert_eq(string_values(file.read_dataset("/chunked")), [
    "first", "fallback", "fallback", "fallback", "fallback", "last",
  ])
  assert_eq(string_values(file.read_dataset("/extensible")), [
    "grow", "grow", "grow", "grow", "grow",
  ])
  let sources : Array[&Hdf5Source] = [bytes]
  let random = Hdf5File::parse_source(sources[0])
  match
    random.read_dataset_selection("/chunked", Points([[1UL], [4UL], [5UL]])).values {
    Strings(values) => assert_eq(values, ["fallback", "fallback", "last"])
    _ => fail("expected variable-length fill selections")
  }
}

///|
fn fill_test_header(data : Bytes, old~ : Bool = false) -> ObjectHeader {
  {
    _version: 2,
    messages: [
      {
        msg_type: if old {
          msg_fill_value_old
        } else {
          msg_fill_value
        },
        flags: b'\x00',
        data,
      },
    ],
  }
}

///|
test "fill message versions and default values" {
  let datatype : Datatype = {
    class: 0,
    version: 1,
    size: 4,
    signed: true,
    little_endian: true,
    vlen_is_string: false,
    enum_members: [],
    compound_members: [],
    base_types: [],
    array_dimensions: [],
    bit_offset: 0,
    bit_precision: 0,
    float_sign_position: 0,
    float_exponent_position: 0,
    float_exponent_size: 0,
    float_mantissa_position: 0,
    float_mantissa_size: 0,
    float_exponent_bias: 0,
    float_normalization: 0,
    string_padding: 0,
    string_charset: 0,
    opaque_tag: "",
    reference_kind: 0,
  }
  for
    data in [
      b"\x01\x03\x02\x01\x04\x00\x00\x00\xf9\xff\xff\xff", b"\x02\x03\x02\x01\x04\x00\x00\x00\xf9\xff\xff\xff",
      b"\x03\x2b\x04\x00\x00\x00\xf9\xff\xff\xff",
    ] {
    assert_eq(
      dataset_fill_value(fill_test_header(data), datatype),
      b"\xf9\xff\xff\xff",
    )
  }
  assert_eq(
    dataset_fill_value(
      fill_test_header(b"\x04\x00\x00\x00\xf9\xff\xff\xff", old=true),
      datatype,
    ),
    b"\xf9\xff\xff\xff",
  )
  assert_eq(dataset_fill_value(fill_test_header(b"\x03\x0b"), datatype), b"")
  assert_eq(
    dataset_fill_value(
      fill_test_header(b"\x02\x03\x02\x01\x00\x00\x00\x00"),
      datatype,
    ),
    b"",
  )
}

///|
test "undefined disabled and malformed fill messages are rejected" {
  let datatype : Datatype = {
    class: 0,
    version: 1,
    size: 4,
    signed: true,
    little_endian: true,
    vlen_is_string: false,
    enum_members: [],
    compound_members: [],
    base_types: [],
    array_dimensions: [],
    bit_offset: 0,
    bit_precision: 0,
    float_sign_position: 0,
    float_exponent_position: 0,
    float_exponent_size: 0,
    float_mantissa_position: 0,
    float_mantissa_size: 0,
    float_exponent_bias: 0,
    float_normalization: 0,
    string_padding: 0,
    string_charset: 0,
    opaque_tag: "",
    reference_kind: 0,
  }
  for data in [b"\x03\x1b", b"\x03\x07", b"\x02\x03\x02\x00", b"\x04\x00"] {
    let error = try {
      dataset_fill_value(fill_test_header(data), datatype) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(UnsupportedFeature(_)))
  }
  for
    data in [
      b"\x03", b"\x03\x3b", b"\x03\xcb", b"\x03\x2b\x04\x00\x00\x00\x01", b"\x03\x2b\x02\x00\x00\x00\x01\x00",
    ] {
    let error = try {
      dataset_fill_value(fill_test_header(data), datatype) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(
      error is Some(InvalidField(_)) || error is Some(UnexpectedEof(_)),
    )
  }
}
