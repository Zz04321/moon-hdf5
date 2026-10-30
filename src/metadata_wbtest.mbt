///|
test "integer precision and bit offsets decode through every value path" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/numeric_layouts.h5"),
  )
  for name in ["offset", "precision"] {
    let expected = [42L, -2L, -32768L]
    assert_eq(integer_values(file.read_dataset("/" + name)), expected)
    match find_attribute(file.attributes("/" + name), "custom").values {
      Some(Integers(values)) => assert_eq(values, expected)
      _ => abort("expected signed custom-integer attribute")
    }
  }
  let expected_unsigned = [0UL, 1UL, 703710UL, 1048575UL]
  assert_eq(unsigned_values(file.read_dataset("/unsigned")), expected_unsigned)
  match find_attribute(file.attributes("/unsigned"), "custom").values {
    Some(UnsignedIntegers(values)) => assert_eq(values, expected_unsigned)
    _ => abort("expected unsigned custom-integer attribute")
  }
  let expected_big_endian = [-2L, 42L, 524287L]
  assert_eq(
    integer_values(file.read_dataset("/big_endian")),
    expected_big_endian,
  )
  match find_attribute(file.attributes("/big_endian"), "custom").values {
    Some(Integers(values)) => assert_eq(values, expected_big_endian)
    _ => abort("expected big-endian custom-integer attribute")
  }
  let expected_signed_20 = [42L, -2L, -524288L, 524287L]
  assert_eq(integer_values(file.read_dataset("/signed_20")), expected_signed_20)
  match find_attribute(file.attributes("/signed_20"), "custom").values {
    Some(Integers(values)) => assert_eq(values, expected_signed_20)
    _ => abort("expected 20-bit signed custom-integer attribute")
  }
  let expected_unsigned_40 = [0UL, 1UL, 0xFFFFFFFFFFUL]
  assert_eq(
    unsigned_values(file.read_dataset("/unsigned_40_be")),
    expected_unsigned_40,
  )
  match find_attribute(file.attributes("/unsigned_40_be"), "custom").values {
    Some(UnsignedIntegers(values)) => assert_eq(values, expected_unsigned_40)
    _ => abort("expected 40-bit unsigned custom-integer attribute")
  }
  assert_eq(integer_values(file.read_dataset("/fill")), [-2L, -2L, -2L])
  assert_eq(float_values(file.read_dataset("/custom_float")), [1.5])
  let expected_float = [1.5, -2.25, 0.125]
  assert_eq(
    float_values(file.read_dataset("/custom_binary_float")),
    expected_float,
  )
  match
    find_attribute(file.attributes("/custom_binary_float"), "custom").values {
    Some(Floats(values)) => assert_eq(values, expected_float)
    _ => abort("expected custom binary-float attribute")
  }
}

///|
test "invalid integer storage and bit layouts are rejected" {
  let cases : Array[(Bytes, Hdf5Error)] = [
    (
      b"\x10\x08\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00",
      InvalidField("invalid integer offset or precision"),
    ),
    (
      b"\x10\x08\x00\x00\x04\x00\x00\x00\x20\x00\x01\x00",
      InvalidField("invalid integer offset or precision"),
    ),
    (
      b"\x10\x08\x00\x00\x09\x00\x00\x00\x00\x00\x40\x00",
      UnsupportedFeature("integer width is outside 1..8 bytes"),
    ),
    (
      b"\x10\x10\x00\x00\x04\x00\x00\x00\x00\x00\x20\x00",
      UnsupportedFeature("integer datatype flags are not supported"),
    ),
  ]
  for (data, expected) in cases {
    let error = try {
      parse_datatype_data(data) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_eq(error, Some(expected))
  }
}

///|
test "unsupported attribute datatypes are not reported as empty values" {
  let name = b"x\x00"
  let datatype = b"\x0c\x00\x00\x00\x04\x00\x00\x00"
  let dataspace = b"\x01\x00\x00\x00\x00\x00\x00\x00"
  let buffer = @buffer.Buffer::Buffer(size_hint=32)
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x00')
  buffer.write_uint16_le(name.length().to_uint16())
  buffer.write_uint16_le(datatype.length().to_uint16())
  buffer.write_uint16_le(dataspace.length().to_uint16())
  buffer.write_byte(b'\x00')
  buffer.write_bytes(name)
  buffer.write_bytes(datatype)
  buffer.write_bytes(dataspace)
  let error = try {
    parse_attribute_data(buffer.to_bytes(), b'\x00', b"", 8, 8, 0UL) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(UnsupportedFeature("unknown datatype class")))
}

///|
test "numeric datatype property truncation is rejected" {
  for
    data in [
      b"\x10\x08\x00\x00\x04\x00\x00\x00", b"\x11\x20\x1f\x00\x04\x00\x00\x00\x00\x00\x20\x00",
    ] {
    let error = try {
      parse_datatype_data(data) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(UnexpectedEof(_)))
  }
}

///|
test "attribute info validates address widths and index flags" {
  for width in [1, 2, 4, 8] {
    for flags in [0, 1, 3] {
      for allocated_index in [-1, 0, 1, 2] {
        let count = if flags == 3 { 3 } else { 2 }
        let buffer = @buffer.Buffer::Buffer(size_hint=32)
        buffer.write_byte(b'\x00')
        buffer.write_byte(flags.to_byte())
        if (flags & 1) != 0 {
          buffer.write_uint16_le(2)
        }
        for index in 0..<count {
          for _ in 0..<width {
            buffer.write_byte(
              if index == allocated_index {
                b'\x00'
              } else {
                b'\xff'
              },
            )
          }
        }
        let header : ObjectHeader = {
          _version: 2,
          messages: [
            {
              msg_type: msg_attribute_info,
              flags: b'\x00',
              data: buffer.to_bytes(),
            },
          ],
        }
        let error = try {
          read_attributes(b"", header, width, 8, 0UL) |> ignore
          None
        } catch {
          error => Some(error)
        }
        if allocated_index == 0 {
          assert_eq(
            error,
            Some(InvalidField("dense attributes have no name or order index")),
          )
        } else if allocated_index > 0 && allocated_index < count {
          assert_eq(
            error,
            Some(
              InvalidField("attribute index is present without a fractal heap"),
            ),
          )
        } else {
          assert_eq(error, None)
        }
      }
    }
  }
}

///|
test "malformed attribute info cannot masquerade as empty attributes" {
  for data in [b"\x00", b"\x00\x01", b"\x00\x00\xff", b"\x00\x02", b"\x00\x04"] {
    let header : ObjectHeader = {
      _version: 2,
      messages: [{ msg_type: msg_attribute_info, flags: b'\x00', data, }],
    }
    let error = try {
      read_attributes(b"", header, 8, 8, 0UL) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(
      error is Some(InvalidField(_)) || error is Some(UnexpectedEof(_)),
    )
  }
}

///|
test "standard numeric widths and byte orders remain readable" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/numeric_layouts.h5"),
  )
  for endian in ["le", "be"] {
    for width in [1, 2, 4, 8] {
      assert_eq(
        integer_values(
          file.read_dataset("/" + endian + "_i" + width.to_string()),
        ),
        [1L, 42L],
      )
      assert_eq(
        unsigned_values(
          file.read_dataset("/" + endian + "_u" + width.to_string()),
        ),
        [1UL, 42UL],
      )
    }
    for width in [2, 4, 8] {
      assert_eq(
        float_values(file.read_dataset("/" + endian + "_f" + width.to_string())),
        [1.0, 42.0],
      )
    }
  }
  let half = float_values(file.read_dataset("/half_special"))
  assert_eq(half, [
    1.0, -2.5, 65504.0, 6.103515625e-05, 5.960464477539063e-08, 0.0, 0.0,
  ])
  assert_true(1.0 / half[6] < 0.0)
  match find_attribute(file.attributes("/half_special"), "custom").values {
    Some(Floats(values)) => assert_eq(values, [1.0, -2.5, 65504.0])
    _ => abort("expected binary16 attribute")
  }
  assert_eq(float_values(file.read_dataset("/half_fill")), [-2.5, -2.5, -2.5])
}

///|
test "binary16 special bit patterns convert without arithmetic rounding" {
  assert_eq(float16_to_float(0x3C00U).to_double(), 1.0)
  assert_eq(float16_to_float(0x7BFFU).to_double(), 65504.0)
  assert_eq(float16_to_float(0x0001U).to_double(), 5.960464477539063e-08)
  assert_true(float16_to_float(0x7C00U).is_inf())
  assert_true(float16_to_float(0xFC00U).to_double() < 0.0)
  assert_true(float16_to_float(0x7E00U).is_nan())
}

///|
test "attribute storage is independent of creation order tracking" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/attribute_storage.h5"),
  )
  for path in ["/compact", "/compact_ordered"] {
    let attributes = file.attributes(path)
    assert_eq(attributes.length(), 2)
    assert_eq(
      values_integers(find_attribute(attributes, "attr_00").values.unwrap()),
      [0L],
    )
    assert_eq(
      values_integers(find_attribute(attributes, "attr_01").values.unwrap()),
      [1L],
    )
  }
  let names = [
    "attr_00", "attr_01", "attr_02", "attr_03", "attr_04", "attr_05", "attr_06",
    "attr_07", "attr_08", "attr_09", "attr_10", "attr_11",
  ]
  for path in ["/dense", "/dense_ordered"] {
    let attributes = file.attributes(path)
    assert_eq(attributes.map(attribute => attribute.name), names)
    for index in 0..<names.length() {
      assert_eq(
        values_integers(
          find_attribute(attributes, names[index]).values.unwrap(),
        ),
        [index.to_int64()],
      )
    }
  }
}
