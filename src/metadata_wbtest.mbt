///|
test "nonstandard numeric layouts are explicitly rejected" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/numeric_layouts.h5"),
  )
  for name in ["offset", "precision", "custom_float"] {
    let error = try {
      file.open_dataset("/" + name) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(UnsupportedFeature(_)))
  }
  for name in ["offset", "precision"] {
    let error = try {
      file.attributes("/" + name) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(UnsupportedFeature(_)))
  }
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
          version: 2,
          messages: [
            {
              msg_type: msg_attribute_info,
              flags: b'\x00',
              data_offset: 0,
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
      version: 2,
      messages: [
        { msg_type: msg_attribute_info, flags: b'\x00', data_offset: 0, data, },
      ],
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
          file.open_dataset("/" + endian + "_i" + width.to_string()),
        ),
        [1L, 42L],
      )
      assert_eq(
        unsigned_values(
          file.open_dataset("/" + endian + "_u" + width.to_string()),
        ),
        [1UL, 42UL],
      )
    }
    for width in [4, 8] {
      assert_eq(
        float_values(file.open_dataset("/" + endian + "_f" + width.to_string())),
        [1.0, 42.0],
      )
    }
  }
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
