///|
test "filter pipelines decode combinations edge chunks and skipped filters" {
  for path in ["testdata/filters_v1.h5", "testdata/filters_v2.h5"] {
    let file = Hdf5File::parse(@fs.read_file_to_bytes(path))
    for name in ["gzip", "shuffle", "fletcher", "combined"] {
      let values = integer_values(file.open_dataset("/" + name))
      assert_eq(values.length(), 35)
      for index in 0..<35 {
        assert_eq(values[index], index.to_int64())
      }
    }
    assert_eq(unsigned_values(file.open_dataset("/special/odd")), [
      0UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL,
    ])
    assert_eq(integer_values(file.open_dataset("/special/partial")), [
      42L, -7L, -7L, -7L, -7L, -7L,
    ])
    assert_eq(integer_values(file.open_dataset("/special/skipped")), [
      9L, 8L, 7L, 6L,
    ])
    assert_eq(integer_values(file.open_dataset("/special/skip_deflate")), [
      9L, 8L, 7L, 6L,
    ])
    assert_eq(float_values(file.open_dataset("/special/floats")), [
      1.5, -2.25, 0.0, 123.5, 7.25,
    ])
    let error = try {
      file.open_dataset("/special/corrupt") |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_eq(error, Some(InvalidField("Fletcher32 checksum mismatch")))
  }
}

///|
test "unknown active filters are rejected but skipped filters are allowed" {
  let filters : Array[Filter] = [{ id: 32000, parameters: [], }]
  assert_eq(decode_filters(b"abcd", filters, 1U, 1, 4), b"abcd")
  let error = try {
    decode_filters(b"abcd", filters, 0U, 1, 4) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(UnsupportedFeature(_)))
}

///|
test "deflate decoding bounds output and validates checksums" {
  let filters : Array[Filter] = [{ id: 1, parameters: [6U], }]
  let compressed = @zlib.zlib(b"abcdefgh"[:])
  assert_eq(decode_filters(compressed, filters, 0U, 1, 8), b"abcdefgh")
  let error = try {
    decode_filters(compressed, filters, 0U, 1, 1) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(InvalidField(_)))
  let corrupted = compressed.to_array()
  corrupted[corrupted.length() - 1] = corrupted[corrupted.length() - 1] ^
    b'\x01'
  let checksum_error = try {
    decode_filters(Bytes::from_array(corrupted), filters, 0U, 1, 8) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(checksum_error is Some(InvalidField(_)))
}

///|
test "filter pipeline framing and masks reject invalid input" {
  for
    data in [
      b"\x01", b"\x02\x21", b"\x02\x01\x01\x00", b"\x02\x01\x01\x00\x04\x00\x00\x00",
    ] {
    let error = try {
      parse_filters({
        msg_type: msg_filter_pipeline,
        flags: b'\x00',
        data_offset: 0,
        data,
      })
      |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(
      error is Some(InvalidField(_)) || error is Some(UnexpectedEof(_)),
    )
  }
  let error = try {
    decode_filters(b"abcd", [], 1U, 1, 4) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(InvalidField("chunk filter mask exceeds pipeline")))
  let size_error = try {
    decode_filters(b"abc", [], 0U, 1, 4) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(
    size_error,
    Some(InvalidField("decoded chunk size does not match declared shape")),
  )
}
