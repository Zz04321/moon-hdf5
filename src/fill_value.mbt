///|
fn datatype_contains_vlen(datatype : Datatype) -> Bool {
  if datatype.class == 9 {
    return true
  }
  for field in datatype.compound_members {
    if datatype_contains_vlen(field.datatype) {
      return true
    }
  }
  for base in datatype.base_types {
    if datatype_contains_vlen(base) {
      return true
    }
  }
  false
}

///|
fn dataset_fill_value(
  header : ObjectHeader,
  datatype : Datatype,
) -> Bytes raise Hdf5Error {
  let message = match header.first(msg_fill_value) {
    Some(message) => message
    None =>
      match header.first(msg_fill_value_old) {
        Some(message) => message
        None => return b""
      }
  }
  guard (message.flags.to_int() & 0x02) == 0 else {
    raise UnsupportedFeature("shared fill value messages are not supported yet")
  }
  let reader = Reader::new(message.data)
  if message.msg_type == msg_fill_value {
    let version = reader.read_u8().to_int()
    let (allocation, writing, defined, has_value) = match version {
      1 | 2 => {
        let allocation = reader.read_u8().to_int()
        let writing = reader.read_u8().to_int()
        let defined = reader.read_u8().to_int()
        guard defined <= 1 else {
          raise InvalidField("invalid fill value definition")
        }
        (allocation, writing, defined == 1, defined == 1)
      }
      3 => {
        let flags = reader.read_u8().to_int()
        guard (flags & 0xC0) == 0 && (flags & 0x30) != 0x30 else {
          raise InvalidField("invalid fill value flags")
        }
        (flags & 3, (flags >> 2) & 3, (flags & 0x10) == 0, (flags & 0x20) != 0)
      }
      _ => raise UnsupportedFeature("unsupported fill value message version")
    }
    guard allocation >= 1 && allocation <= 3 && writing <= 2 else {
      raise InvalidField("invalid fill value allocation or writing policy")
    }
    guard defined && writing != 1 else {
      raise UnsupportedFeature(
        "undefined or disabled fill values are not supported yet",
      )
    }
    if !has_value {
      return b""
    }
  }
  let size = reader.read_u32_le().reinterpret_as_int()
  guard size >= 0 else { raise InvalidField("invalid fill value size") }
  if size == 0 {
    return b""
  }
  guard size == datatype.size else {
    raise InvalidField("fill value size does not match datatype")
  }
  guard !datatype_contains_vlen(datatype) else {
    raise UnsupportedFeature(
      "custom variable-length fill values are not supported yet",
    )
  }
  reader.read_bytes(size)
}

///|
fn filled_payload(
  count : Int64,
  size : Int32,
  fill : Bytes,
) -> FixedArray[Byte] raise Hdf5Error {
  guard count >= 0 && size > 0 && count <= 0x7FFFFFFFL / size.to_int64() else {
    raise InvalidField("dataset byte size is too large")
  }
  guard fill.length() == 0 || fill.length() == size else {
    raise InvalidField("fill value size does not match datatype")
  }
  let output = FixedArray::make(count.to_int() * size, b'\x00')
  if fill.length() > 0 {
    for element in 0..<count.to_int() {
      output.blit_from_bytes(element * size, fill, 0, size)
    }
  }
  output
}
