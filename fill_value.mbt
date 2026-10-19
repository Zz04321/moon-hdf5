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
        "undefined or disabled chunk fill values are not supported yet",
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
  guard datatype.class == 0 || datatype.class == 1 || datatype.class == 3 else {
    raise UnsupportedFeature(
      "custom fill values require a fixed-size integer, float or string datatype",
    )
  }
  reader.read_bytes(size)
}
