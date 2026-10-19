///|
pub(all) struct Datatype {
  class : Int32
  version : Int32
  size : Int32
  signed : Bool
  little_endian : Bool
  vlen_is_string : Bool
} derive(@debug.Debug)

///|
pub extend Datatype with @debug.Debug::{to_repr}

///|
fn parse_datatype_data(data : Bytes) -> Datatype raise Hdf5Error {
  let reader = Reader::new(data)
  let class_and_version = reader.read_u8().to_int()
  let class = class_and_version & 0x0F
  let version = (class_and_version >> 4) & 0x0F
  let bit_field_0 = reader.read_u8().to_int()
  let bit_field_1 = reader.read_u8().to_int()
  let bit_field_2 = reader.read_u8().to_int()
  let size = reader.read_u32_le().reinterpret_as_int()
  guard size >= 0 else { raise InvalidField("negative datatype size") }
  let signed = if class == 0 { (bit_field_0 & 0x08) != 0 } else { false }
  let little_endian = (bit_field_0 & 0x01) == 0
  let vlen_is_string = class == 9 && (bit_field_0 & 0x01) != 0
  if class == 0 || class == 1 {
    guard version >= 1 && version <= 3 else {
      raise UnsupportedFeature("unsupported numeric datatype version")
    }
    let offset = reader.read_u16_le().to_int()
    let precision = reader.read_u16_le().to_int()
    if class == 0 {
      guard (size == 1 || size == 2 || size == 4 || size == 8) &&
        offset == 0 &&
        precision == size * 8 &&
        (bit_field_0 & 0xF6) == 0 &&
        bit_field_1 == 0 &&
        bit_field_2 == 0 else {
        raise UnsupportedFeature("nonstandard integer layout is not supported")
      }
    } else {
      let exponent_position = reader.read_u8().to_int()
      let exponent_size = reader.read_u8().to_int()
      let mantissa_position = reader.read_u8().to_int()
      let mantissa_size = reader.read_u8().to_int()
      let exponent_bias = reader.read_u32_le()
      let ieee_fields = if size == 4 {
        bit_field_1 == 31 &&
        exponent_position == 23 &&
        exponent_size == 8 &&
        mantissa_size == 23 &&
        exponent_bias == 127U
      } else if size == 8 {
        bit_field_1 == 63 &&
        exponent_position == 52 &&
        exponent_size == 11 &&
        mantissa_size == 52 &&
        exponent_bias == 1023U
      } else {
        false
      }
      guard ieee_fields &&
        offset == 0 &&
        precision == size * 8 &&
        (bit_field_0 & 0xFE) == 0x20 &&
        bit_field_2 == 0 &&
        mantissa_position == 0 else {
        raise UnsupportedFeature(
          "non-IEEE floating-point layout is not supported",
        )
      }
    }
  }
  { class, version, size, signed, little_endian, vlen_is_string, }
}

///|
pub fn parse_datatype(message : HeaderMessage) -> Datatype raise Hdf5Error {
  guard message.msg_type == msg_datatype else {
    raise InvalidField("message is not a datatype message")
  }
  parse_datatype_data(message.data)
}
