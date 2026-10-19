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
  let _bit_field_1 = reader.read_u8()
  let _bit_field_2 = reader.read_u8()
  let size = reader.read_u32_le().reinterpret_as_int()
  guard size >= 0 else { raise InvalidField("negative datatype size") }
  let signed = if class == 0 { (bit_field_0 & 0x08) != 0 } else { false }
  let little_endian = (bit_field_0 & 0x01) == 0
  let vlen_is_string = class == 9 && (bit_field_0 & 0x01) != 0
  { class, version, size, signed, little_endian, vlen_is_string, }
}

///|
pub fn parse_datatype(message : HeaderMessage) -> Datatype raise Hdf5Error {
  guard message.msg_type == msg_datatype else {
    raise InvalidField("message is not a datatype message")
  }
  parse_datatype_data(message.data)
}
