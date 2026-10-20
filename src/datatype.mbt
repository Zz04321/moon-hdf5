///|
/// One value in an enumeration datatype, preserving the signedness of its
/// integer base type.
pub(all) enum EnumValue {
  Signed(Int64)
  Unsigned(UInt64)
} derive(Eq, @debug.Debug)

///|
pub extend EnumValue with Eq::{not_equal, equal}

///|
pub extend EnumValue with @debug.Debug::{to_repr}

///|
/// A named member of an enumeration datatype.
pub(all) struct EnumMember {
  name : String
  value : EnumValue
} derive(@debug.Debug)

///|
pub extend EnumMember with @debug.Debug::{to_repr}

///|
pub(all) struct Datatype {
  class : Int32
  version : Int32
  size : Int32
  signed : Bool
  little_endian : Bool
  vlen_is_string : Bool
  enum_members : Array[EnumMember]
} derive(@debug.Debug)

///|
pub extend Datatype with @debug.Debug::{to_repr}

///|
fn parse_datatype_reader(
  reader : Reader,
  depth : Int32,
) -> Datatype raise Hdf5Error {
  guard depth < 16 else { raise InvalidField("datatype nesting is too deep") }
  let class_and_version = reader.read_u8().to_int()
  let class = class_and_version & 0x0F
  let version = (class_and_version >> 4) & 0x0F
  let bit_field_0 = reader.read_u8().to_int()
  let bit_field_1 = reader.read_u8().to_int()
  let bit_field_2 = reader.read_u8().to_int()
  let size = reader.read_u32_le().reinterpret_as_int()
  guard size > 0 else { raise InvalidField("invalid datatype size") }
  let mut signed = if class == 0 { (bit_field_0 & 0x08) != 0 } else { false }
  let mut little_endian = (bit_field_0 & 0x01) == 0
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
  let enum_members : Array[EnumMember] = if class == 8 {
    guard version >= 1 && version <= 3 else {
      raise UnsupportedFeature("unsupported enum datatype version")
    }
    let member_count = bit_field_0 | (bit_field_1 << 8)
    guard member_count > 0 else {
      raise InvalidField("enum datatype has no members")
    }
    let base = parse_datatype_reader(reader, depth + 1)
    guard base.class == 0 else {
      raise UnsupportedFeature("enum base type is not an integer")
    }
    guard base.size == size else {
      raise InvalidField("enum datatype size does not match its base type")
    }
    signed = base.signed
    little_endian = base.little_endian
    let names : Array[String] = []
    for _ in 0..<member_count {
      let start = reader.position()
      let mut end = start
      while end < reader.limit && reader.bytes.unsafe_get(end) != b'\x00' {
        end = end + 1
      }
      guard end < reader.limit else { raise UnexpectedEof(start.to_int64()) }
      let name_length = end - start
      guard name_length > 0 else {
        raise InvalidField("enum member name is empty")
      }
      let name_bytes = reader.read_bytes(name_length)
      reader.skip(1)
      let name = @utf8.decode(name_bytes.get_view().unwrap()) catch {
        _ => raise InvalidField("invalid UTF-8 enum member name")
      }
      names.push(name)
      if version < 3 {
        let used = name_length + 1
        let padded = (used + 7) / 8 * 8
        reader.skip(padded - used)
      }
    }
    guard base.size <= 0x7FFFFFFF / member_count else {
      raise InvalidField("enum datatype is too large")
    }
    let raw_values = reader.read_bytes(member_count * base.size)
    let members : Array[EnumMember] = []
    for index in 0..<member_count {
      let raw = read_uint_at(
        raw_values,
        index * base.size,
        base.size,
        base.little_endian,
      )
      let value = if base.signed {
        EnumValue::Signed(decode_signed(raw, base.size))
      } else {
        EnumValue::Unsigned(raw)
      }
      members.push({ name: names[index], value, })
    }
    members
  } else {
    []
  }
  { class, version, size, signed, little_endian, vlen_is_string, enum_members, }
}

///|
fn parse_datatype_data(data : Bytes) -> Datatype raise Hdf5Error {
  parse_datatype_reader(Reader::new(data), 0)
}

///|
pub fn parse_datatype(message : HeaderMessage) -> Datatype raise Hdf5Error {
  guard message.msg_type == msg_datatype else {
    raise InvalidField("message is not a datatype message")
  }
  parse_datatype_data(message.data)
}
