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
/// A named field within a compound datatype.
pub(all) struct CompoundMember {
  name : String
  offset : Int32
  datatype : Datatype
} derive(@debug.Debug)

///|
pub extend CompoundMember with @debug.Debug::{to_repr}

///|
pub(all) struct Datatype {
  class : Int32
  version : Int32
  size : Int32
  signed : Bool
  little_endian : Bool
  vlen_is_string : Bool
  enum_members : Array[EnumMember]
  compound_members : Array[CompoundMember]
  base_types : Array[Datatype]
  array_dimensions : Array[UInt64]
  bit_offset : Int32
  bit_precision : Int32
  string_padding : Int32
  string_charset : Int32
  opaque_tag : String
  reference_kind : Int32
} derive(@debug.Debug)

///|
pub extend Datatype with @debug.Debug::{to_repr}

///|
fn read_datatype_name(
  reader : Reader,
  padded : Bool,
  description : String,
) -> String raise Hdf5Error {
  let start = reader.position()
  let mut end = start
  while end < reader.limit && reader.bytes.unsafe_get(end) != b'\x00' {
    end = end + 1
  }
  guard end < reader.limit else { raise UnexpectedEof(start.to_int64()) }
  guard end > start else { raise InvalidField(description + " is empty") }
  let length = end - start
  let raw = reader.read_bytes(length)
  reader.skip(1)
  if padded {
    let used = length + 1
    reader.skip((used + 7) / 8 * 8 - used)
  }
  @utf8.decode(raw.get_view().unwrap()) catch {
    _ => raise InvalidField("invalid UTF-8 " + description)
  }
}

///|
fn datatype_offset_width(size : Int32) -> Int32 {
  if size <= 0x100 {
    1
  } else if size <= 0x10000 {
    2
  } else {
    4
  }
}

///|
fn datatype_array_size(
  dimensions : Array[UInt64],
  base_size : Int32,
) -> Int32 raise Hdf5Error {
  let mut size = base_size.to_int64()
  for dimension in dimensions {
    guard dimension > 0UL &&
      dimension <= 0x7FFFFFFFUL &&
      size <= 0x7FFFFFFFL / dimension.reinterpret_as_int64() else {
      raise InvalidField("array datatype size is too large")
    }
    size = size * dimension.reinterpret_as_int64()
  }
  size.to_int()
}

///|
fn array_datatype(
  version : Int32,
  dimensions : Array[UInt64],
  base : Datatype,
) -> Datatype raise Hdf5Error {
  {
    class: 10,
    version,
    size: datatype_array_size(dimensions, base.size),
    signed: false,
    little_endian: true,
    vlen_is_string: false,
    enum_members: [],
    compound_members: [],
    base_types: [base],
    array_dimensions: dimensions,
    bit_offset: 0,
    bit_precision: 0,
    string_padding: 0,
    string_charset: 0,
    opaque_tag: "",
    reference_kind: 0,
  }
}

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
  let mut vlen_is_string = false
  let enum_members : Array[EnumMember] = []
  let compound_members : Array[CompoundMember] = []
  let base_types : Array[Datatype] = []
  let array_dimensions : Array[UInt64] = []
  let mut bit_offset = 0
  let mut bit_precision = 0
  let mut string_padding = 0
  let mut string_charset = 0
  let mut opaque_tag = ""
  let mut reference_kind = 0
  if class == 0 || class == 1 {
    guard version >= 1 && version <= 3 else {
      raise UnsupportedFeature("unsupported numeric datatype version")
    }
    let offset = reader.read_u16_le().to_int()
    let precision = reader.read_u16_le().to_int()
    bit_offset = offset
    bit_precision = precision
    if class == 0 {
      guard size == 1 || size == 2 || size == 4 || size == 8 else {
        raise UnsupportedFeature("integer width is not 1, 2, 4 or 8 bytes")
      }
      guard (bit_field_0 & 0xF6) == 0 && bit_field_1 == 0 && bit_field_2 == 0 else {
        raise UnsupportedFeature("integer datatype flags are not supported")
      }
      guard precision > 0 && offset + precision <= size * 8 else {
        raise InvalidField("invalid integer offset or precision")
      }
    } else {
      let exponent_position = reader.read_u8().to_int()
      let exponent_size = reader.read_u8().to_int()
      let mantissa_position = reader.read_u8().to_int()
      let mantissa_size = reader.read_u8().to_int()
      let exponent_bias = reader.read_u32_le()
      let ieee_fields = if size == 2 {
        bit_field_1 == 15 &&
        exponent_position == 10 &&
        exponent_size == 5 &&
        mantissa_size == 10 &&
        exponent_bias == 15U
      } else if size == 4 {
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
  } else if class == 2 {
    guard version >= 1 &&
      version <= 3 &&
      (bit_field_0 & 0xFE) == 0 &&
      bit_field_1 == 0 &&
      bit_field_2 == 0 &&
      (size == 1 || size == 2 || size == 4 || size == 8) else {
      raise UnsupportedFeature("unsupported time datatype layout")
    }
    bit_precision = reader.read_u16_le().to_int()
    guard bit_precision > 0 && bit_precision <= size * 8 else {
      raise InvalidField("invalid time datatype precision")
    }
  } else if class == 3 {
    string_padding = bit_field_0 & 0x0F
    string_charset = (bit_field_0 >> 4) & 0x0F
    guard version >= 1 &&
      version <= 3 &&
      string_padding <= 2 &&
      string_charset <= 1 &&
      bit_field_1 == 0 &&
      bit_field_2 == 0 else {
      raise UnsupportedFeature("unsupported string datatype layout")
    }
  } else if class == 4 {
    guard version >= 1 &&
      version <= 3 &&
      (bit_field_0 & 0xF8) == 0 &&
      bit_field_1 == 0 &&
      bit_field_2 == 0 &&
      (size == 1 || size == 2 || size == 4 || size == 8) else {
      raise UnsupportedFeature("unsupported bitfield datatype layout")
    }
    bit_offset = reader.read_u16_le().to_int()
    bit_precision = reader.read_u16_le().to_int()
    guard bit_precision > 0 && bit_offset + bit_precision <= size * 8 else {
      raise InvalidField("invalid bitfield offset or precision")
    }
  } else if class == 5 {
    guard version >= 1 && version <= 3 && bit_field_1 == 0 && bit_field_2 == 0 else {
      raise UnsupportedFeature("unsupported opaque datatype version")
    }
    let tag_length = bit_field_0
    if tag_length > 0 {
      let padded = (tag_length + 7) / 8 * 8
      let raw = reader.read_bytes(padded)
      let mut end = 0
      while end < tag_length && raw.unsafe_get(end) != b'\x00' {
        guard raw.unsafe_get(end).to_int() < 128 else {
          raise InvalidField("opaque datatype tag is not ASCII")
        }
        end = end + 1
      }
      guard end < tag_length else {
        raise InvalidField("opaque datatype tag is not terminated")
      }
      opaque_tag = @utf8.decode(raw.get_view(start=0, end~).unwrap()) catch {
        _ => raise InvalidField("invalid opaque datatype tag")
      }
    }
  } else if class == 6 {
    guard version >= 1 && (version <= 4 || version == 5) && bit_field_2 == 0 else {
      raise UnsupportedFeature("unsupported compound datatype version")
    }
    let member_count = bit_field_0 | (bit_field_1 << 8)
    guard member_count > 0 else {
      raise InvalidField("compound datatype has no members")
    }
    for _ in 0..<member_count {
      let name = read_datatype_name(reader, version < 3, "compound member name")
      let member_offset = if version < 3 {
        reader.read_u32_le().to_uint64()
      } else {
        reader.read_uint_le(datatype_offset_width(size))
      }
      guard member_offset <= 0x7FFFFFFFUL else {
        raise InvalidField("compound member offset is too large")
      }
      let legacy_dimensions : Array[UInt64] = []
      if version == 1 {
        let rank = reader.read_u8().to_int()
        guard rank <= 4 && reader.read_bytes(3) == b"\x00\x00\x00" else {
          raise InvalidField("invalid legacy compound member dimensions")
        }
        guard reader.read_u32_le() == 0U && reader.read_u32_le() == 0U else {
          raise UnsupportedFeature("permuted compound arrays are not supported")
        }
        for axis in 0..<4 {
          let dimension = reader.read_u32_le().to_uint64()
          if axis < rank {
            legacy_dimensions.push(dimension)
          }
        }
      }
      let parsed = parse_datatype_reader(reader, depth + 1)
      let member_type = if legacy_dimensions.length() > 0 {
        array_datatype(version, legacy_dimensions, parsed)
      } else {
        parsed
      }
      guard member_offset + member_type.size.to_uint64() <= size.to_uint64() else {
        raise InvalidField("compound member extends beyond datatype size")
      }
      for existing in compound_members {
        guard existing.name != name else {
          raise InvalidField("duplicate compound member name")
        }
      }
      compound_members.push({
        name,
        offset: member_offset.to_int(),
        datatype: member_type,
      })
    }
  } else if class == 7 {
    reference_kind = bit_field_0 & 0x0F
    if version < 4 {
      guard version >= 1 &&
        reference_kind <= 1 &&
        (bit_field_0 & 0xF0) == 0 &&
        bit_field_1 == 0 &&
        bit_field_2 == 0 else {
        raise UnsupportedFeature("unsupported reference datatype layout")
      }
    } else {
      guard version == 4 &&
        (reference_kind == 2 || reference_kind == 3 || reference_kind == 4) &&
        bit_field_0 >> 4 == 1 &&
        bit_field_1 == 0 &&
        bit_field_2 == 0 else {
        raise UnsupportedFeature("unsupported reference datatype layout")
      }
    }
  } else if class == 8 {
    guard version >= 1 && version <= 4 && bit_field_2 == 0 else {
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
    base_types.push(base)
    let names : Array[String] = []
    for _ in 0..<member_count {
      names.push(read_datatype_name(reader, version < 3, "enum member name"))
    }
    guard base.size <= 0x7FFFFFFF / member_count else {
      raise InvalidField("enum datatype is too large")
    }
    let raw_values = reader.read_bytes(member_count * base.size)
    for index in 0..<member_count {
      let raw = read_uint_at(
        raw_values,
        index * base.size,
        base.size,
        base.little_endian,
      )
      let value = if base.signed {
        EnumValue::Signed(decode_integer_signed(raw, base))
      } else {
        EnumValue::Unsigned(decode_integer_unsigned(raw, base))
      }
      enum_members.push({ name: names[index], value, })
    }
  } else if class == 9 {
    let kind = bit_field_0 & 0x0F
    vlen_is_string = kind == 1
    string_padding = (bit_field_0 >> 4) & 0x0F
    string_charset = bit_field_1 & 0x0F
    guard version >= 1 &&
      version <= 4 &&
      (kind == 0 || kind == 1) &&
      (bit_field_1 & 0xF0) == 0 &&
      bit_field_2 == 0 &&
      (
        (kind == 0 && string_padding == 0 && string_charset == 0) ||
        (kind == 1 && string_padding <= 2 && string_charset <= 1)
      ) else {
      raise UnsupportedFeature("unsupported variable-length datatype layout")
    }
    base_types.push(parse_datatype_reader(reader, depth + 1))
  } else if class == 10 {
    guard version >= 2 &&
      version <= 5 &&
      bit_field_0 == 0 &&
      bit_field_1 == 0 &&
      bit_field_2 == 0 else {
      raise UnsupportedFeature("unsupported array datatype version")
    }
    let rank = reader.read_u8().to_int()
    guard rank > 0 && rank <= 32 else {
      raise InvalidField("invalid array datatype rank")
    }
    if version == 2 {
      guard reader.read_bytes(3) == b"\x00\x00\x00" else {
        raise InvalidField("invalid array datatype reserved bytes")
      }
    }
    for _ in 0..<rank {
      array_dimensions.push(reader.read_u32_le().to_uint64())
    }
    if version == 2 {
      for axis in 0..<rank {
        guard reader.read_u32_le().reinterpret_as_int() == axis else {
          raise UnsupportedFeature("permuted array datatypes are not supported")
        }
      }
    }
    let base = parse_datatype_reader(reader, depth + 1)
    guard datatype_array_size(array_dimensions, base.size) == size else {
      raise InvalidField("array datatype size does not match its base type")
    }
    base_types.push(base)
  } else if class == 11 {
    guard (version == 4 || version == 5) &&
      bit_field_0 == 1 &&
      bit_field_1 == 0 &&
      bit_field_2 == 0 else {
      raise UnsupportedFeature("unsupported complex datatype layout")
    }
    let base = parse_datatype_reader(reader, depth + 1)
    guard base.class == 1 &&
      base.size <= 0x7FFFFFFF / 2 &&
      size == base.size * 2 else {
      raise InvalidField("complex datatype size does not match its base type")
    }
    base_types.push(base)
  } else {
    raise UnsupportedFeature("unknown datatype class")
  }
  {
    class,
    version,
    size,
    signed,
    little_endian,
    vlen_is_string,
    enum_members,
    compound_members,
    base_types,
    array_dimensions,
    bit_offset,
    bit_precision,
    string_padding,
    string_charset,
    opaque_tag,
    reference_kind,
  }
}

///|
fn parse_datatype_data(data : Bytes) -> Datatype raise Hdf5Error {
  parse_datatype_reader(Reader::new(data), 0)
}

///|
fn parse_datatype(message : HeaderMessage) -> Datatype raise Hdf5Error {
  guard message.msg_type == msg_datatype else {
    raise InvalidField("message is not a datatype message")
  }
  parse_datatype_data(message.data)
}
