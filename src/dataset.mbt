///|
/// Values decoded from an HDF5 dataset.
pub(all) enum DatasetValues {
  Integers(Array[Int64])
  UnsignedIntegers(Array[UInt64])
  Floats(Array[Double])
  Strings(Array[String])
  Enums(Array[String])
  Structured(Array[DatasetValue])
} derive(@debug.Debug)

///|
pub extend DatasetValues with @debug.Debug::{to_repr}

///|
/// One recursively decoded value for structured HDF5 datatypes.
pub(all) enum DatasetValue {
  SignedInteger(Int64)
  UnsignedInteger(UInt64)
  FloatingPoint(Double)
  Text(String)
  Enumeration(String)
  Bitfield(UInt64)
  Opaque(Bytes)
  Compound(Array[CompoundFieldValue])
  Elements(Array[DatasetValue])
  Reference(Bytes)
  Sequence(Array[DatasetValue])
  Time(UInt64)
  Complex(Double, Double)
} derive(Eq, @debug.Debug)

///|
pub extend DatasetValue with Eq::{not_equal, equal}

///|
pub extend DatasetValue with @debug.Debug::{to_repr}

///|
/// A named value decoded from a compound datatype member.
pub(all) struct CompoundFieldValue {
  name : String
  value : DatasetValue
} derive(Eq, @debug.Debug)

///|
pub extend CompoundFieldValue with Eq::{not_equal, equal}

///|
pub extend CompoundFieldValue with @debug.Debug::{to_repr}

///|
/// Dataset metadata that can be inspected without reading its stored values.
pub(all) struct DatasetMetadata {
  dataspace : Dataspace
  datatype : Datatype
  layout : DataLayout
} derive(@debug.Debug)

///|
pub extend DatasetMetadata with @debug.Debug::{to_repr}

///|
/// A dataset together with its decoded values.
pub(all) struct Dataset {
  dataspace : Dataspace
  datatype : Datatype
  layout : DataLayout
  values : DatasetValues
} derive(@debug.Debug)

///|
pub extend Dataset with @debug.Debug::{to_repr}

///|
fn datatype_class_fixed_point() -> Int32 {
  0
}

///|
fn datatype_class_floating_point() -> Int32 {
  1
}

///|
fn datatype_class_string() -> Int32 {
  3
}

///|
fn read_uint_at(
  bytes : Bytes,
  offset : Int32,
  size : Int32,
  little_endian : Bool,
) -> UInt64 raise Hdf5Error {
  guard offset >= 0 &&
    size >= 0 &&
    offset.to_int64() + size.to_int64() <= bytes.length().to_int64() else {
    raise UnexpectedEof(offset.to_int64())
  }
  guard size >= 1 && size <= 8 else {
    raise UnsupportedFeature("datatype width is outside 1..8 bytes")
  }
  let mut value = 0UL
  if little_endian {
    for index in 0..<size {
      value = value |
        (bytes.unsafe_get(offset + index).to_uint64() << (index * 8))
    }
  } else {
    for index in 0..<size {
      value = (value << 8) | bytes.unsafe_get(offset + index).to_uint64()
    }
  }
  value
}

///|
fn decode_signed_bits(raw : UInt64, bits : Int32) -> Int64 {
  if bits == 64 {
    return raw.reinterpret_as_int64()
  }
  let sign_bit = 1UL << (bits - 1)
  let value = raw.reinterpret_as_int64()
  if (raw & sign_bit) != 0UL {
    value - (1L << bits)
  } else {
    value
  }
}

///|
fn float16_to_float(bits : UInt32) -> Float {
  let sign = (bits >> 15) & 1U
  let exponent = ((bits >> 10) & 0x1FU).reinterpret_as_int()
  let fraction = bits & 0x3FFU
  let binary32 = if exponent == 0 {
    if fraction == 0U {
      sign << 31
    } else {
      let mut normalized = fraction
      let mut unbiased = -14
      while (normalized & 0x400U) == 0U {
        normalized = normalized << 1
        unbiased = unbiased - 1
      }
      (sign << 31) |
      ((unbiased + 127).reinterpret_as_uint() << 23) |
      ((normalized & 0x3FFU) << 13)
    }
  } else if exponent == 0x1F {
    (sign << 31) | (0xFFU << 23) | (fraction << 13)
  } else {
    (sign << 31) |
    ((exponent + 112).reinterpret_as_uint() << 23) |
    (fraction << 13)
  }
  Float::reinterpret_from_uint(binary32)
}

///|
fn read_float_at(
  bytes : Bytes,
  offset : Int32,
  size : Int32,
  little_endian : Bool,
) -> Double raise Hdf5Error {
  if size == 2 {
    let bits = if little_endian {
      bytes.unsafe_read_uint16_le(offset)
    } else {
      bytes.unsafe_read_uint16_be(offset)
    }
    float16_to_float(bits.to_uint()).to_double()
  } else if size == 4 {
    let bits = if little_endian {
      bytes.unsafe_read_uint32_le(offset)
    } else {
      bytes.unsafe_read_uint32_be(offset)
    }
    Float::reinterpret_from_uint(bits).to_double()
  } else if size == 8 {
    let bits = read_uint_at(bytes, offset, 8, little_endian)
    bits.reinterpret_as_double()
  } else {
    raise UnsupportedFeature("floating-point width is not 2, 4 or 8 bytes")
  }
}

///|
fn is_ieee_float_layout(datatype : Datatype) -> Bool {
  if datatype.size == 2 {
    datatype.float_sign_position == 15 &&
    datatype.float_exponent_position == 10 &&
    datatype.float_exponent_size == 5 &&
    datatype.float_mantissa_position == 0 &&
    datatype.float_mantissa_size == 10 &&
    datatype.float_exponent_bias == 15 &&
    datatype.bit_offset == 0 &&
    datatype.bit_precision == 16
  } else if datatype.size == 4 {
    datatype.float_sign_position == 31 &&
    datatype.float_exponent_position == 23 &&
    datatype.float_exponent_size == 8 &&
    datatype.float_mantissa_position == 0 &&
    datatype.float_mantissa_size == 23 &&
    datatype.float_exponent_bias == 127 &&
    datatype.bit_offset == 0 &&
    datatype.bit_precision == 32
  } else if datatype.size == 8 {
    datatype.float_sign_position == 63 &&
    datatype.float_exponent_position == 52 &&
    datatype.float_exponent_size == 11 &&
    datatype.float_mantissa_position == 0 &&
    datatype.float_mantissa_size == 52 &&
    datatype.float_exponent_bias == 1023 &&
    datatype.bit_offset == 0 &&
    datatype.bit_precision == 64
  } else {
    false
  }
}

///|
fn binary_float_to_double(raw : UInt64, datatype : Datatype) -> Double {
  let sign_bit = 0x8000000000000000UL
  let fraction_mask = 0x000FFFFFFFFFFFFFUL
  let value = masked_value(raw, datatype.bit_offset, datatype.bit_precision)
  let sign = if ((value >> datatype.float_sign_position) & 1UL) != 0UL {
    sign_bit
  } else {
    0UL
  }
  let exponent = masked_value(
    value,
    datatype.float_exponent_position,
    datatype.float_exponent_size,
  ).reinterpret_as_int64()
  let max_exponent = (1UL << datatype.float_exponent_size) - 1UL
  let mantissa = masked_value(
    value,
    datatype.float_mantissa_position,
    datatype.float_mantissa_size,
  )
  if exponent == max_exponent.reinterpret_as_int64() {
    return if mantissa == 0UL {
      (sign | 0x7FF0000000000000UL).reinterpret_as_double()
    } else {
      (sign | 0x7FF8000000000000UL).reinterpret_as_double()
    }
  }
  if exponent == 0L && mantissa == 0UL {
    return sign.reinterpret_as_double()
  }
  let mantissa_size = datatype.float_mantissa_size.to_int64()
  let mut exponent_field = 0L
  let mut fraction = 0UL
  if exponent == 0L {
    let mut highest = 0
    for bit in 0..<datatype.float_mantissa_size {
      if ((mantissa >> bit) & 1UL) != 0UL {
        highest = bit
      }
    }
    fraction = if highest <= 52 {
      (mantissa << (52 - highest)) & fraction_mask
    } else {
      (mantissa >> (highest - 52)) & fraction_mask
    }
    exponent_field = 1023L +
      1L -
      datatype.float_exponent_bias.to_int64() -
      (mantissa_size - highest.to_int64())
  } else {
    fraction = if mantissa_size <= 52L {
      (mantissa << (52 - mantissa_size.to_int())) & fraction_mask
    } else {
      mantissa >> (mantissa_size.to_int() - 52)
    }
    exponent_field = 1023L + exponent - datatype.float_exponent_bias.to_int64()
  }
  if exponent_field >= 2047L {
    return (sign | 0x7FF0000000000000UL).reinterpret_as_double()
  }
  if exponent_field <= 0L {
    let shift = 1L - exponent_field
    if shift >= 53L {
      return sign.reinterpret_as_double()
    }
    let significand = (1UL << 52) | fraction
    return (sign | (significand >> shift.to_int())).reinterpret_as_double()
  }
  (sign | (exponent_field.reinterpret_as_uint64() << 52) | fraction).reinterpret_as_double()
}

///|
fn read_float_value(
  bytes : Bytes,
  offset : Int32,
  datatype : Datatype,
) -> Double raise Hdf5Error {
  if is_ieee_float_layout(datatype) {
    read_float_at(bytes, offset, datatype.size, datatype.little_endian)
  } else {
    let raw = read_uint_at(bytes, offset, datatype.size, datatype.little_endian)
    binary_float_to_double(raw, datatype)
  }
}

///|
fn read_fixed_string_at(
  bytes : Bytes,
  offset : Int32,
  length : Int32,
  padding : Int32,
  charset : Int32,
) -> String raise Hdf5Error {
  guard offset >= 0 &&
    length >= 0 &&
    offset.to_int64() + length.to_int64() <= bytes.length().to_int64() else {
    raise UnexpectedEof(offset.to_int64())
  }
  let mut end = offset + length
  if padding == 0 {
    let mut cursor = offset
    while cursor < end && bytes.unsafe_get(cursor) != b'\x00' {
      cursor = cursor + 1
    }
    end = cursor
  } else {
    let pad = if padding == 1 { b'\x00' } else { b' ' }
    while end > offset && bytes.unsafe_get(end - 1) == pad {
      end = end - 1
    }
  }
  if charset == 0 {
    for index in offset..<end {
      guard bytes.unsafe_get(index).to_int() < 128 else {
        raise InvalidField("non-ASCII byte in ASCII string data")
      }
    }
  }
  let view = bytes.get_view(start=offset, end~).unwrap()
  @utf8.decode(view) catch {
    _ => raise InvalidField("invalid UTF-8 string data")
  }
}

///|
fn enum_value_name(raw : UInt64, datatype : Datatype) -> String raise Hdf5Error {
  guard datatype.base_types.length() == 1 &&
    datatype.base_types[0].class == datatype_class_fixed_point() else {
    raise InvalidField("enum datatype is missing its integer base type")
  }
  let base = datatype.base_types[0]
  let value = if base.signed {
    EnumValue::Signed(decode_integer_signed(raw, base))
  } else {
    EnumValue::Unsigned(decode_integer_unsigned(raw, base))
  }
  for enum_member in datatype.enum_members {
    if enum_member.value == value {
      return enum_member.name
    }
  }
  raise InvalidField("enum value does not match any member")
}

///|
fn copied_value_bytes(
  payload : Bytes,
  offset : Int32,
  size : Int32,
) -> Bytes raise Hdf5Error {
  guard offset >= 0 &&
    size >= 0 &&
    offset.to_int64() + size.to_int64() <= payload.length().to_int64() else {
    raise UnexpectedEof(offset.to_int64())
  }
  payload.get_view(start=offset, end=offset + size).unwrap().to_owned()
}

///|
fn masked_value(raw : UInt64, offset : Int32, precision : Int32) -> UInt64 {
  let shifted = raw >> offset
  if precision == 64 {
    shifted
  } else {
    shifted & ((1UL << precision) - 1UL)
  }
}

///|
fn decode_integer_unsigned(
  raw : UInt64,
  datatype : Datatype,
) -> UInt64 raise Hdf5Error {
  guard datatype.size >= 1 && datatype.size <= 8 else {
    raise UnsupportedFeature("integer width is outside 1..8 bytes")
  }
  guard datatype.bit_offset >= 0 &&
    datatype.bit_precision > 0 &&
    datatype.bit_precision <= 64 &&
    datatype.bit_offset + datatype.bit_precision <= datatype.size * 8 else {
    raise InvalidField("invalid integer offset or precision")
  }
  masked_value(raw, datatype.bit_offset, datatype.bit_precision)
}

///|
fn decode_integer_signed(
  raw : UInt64,
  datatype : Datatype,
) -> Int64 raise Hdf5Error {
  decode_signed_bits(
    decode_integer_unsigned(raw, datatype),
    datatype.bit_precision,
  )
}

///|
fn decode_dataset_value(
  source : &Hdf5Source?,
  bytes : Bytes,
  payload : Bytes,
  offset : Int32,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> DatasetValue raise Hdf5Error {
  match datatype.class {
    0 => {
      let raw = read_uint_at(
        payload,
        offset,
        datatype.size,
        datatype.little_endian,
      )
      if datatype.signed {
        SignedInteger(decode_integer_signed(raw, datatype))
      } else {
        UnsignedInteger(decode_integer_unsigned(raw, datatype))
      }
    }
    1 => FloatingPoint(read_float_value(payload, offset, datatype))
    2 => {
      let raw = read_uint_at(
        payload,
        offset,
        datatype.size,
        datatype.little_endian,
      )
      Time(masked_value(raw, 0, datatype.bit_precision))
    }
    3 =>
      Text(
        read_fixed_string_at(
          payload,
          offset,
          datatype.size,
          datatype.string_padding,
          datatype.string_charset,
        ),
      )
    4 => {
      let raw = read_uint_at(
        payload,
        offset,
        datatype.size,
        datatype.little_endian,
      )
      Bitfield(masked_value(raw, datatype.bit_offset, datatype.bit_precision))
    }
    5 => Opaque(copied_value_bytes(payload, offset, datatype.size))
    6 => {
      let fields : Array[CompoundFieldValue] = []
      for compound_member in datatype.compound_members {
        fields.push({
          name: compound_member.name,
          value: decode_dataset_value(
            source,
            bytes,
            payload,
            offset + compound_member.offset,
            compound_member.datatype,
            offset_size,
            length_size,
            base_address,
          ),
        })
      }
      Compound(fields)
    }
    7 => Reference(copied_value_bytes(payload, offset, datatype.size))
    8 => {
      let raw = read_uint_at(
        payload,
        offset,
        datatype.size,
        datatype.little_endian,
      )
      Enumeration(enum_value_name(raw, datatype))
    }
    9 => {
      let declared = read_uint_at(payload, offset, 4, true)
      guard declared <= 0x7FFFFFFFUL else {
        let description = if datatype.vlen_is_string {
          "variable-length string is too large"
        } else {
          "variable-length sequence is too large"
        }
        raise InvalidField(description)
      }
      let count = declared.to_int()
      if count == 0 {
        if datatype.vlen_is_string {
          Text("")
        } else {
          Sequence([])
        }
      } else {
        let heap_address = read_uint_at(payload, offset + 4, offset_size, true)
        let object_index = read_uint_at(
          payload,
          offset + 4 + offset_size,
          4,
          true,
        )
        guard object_index <= 0x7FFFFFFFUL else {
          raise InvalidField("global heap object index is too large")
        }
        let data = match source {
          Some(random_access) =>
            read_source_global_heap_object(
              random_access,
              heap_address,
              object_index.to_int(),
              length_size,
              base_address,
            )
          None =>
            read_global_heap_object(
              bytes,
              heap_address,
              object_index.to_int(),
              offset_size,
              length_size,
              base_address,
            )
        }
        if datatype.vlen_is_string {
          guard data.length() >= count else {
            raise InvalidField(
              "global heap string is shorter than declared length",
            )
          }
          Text(read_fixed_string_at(data, 0, count, 1, datatype.string_charset))
        } else {
          guard datatype.base_types.length() == 1 else {
            raise InvalidField(
              "variable-length datatype is missing its base type",
            )
          }
          let base = datatype.base_types[0]
          guard count.to_int64() <= 0x7FFFFFFFL / base.size.to_int64() &&
            data.length().to_int64() >= count.to_int64() * base.size.to_int64() else {
            raise InvalidField(
              "global heap sequence is shorter than declared length",
            )
          }
          let values : Array[DatasetValue] = []
          for index in 0..<count {
            values.push(
              decode_dataset_value(
                source,
                bytes,
                data,
                index * base.size,
                base,
                offset_size,
                length_size,
                base_address,
              ),
            )
          }
          Sequence(values)
        }
      }
    }
    10 => {
      guard datatype.base_types.length() == 1 else {
        raise InvalidField("array datatype is missing its base type")
      }
      let base = datatype.base_types[0]
      guard datatype.size % base.size == 0 else {
        raise InvalidField(
          "array datatype size is not divisible by its base type",
        )
      }
      let values : Array[DatasetValue] = []
      for index in 0..<(datatype.size / base.size) {
        values.push(
          decode_dataset_value(
            source,
            bytes,
            payload,
            offset + index * base.size,
            base,
            offset_size,
            length_size,
            base_address,
          ),
        )
      }
      Elements(values)
    }
    11 => {
      guard datatype.base_types.length() == 1 else {
        raise InvalidField("complex datatype is missing its base type")
      }
      let base = datatype.base_types[0]
      Complex(
        read_float_value(payload, offset, base),
        read_float_value(payload, offset + base.size, base),
      )
    }
    _ => raise UnsupportedFeature("datatype class is not supported yet")
  }
}

///|
fn decode_values(
  bytes : Bytes,
  payload : Bytes,
  count : Int64,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  source~ : &Hdf5Source? = None,
) -> DatasetValues raise Hdf5Error {
  guard count >= 0 && count <= 0x7FFFFFFFL else {
    raise InvalidField("dataset has too many elements")
  }
  let n = count.to_int()
  guard datatype.size > 0 else { raise InvalidField("invalid datatype size") }
  guard payload.length().to_int64() >= count * datatype.size.to_int64() else {
    raise InvalidField("dataset storage is smaller than its declared shape")
  }
  match datatype.class {
    c if c == datatype_class_fixed_point() =>
      if datatype.signed {
        let values : Array[Int64] = []
        for i in 0..<n {
          let raw = read_uint_at(
            payload,
            i * datatype.size,
            datatype.size,
            datatype.little_endian,
          )
          values.push(decode_integer_signed(raw, datatype))
        }
        Integers(values)
      } else {
        let values : Array[UInt64] = []
        for i in 0..<n {
          let raw = read_uint_at(
            payload,
            i * datatype.size,
            datatype.size,
            datatype.little_endian,
          )
          values.push(decode_integer_unsigned(raw, datatype))
        }
        UnsignedIntegers(values)
      }
    c if c == datatype_class_floating_point() => {
      let values : Array[Double] = []
      for i in 0..<n {
        values.push(read_float_value(payload, i * datatype.size, datatype))
      }
      Floats(values)
    }
    c if c == datatype_class_string() => {
      let values : Array[String] = []
      for i in 0..<n {
        values.push(
          read_fixed_string_at(
            payload,
            i * datatype.size,
            datatype.size,
            datatype.string_padding,
            datatype.string_charset,
          ),
        )
      }
      Strings(values)
    }
    8 => {
      let values : Array[String] = []
      for i in 0..<n {
        let raw = read_uint_at(
          payload,
          i * datatype.size,
          datatype.size,
          datatype.little_endian,
        )
        values.push(enum_value_name(raw, datatype))
      }
      Enums(values)
    }
    9 if datatype.vlen_is_string => {
      let values : Array[String] = []
      for i in 0..<n {
        match
          decode_dataset_value(
            source,
            bytes,
            payload,
            i * datatype.size,
            datatype,
            offset_size,
            length_size,
            base_address,
          ) {
          Text(value) => values.push(value)
          _ => raise InvalidField("invalid variable-length string value")
        }
      }
      Strings(values)
    }
    _ => {
      let values : Array[DatasetValue] = []
      for i in 0..<n {
        values.push(
          decode_dataset_value(
            source,
            bytes,
            payload,
            i * datatype.size,
            datatype,
            offset_size,
            length_size,
            base_address,
          ),
        )
      }
      Structured(values)
    }
  }
}

///|
fn message_first(
  header : ObjectHeader,
  msg_type : Int32,
) -> HeaderMessage raise Hdf5Error {
  match header.first(msg_type) {
    Some(message) => message
    None =>
      raise InvalidField("dataset is missing a required object header message")
  }
}

///|
/// Parses the dataspace, datatype and storage layout from a dataset object
/// header without reading the stored values.
fn read_dataset_metadata(
  header : ObjectHeader,
  offset_size : Int32,
  length_size : Int32,
) -> DatasetMetadata raise Hdf5Error {
  let dataspace = parse_dataspace(
    message_first(header, msg_dataspace),
    length_size,
  )
  let datatype = parse_datatype(message_first(header, msg_datatype))
  let layout = parse_data_layout(
    message_first(header, msg_data_layout),
    offset_size,
    length_size,
  )
  { dataspace, datatype, layout, }
}

///|
fn read_dataset_values(
  bytes : Bytes,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> DatasetValues raise Hdf5Error {
  let dataspace = metadata.dataspace
  let datatype = metadata.datatype
  let layout = metadata.layout
  let filters = match header.first(msg_filter_pipeline) {
    Some(message) => parse_filters(message)
    None => []
  }
  guard filters.length() == 0 || layout.layout_class == 2 else {
    raise UnsupportedFeature("filters require chunked storage")
  }
  if header.first(msg_external_data_files) is Some(_) {
    raise UnsupportedFeature("external dataset storage requires Hdf5Resolver")
  }
  let count = dataspace.element_count()
  guard datatype.size > 0 &&
    count >= 0 &&
    count <= 0x7FFFFFFFL / datatype.size.to_int64() else {
    raise InvalidField("dataset byte size is too large")
  }
  let payload = match layout.layout_class {
    0 => layout.compact_data
    1 => {
      let undefined = if offset_size == 8 {
        0xFFFFFFFFFFFFFFFFUL
      } else {
        (1UL << (offset_size * 8)) - 1UL
      }
      if count == 0 {
        b""
      } else if layout.data_address == undefined {
        filled_payload(
          count,
          datatype.size,
          dataset_fill_value(header, datatype),
        ).unsafe_reinterpret_as_bytes()
      } else {
        let stored_size = if layout.data_size > 0UL {
          layout.data_size
        } else {
          count.reinterpret_as_uint64() * datatype.size.to_uint64()
        }
        guard stored_size <= 0x7FFFFFFFUL else {
          raise InvalidField("contiguous dataset is too large")
        }
        let offset = address_to_offset(layout.data_address, base_address)
        let reader = Reader::at(bytes, offset)
        reader.read_bytes(stored_size.to_int())
      }
    }
    2 => {
      let fill_value = dataset_fill_value(header, datatype)
      read_chunked_with_filters(
        bytes, layout, dataspace, datatype, offset_size, base_address, fill_value,
        filters, length_size,
      )
    }
    _ => raise UnsupportedFeature("unknown data layout class")
  }
  let values = decode_values(
    bytes, payload, count, datatype, offset_size, length_size, base_address,
  )
  values
}

///|
/// Parses a dataset object header and decodes compact, contiguous or
/// chunked data with supported indexes and filters.
fn read_dataset(
  bytes : Bytes,
  header : ObjectHeader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Dataset raise Hdf5Error {
  let metadata = read_dataset_metadata(header, offset_size, length_size)
  let values = read_dataset_values(
    bytes, header, metadata, offset_size, length_size, base_address,
  )
  {
    dataspace: metadata.dataspace,
    datatype: metadata.datatype,
    layout: metadata.layout,
    values,
  }
}
