///|
priv struct Filter {
  id : Int32
  parameters : Array[UInt32]
}

///|
fn parse_filters(message : HeaderMessage) -> Array[Filter] raise Hdf5Error {
  guard (message.flags.to_int() & 2) == 0 else {
    raise UnsupportedFeature("shared filter pipeline messages")
  }
  let reader = Reader::new(message.data)
  let version = reader.read_u8().to_int()
  guard version == 1 || version == 2 else {
    raise UnsupportedFeature("unsupported filter pipeline version")
  }
  let count = reader.read_u8().to_int()
  guard count <= 32 else { raise InvalidField("too many filters") }
  if version == 1 {
    reader.skip(6)
  }
  let filters : Array[Filter] = []
  for _ in 0..<count {
    let id = reader.read_u16_le().to_int()
    let name_size = if version == 1 || id >= 256 {
      reader.read_u16_le().to_int()
    } else {
      0
    }
    let flags = reader.read_u16_le().to_int()
    guard (flags & 0xFFFE) == 0 else {
      raise InvalidField("invalid filter flags")
    }
    let parameter_count = reader.read_u16_le().to_int()
    reader.skip(name_size)
    let parameters : Array[UInt32] = []
    for _ in 0..<parameter_count {
      parameters.push(reader.read_u32_le())
    }
    if version == 1 && parameter_count % 2 != 0 {
      reader.skip(4)
    }
    filters.push({ id, parameters, })
  }
  filters
}

///|
fn strip_fletcher32(data : Bytes) -> Bytes raise Hdf5Error {
  guard data.length() >= 4 else {
    raise InvalidField("truncated Fletcher32 chunk")
  }
  let size = data.length() - 4
  let mut first = 0U
  let mut second = 0U
  let mut offset = 0
  while offset < size {
    let high = data[offset].to_uint() << 8
    let low = if offset + 1 < size { data[offset + 1].to_uint() } else { 0U }
    first = first + high + low
    first = (first & 0xFFFFU) + (first >> 16)
    second = second + first
    second = (second & 0xFFFFU) + (second >> 16)
    offset = offset + 2
  }
  first = (first & 0xFFFFU) + (first >> 16)
  second = (second & 0xFFFFU) + (second >> 16)
  let checksum = (second << 16) | first
  let legacy = ((checksum & 0x00FF00FFU) << 8) | ((checksum & 0xFF00FF00U) >> 8)
  let stored = Reader::at(data, size).read_u32_le()
  guard stored == checksum || stored == legacy else {
    raise InvalidField("Fletcher32 checksum mismatch")
  }
  data.get_view(start=0, end=size).unwrap().to_owned()
}

///|
fn scaleoffset_parameter_value(
  parameters : Array[UInt32],
  size : Int32,
) -> UInt64 {
  let mut value = 0UL
  for index in 0..<size {
    let word = parameters[8 + index / 4].to_uint64()
    let byte = (word >> (index % 4 * 8)) & 0xFFUL
    value = value | (byte << (index * 8))
  }
  value
}

///|
fn scaleoffset_write_scalar(
  output : FixedArray[Byte],
  offset : Int32,
  size : Int32,
  order : Int32,
  value : UInt64,
) -> Unit {
  for byte_index in 0..<size {
    let shift = if order == 0 {
      byte_index * 8
    } else {
      (size - 1 - byte_index) * 8
    }
    output[offset + byte_index] = ((value >> shift) & 0xFFUL).to_byte()
  }
}

///|
fn scaleoffset_float_factor32(digits : Int32) -> Float {
  @math.powf(Float(10.0), Float::from_int(digits))
}

///|
fn scaleoffset_float_factor64(digits : Int32) -> Double {
  @math.pow(10.0, Double::from_int(digits))
}

///|
fn decode_scaleoffset(
  data : Bytes,
  filter : Filter,
  element_size : Int32,
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  guard element_size > 0 &&
    expected_size >= 0 &&
    expected_size % element_size == 0 else {
    raise InvalidField("invalid scale-offset chunk size")
  }
  guard filter.parameters.length() == 20 else {
    raise UnsupportedFeature("unsupported scale-offset parameters")
  }
  let scale_type = filter.parameters[0].reinterpret_as_int()
  let datatype_class = filter.parameters[3].reinterpret_as_int()
  let datatype_size = filter.parameters[4].reinterpret_as_int()
  let sign = filter.parameters[5].reinterpret_as_int()
  let order = filter.parameters[6].reinterpret_as_int()
  let fill_available = filter.parameters[7].reinterpret_as_int()
  let filter_value = filter.parameters[1].reinterpret_as_int()
  guard datatype_size == element_size &&
    (sign == 0 || sign == 1) &&
    (order == 0 || order == 1) &&
    (fill_available == 0 || fill_available == 1) else {
    raise InvalidField("invalid scale-offset filter parameters")
  }
  let is_integer = datatype_class == 0 && scale_type == 2
  let is_float = datatype_class == 1 &&
    scale_type == 0 &&
    (element_size == 4 || element_size == 8)
  guard is_integer || is_float else {
    raise UnsupportedFeature("scale-offset datatype class is not supported")
  }
  if is_integer && filter_value == element_size * 8 {
    guard data.length() == expected_size else {
      raise InvalidField("unfiltered scale-offset chunk has the wrong size")
    }
    return data
  }
  guard data.length() >= 21 else {
    raise InvalidField("truncated scale-offset chunk")
  }
  let reader = Reader::new(data)
  let minbits = reader.read_u32_le().reinterpret_as_int()
  let minval_size = data[4].to_int()
  guard minbits >= 0 &&
    minbits < element_size * 8 &&
    minval_size >= 0 &&
    minval_size <= 8 &&
    (!is_float || minval_size >= element_size) else {
    raise InvalidField("invalid scale-offset chunk header")
  }
  let mut minval = 0UL
  for index in 0..<minval_size {
    minval = minval | (data[5 + index].to_uint64() << (index * 8))
  }
  let count = expected_size / element_size
  let element_bits = minbits.to_int64()
  let total_bits = count.to_int64() * element_bits
  guard total_bits >= 0 &&
    total_bits <= 0x7FFFFFFF &&
    (total_bits + 7) / 8 <= (data.length() - 21).to_int64() else {
    raise InvalidField("truncated scale-offset bit stream")
  }
  let fill = if fill_available == 1 {
    scaleoffset_parameter_value(filter.parameters, element_size)
  } else {
    0UL
  }
  let maximum_digits = if element_size == 4 { 38 } else { 308 }
  guard !is_float || (filter_value >= 0 && filter_value <= maximum_digits) else {
    raise UnsupportedFeature("unsupported floating-point scale factor")
  }
  let minimum = if is_float {
    read_float_at(data, 5, element_size, true)
  } else {
    0.0
  }
  let factor32 = if is_float && element_size == 4 {
    scaleoffset_float_factor32(filter_value)
  } else {
    Float(1.0)
  }
  let factor64 = if is_float && element_size == 8 {
    scaleoffset_float_factor64(filter_value)
  } else {
    1.0
  }
  let marker = if minbits == 0 { 0UL } else { (1UL << minbits) - 1UL }
  let mask = if element_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (element_size * 8)) - 1UL
  }
  let output = FixedArray::make(expected_size, b'\x00')
  let mut bit_index = 0
  for element in 0..<count {
    let mut packed = 0UL
    let mut remaining = minbits
    while remaining > 0 {
      let byte_index = bit_index / 8
      let bit_offset = bit_index % 8
      let available = 8 - bit_offset
      let take = if remaining < available { remaining } else { available }
      let part = (data[21 + byte_index].to_uint() >> (available - take)) &
        ((1U << take) - 1U)
      packed = (packed << take) | part.to_uint64()
      bit_index = bit_index + take
      remaining = remaining - take
    }
    let decoded = if fill_available == 1 && packed == marker {
      Some(fill)
    } else {
      None
    }
    let offset = element * element_size
    match decoded {
      Some(value) =>
        scaleoffset_write_scalar(output, offset, element_size, order, value)
      None =>
        if is_float {
          if element_size == 4 {
            let value = Float::from_double(packed.to_double()) / factor32 +
              Float::from_double(minimum)
            scaleoffset_write_scalar(
              output,
              offset,
              element_size,
              order,
              value.reinterpret_as_uint().to_uint64(),
            )
          } else {
            let value = packed.to_double() / factor64 + minimum
            scaleoffset_write_scalar(
              output,
              offset,
              element_size,
              order,
              value.reinterpret_as_uint64(),
            )
          }
        } else {
          scaleoffset_write_scalar(
            output,
            offset,
            element_size,
            order,
            (packed + minval) & mask,
          )
        }
    }
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn decode_filters(
  data : Bytes,
  filters : Array[Filter],
  mask : UInt32,
  element_size : Int32,
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  guard filters.length() == 32 || mask >> filters.length() == 0U else {
    raise InvalidField("chunk filter mask exceeds pipeline")
  }
  guard expected_size >= 0 &&
    expected_size <= 0x7FFFFFFF - filters.length() * 32 else {
    raise InvalidField("filtered chunk is too large")
  }
  let limit = expected_size + filters.length() * 32
  let mut output = data
  let mut index = filters.length()
  while index > 0 {
    index = index - 1
    if (mask & (1U << index)) != 0U {
      continue
    }
    let filter = filters[index]
    output = match filter.id {
      1 => {
        guard filter.parameters.length() == 1 && filter.parameters[0] <= 9U else {
          raise InvalidField("invalid deflate parameters")
        }
        guard output.length() >= 2 && output[0].to_int() >> 4 <= 7 else {
          raise InvalidField("invalid zlib window size")
        }
        @zlib.unzlib(output[:], checksum=true, limit~) catch {
          _ => raise InvalidField("invalid or oversized deflate chunk")
        }
      }
      2 => {
        guard filter.parameters.length() == 1 else {
          raise InvalidField("invalid shuffle element size")
        }
        let shuffle_size = if element_size > 0 {
          element_size
        } else {
          let parameter = filter.parameters[0]
          guard parameter > 0U && parameter <= 0x7FFFFFFFU else {
            raise InvalidField("invalid shuffle element size")
          }
          parameter.reinterpret_as_int()
        }
        guard filter.parameters[0] == shuffle_size.reinterpret_as_uint() else {
          raise InvalidField("invalid shuffle element size")
        }
        guard output.length() <= limit else {
          raise InvalidField("oversized shuffle chunk")
        }
        let elements = output.length() / shuffle_size
        let restored = FixedArray::make(output.length(), b'\x00')
        for byte_index in 0..<shuffle_size {
          for element in 0..<elements {
            restored[element * shuffle_size + byte_index] = output[byte_index *
              elements +
              element]
          }
        }
        for tail in (elements * shuffle_size)..<output.length() {
          restored[tail] = output[tail]
        }
        restored.unsafe_reinterpret_as_bytes()
      }
      3 => {
        guard filter.parameters.length() == 0 else {
          raise InvalidField("invalid Fletcher32 parameters")
        }
        strip_fletcher32(output)
      }
      6 => decode_scaleoffset(output, filter, element_size, expected_size)
      _ =>
        raise UnsupportedFeature(
          "filter " + filter.id.to_string() + " is not supported",
        )
    }
  }
  guard output.length() == expected_size else {
    raise InvalidField("decoded chunk size does not match declared shape")
  }
  output
}
