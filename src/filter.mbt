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
  guard expected_size >= 0 && expected_size <= 0x7FFFFFFF - filters.length() * 4 else {
    raise InvalidField("filtered chunk is too large")
  }
  let limit = expected_size + filters.length() * 4
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
        guard filter.parameters.length() == 1 &&
          filter.parameters[0] == element_size.reinterpret_as_uint() &&
          element_size > 0 else {
          raise InvalidField("invalid shuffle element size")
        }
        guard output.length() <= limit else {
          raise InvalidField("oversized shuffle chunk")
        }
        let elements = output.length() / element_size
        let restored = FixedArray::make(output.length(), b'\x00')
        for byte_index in 0..<element_size {
          for element in 0..<elements {
            restored[element * element_size + byte_index] = output[byte_index *
              elements +
              element]
          }
        }
        for tail in (elements * element_size)..<output.length() {
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
