///|
/// Values decoded from an HDF5 dataset.
pub(all) enum DatasetValues {
  Integers(Array[Int64])
  UnsignedIntegers(Array[UInt64])
  Floats(Array[Double])
  Strings(Array[String])
} derive(@debug.Debug)

///|
pub extend DatasetValues with @debug.Debug::{to_repr}

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
  match size {
    1 => bytes.unsafe_get(offset).to_uint64()
    2 =>
      if little_endian {
        bytes.unsafe_read_uint16_le(offset).to_uint64()
      } else {
        bytes.unsafe_read_uint16_be(offset).to_uint64()
      }
    4 =>
      if little_endian {
        bytes.unsafe_read_uint32_le(offset).to_uint64()
      } else {
        bytes.unsafe_read_uint32_be(offset).to_uint64()
      }
    8 =>
      if little_endian {
        bytes.unsafe_read_uint64_le(offset)
      } else {
        bytes.unsafe_read_uint64_be(offset)
      }
    _ => raise UnsupportedFeature("datatype width is not 1, 2, 4 or 8 bytes")
  }
}

///|
fn decode_signed(raw : UInt64, size : Int32) -> Int64 {
  if size == 8 {
    return raw.reinterpret_as_int64()
  }
  let bits = size * 8
  let sign_bit = 1UL << (bits - 1)
  let value = raw.reinterpret_as_int64()
  if (raw & sign_bit) != 0UL {
    value - (1L << bits)
  } else {
    value
  }
}

///|
fn read_float_at(
  bytes : Bytes,
  offset : Int32,
  size : Int32,
  little_endian : Bool,
) -> Double raise Hdf5Error {
  if size == 4 {
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
    raise UnsupportedFeature("floating-point width is not 4 or 8 bytes")
  }
}

///|
fn read_fixed_string_at(
  bytes : Bytes,
  offset : Int32,
  length : Int32,
) -> String raise Hdf5Error {
  guard offset >= 0 &&
    length >= 0 &&
    offset.to_int64() + length.to_int64() <= bytes.length().to_int64() else {
    raise UnexpectedEof(offset.to_int64())
  }
  let mut end = offset + length
  while end > offset && bytes.unsafe_get(end - 1) == b'\x00' {
    end = end - 1
  }
  let view = bytes.get_view(start=offset, end~).unwrap()
  @utf8.decode(view) catch {
    _ => raise InvalidField("invalid UTF-8 string data")
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
) -> DatasetValues raise Hdf5Error {
  guard count >= 0 && count <= 0x7FFFFFFFL else {
    raise InvalidField("dataset has too many elements")
  }
  let n = count.to_int()
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
          values.push(decode_signed(raw, datatype.size))
        }
        Integers(values)
      } else {
        let values : Array[UInt64] = []
        for i in 0..<n {
          values.push(
            read_uint_at(
              payload,
              i * datatype.size,
              datatype.size,
              datatype.little_endian,
            ),
          )
        }
        UnsignedIntegers(values)
      }
    c if c == datatype_class_floating_point() => {
      let values : Array[Double] = []
      for i in 0..<n {
        values.push(
          read_float_at(
            payload,
            i * datatype.size,
            datatype.size,
            datatype.little_endian,
          ),
        )
      }
      Floats(values)
    }
    c if c == datatype_class_string() => {
      let values : Array[String] = []
      for i in 0..<n {
        values.push(
          read_fixed_string_at(payload, i * datatype.size, datatype.size),
        )
      }
      Strings(values)
    }
    9 => {
      guard datatype.vlen_is_string else {
        raise UnsupportedFeature(
          "variable-length sequences are not supported yet",
        )
      }
      let values : Array[String] = []
      for i in 0..<n {
        let element_offset = i * datatype.size
        let length = read_uint_at(payload, element_offset, 4, true)
          .reinterpret_as_int64()
          .to_int()
        if length == 0 {
          values.push("")
        } else {
          let heap_address = read_uint_at(
            payload,
            element_offset + 4,
            offset_size,
            true,
          )
          let object_index = read_uint_at(
              payload,
              element_offset + 4 + offset_size,
              4,
              true,
            )
            .reinterpret_as_int64()
            .to_int()
          let data = read_global_heap_object(
            bytes, heap_address, object_index, offset_size, length_size, base_address,
          )
          guard data.length() >= length else {
            raise InvalidField(
              "global heap string is shorter than declared length",
            )
          }
          let view = data.get_view(start=0, end=length).unwrap()
          let text = @utf8.decode(view) catch {
            _ => raise InvalidField("invalid UTF-8 variable-length string")
          }
          values.push(text)
        }
      }
      Strings(values)
    }
    _ => raise UnsupportedFeature("datatype class is not supported yet")
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
/// Parses a dataset object header and decodes compact, contiguous or
/// unfiltered v1 B-tree chunked data.
pub fn read_dataset(
  bytes : Bytes,
  header : ObjectHeader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Dataset raise Hdf5Error {
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
  if header.first(msg_filter_pipeline) is Some(_) {
    raise UnsupportedFeature("filtered datasets are not supported yet")
  }
  let count = dataspace.element_count()
  let payload = match layout.layout_class {
    0 => layout.compact_data
    1 => {
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
    2 => {
      let fill_value = dataset_fill_value(header, datatype)
      read_chunked_payload(
        bytes,
        layout,
        dataspace,
        datatype,
        offset_size,
        base_address,
        fill_value~,
      )
    }
    _ => raise UnsupportedFeature("unknown data layout class")
  }
  let values = decode_values(
    bytes, payload, count, datatype, offset_size, length_size, base_address,
  )
  { dataspace, datatype, layout, values, }
}
