///|
pub(all) struct DataLayout {
  version : Int32
  layout_class : Int32
  data_address : UInt64
  data_size : UInt64
  chunk_dimensions : Array[UInt32]
  element_size : Int32
  compact_data : Bytes
} derive(@debug.Debug)

///|
pub extend DataLayout with @debug.Debug::{to_repr}

///|
pub fn parse_data_layout(
  message : HeaderMessage,
  offset_size : Int32,
  length_size : Int32,
) -> DataLayout raise Hdf5Error {
  guard message.msg_type == msg_data_layout else {
    raise InvalidField("message is not a data layout message")
  }
  let reader = Reader::new(message.data)
  let version = reader.read_u8().to_int()
  if version == 4 {
    raise UnsupportedFeature(
      "data layout message version 4 is not supported yet",
    )
  }
  let layout_class = if version == 1 || version == 2 {
    let _dimensionality = reader.read_u8().to_int()
    let layout_class = reader.read_u8().to_int()
    reader.skip(5)
    layout_class
  } else if version == 3 {
    reader.read_u8().to_int()
  } else {
    raise InvalidField("unsupported data layout message version")
  }
  let chunk_dimensions : Array[UInt32] = []
  let mut data_address = 0UL
  let mut data_size = 0UL
  let mut element_size = 0
  let mut compact_data : Bytes = b""
  if version == 1 || version == 2 {
    match layout_class {
      0 => {
        let size = reader.read_u32_le().reinterpret_as_int()
        compact_data = reader.read_bytes(size)
      }
      1 | 2 => {
        data_address = reader.read_uint_le(offset_size)
        let dimensionality = message.data.unsafe_get(1).to_int()
        for _ in 1..<dimensionality {
          chunk_dimensions.push(reader.read_u32_le())
        }
        element_size = reader.read_u32_le().reinterpret_as_int()
      }
      _ => raise InvalidField("invalid data layout class")
    }
  } else {
    match layout_class {
      0 => {
        let size = reader.read_u16_le().to_int()
        compact_data = reader.read_bytes(size)
      }
      1 => {
        data_address = reader.read_uint_le(offset_size)
        data_size = reader.read_uint_le(length_size)
      }
      2 => {
        let dimensionality = reader.read_u8().to_int()
        data_address = reader.read_uint_le(offset_size)
        for _ in 1..<dimensionality {
          chunk_dimensions.push(reader.read_u32_le())
        }
        element_size = reader.read_u32_le().reinterpret_as_int()
      }
      _ => raise InvalidField("invalid data layout class")
    }
  }
  {
    version,
    layout_class,
    data_address,
    data_size,
    chunk_dimensions,
    element_size,
    compact_data,
  }
}
