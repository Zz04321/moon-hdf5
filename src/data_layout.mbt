///|
priv enum ChunkIndex {
  BTreeV1
  Single(UInt64?, UInt32)
  Implicit
  FixedArray(Int32)
  ExtensibleArray(ExtensibleArrayConfig)
  BTreeV2(BTreeV2Config)
} derive(@debug.Debug)

///|
/// The storage strategy used for a dataset payload.
pub(all) enum DatasetLayoutKind {
  Compact
  Contiguous
  Chunked
  Virtual
  Unknown
} derive(Eq, @debug.Debug)

///|
pub extend DatasetLayoutKind with Eq::{not_equal, equal}

///|
pub extend DatasetLayoutKind with @debug.Debug::{to_repr}

///|
priv struct BTreeV2Config {
  node_size : Int32
  split_percent : Int32
  merge_percent : Int32
} derive(@debug.Debug)

///|
priv struct ExtensibleArrayConfig {
  max_bits : Int32
  index_elements : Int32
  min_pointers : Int32
  min_elements : Int32
  page_bits : Int32
} derive(Eq, @debug.Debug)

///|
pub struct DataLayout {
  priv version : Int32
  priv layout_class : Int32
  priv data_address : UInt64
  priv data_size : UInt64
  priv chunk_dimensions : Array[UInt32]
  priv element_size : Int32
  priv compact_data : Bytes
  priv chunk_index : ChunkIndex
} derive(@debug.Debug)

///|
pub extend DataLayout with @debug.Debug::{to_repr}

///|
/// Returns the dataset's storage strategy.
pub fn DataLayout::kind(self : DataLayout) -> DatasetLayoutKind {
  match self.layout_class {
    0 => Compact
    1 => Contiguous
    2 => Chunked
    3 => Virtual
    _ => Unknown
  }
}

///|
/// Returns the chunk shape, or an empty array for non-chunked datasets.
pub fn DataLayout::chunk_shape(self : DataLayout) -> Array[UInt32] {
  self.chunk_dimensions.copy()
}

///|
fn parse_data_layout(
  message : HeaderMessage,
  offset_size : Int32,
  length_size : Int32,
) -> DataLayout raise Hdf5Error {
  guard message.msg_type == msg_data_layout else {
    raise InvalidField("message is not a data layout message")
  }
  let reader = Reader::new(message.data)
  let version = reader.read_u8().to_int()
  let layout_class = if version == 1 || version == 2 {
    let _dimensionality = reader.read_u8().to_int()
    let layout_class = reader.read_u8().to_int()
    reader.skip(5)
    layout_class
  } else if version == 3 || version == 4 {
    reader.read_u8().to_int()
  } else {
    raise InvalidField("unsupported data layout message version")
  }
  let chunk_dimensions : Array[UInt32] = []
  let mut data_address = 0UL
  let mut data_size = 0UL
  let mut element_size = 0
  let mut compact_data : Bytes = b""
  let mut chunk_index = BTreeV1
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
      2 =>
        if version == 4 {
          let flags = reader.read_u8().to_int()
          guard (flags & 0xFC) == 0 else {
            raise InvalidField("invalid v4 chunk flags")
          }
          let dimensionality = reader.read_u8().to_int()
          guard dimensionality >= 2 else {
            raise InvalidField("invalid chunk dimensionality")
          }
          let width = reader.read_u8().to_int()
          guard width >= 1 && width <= 8 else {
            raise InvalidField("invalid chunk dimension width")
          }
          for dimension in 0..<dimensionality {
            let value = reader.read_var_uint_le(width)
            guard value > 0UL && value <= 0x7FFFFFFFUL else {
              raise InvalidField("invalid chunk dimension")
            }
            if dimension == dimensionality - 1 {
              element_size = value.to_int()
            } else {
              chunk_dimensions.push(value.to_uint())
            }
          }
          let index = reader.read_u8().to_int()
          chunk_index = match index {
            1 =>
              if (flags & 2) != 0 {
                let size = reader.read_uint_le(length_size)
                let mask = reader.read_u32_le()
                Single(Some(size), mask)
              } else {
                Single(None, 0U)
              }
            2 => {
              guard flags == 0 else {
                raise InvalidField("invalid implicit index flags")
              }
              Implicit
            }
            3 => {
              guard (flags & 2) == 0 else {
                raise InvalidField("invalid fixed array flags")
              }
              let page_bits = reader.read_u8().to_int()
              guard page_bits > 0 && page_bits <= 31 else {
                raise InvalidField("invalid fixed array page bits")
              }
              FixedArray(page_bits)
            }
            4 => {
              guard (flags & 2) == 0 else {
                raise InvalidField("invalid extensible array flags")
              }
              let config : ExtensibleArrayConfig = {
                max_bits: reader.read_u8().to_int(),
                index_elements: reader.read_u8().to_int(),
                min_pointers: reader.read_u8().to_int(),
                min_elements: reader.read_u8().to_int(),
                page_bits: reader.read_u8().to_int(),
              }
              validate_extensible_config(config)
              ExtensibleArray(config)
            }
            5 => {
              guard (flags & 2) == 0 else {
                raise InvalidField("invalid v2 B-tree flags")
              }
              let size = reader.read_u32_le()
              guard size <= 0x7FFFFFFFU else {
                raise InvalidField("v2 B-tree node size is too large")
              }
              let config : BTreeV2Config = {
                node_size: size.reinterpret_as_int(),
                split_percent: reader.read_u8().to_int(),
                merge_percent: reader.read_u8().to_int(),
              }
              validate_btree_v2_config(config)
              BTreeV2(config)
            }
            _ =>
              raise UnsupportedFeature(
                "chunk index type " +
                index.to_string() +
                " is not supported yet",
              )
          }
          data_address = reader.read_uint_le(offset_size)
        } else {
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
    chunk_index,
  }
}
