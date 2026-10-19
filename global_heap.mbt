///|
fn round_up_to_eight(n : Int32) -> Int32 {
  (n + 7) / 8 * 8
}

///|
/// Reads one object from an HDF5 global heap collection.
pub fn read_global_heap_object(
  bytes : Bytes,
  heap_address : UInt64,
  object_index : Int32,
  _offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Bytes raise Hdf5Error {
  let offset = address_to_offset(heap_address, base_address)
  let reader = Reader::at(bytes, offset)
  guard reader.read_bytes(4) == b"GCOL" else {
    raise InvalidField("expected GCOL signature")
  }
  let version = reader.read_u8().to_int()
  guard version == 1 else {
    raise UnsupportedFeature("unsupported global heap version")
  }
  reader.skip(3)
  let collection_size = reader.read_uint_le(length_size)
  guard collection_size >= 16UL && collection_size <= 0x7FFFFFFFUL else {
    raise InvalidField("invalid global heap collection size")
  }
  let end = offset.to_int64() + collection_size.reinterpret_as_int64()
  while reader.position().to_int64() + 16 <= end {
    let index = reader.read_u16_le().to_int()
    let _reference_count = reader.read_u16_le()
    let _reserved = reader.read_u32_le()
    let size = reader.read_uint_le(length_size)
    guard size <= 0x7FFFFFFFUL else {
      raise InvalidField("global heap object is too large")
    }
    if index == 0 {
      break
    }
    let object_end = reader.position().to_int64() + size.reinterpret_as_int64()
    guard object_end <= end else {
      raise InvalidField("global heap object extends past its collection")
    }
    let data = reader.read_bytes(size.to_int())
    let padding = round_up_to_eight(size.to_int()) - size.to_int()
    if index == object_index {
      return data
    }
    guard reader.position().to_int64() + padding.to_int64() <= end else {
      raise InvalidField("global heap padding extends past its collection")
    }
    reader.skip(padding)
  }
  raise InvalidField("global heap object not found")
}
