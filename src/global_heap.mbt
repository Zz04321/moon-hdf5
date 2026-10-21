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
  let sources : Array[&Hdf5Source] = [bytes]
  read_source_global_heap_object(
    sources[0],
    heap_address,
    object_index,
    length_size,
    base_address,
  )
}

///|
fn read_source_global_heap_object(
  source : &Hdf5Source,
  heap_address : UInt64,
  object_index : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Bytes raise Hdf5Error {
  let offset = source_address_to_offset(heap_address, base_address)
  let header_size = 8 + length_size
  let reader = Reader::new(source.read_at(offset, header_size))
  guard reader.read_bytes(4) == b"GCOL" else {
    raise InvalidField("expected GCOL signature")
  }
  let version = reader.read_u8().to_int()
  guard version == 1 else {
    raise UnsupportedFeature("unsupported global heap version")
  }
  reader.skip(3)
  let collection_size = reader.read_uint_le(length_size)
  guard collection_size >= header_size.to_uint64() &&
    offset <= source.length() &&
    collection_size <= source.length() - offset else {
    raise InvalidField("invalid global heap collection size")
  }
  let object_header_size = 8 + length_size
  let mut cursor = header_size.to_uint64()
  while cursor <= collection_size &&
        object_header_size.to_uint64() <= collection_size - cursor {
    let object_header = Reader::new(
      source.read_at(offset + cursor, object_header_size),
    )
    let index = object_header.read_u16_le().to_int()
    let _reference_count = object_header.read_u16_le()
    let _reserved = object_header.read_u32_le()
    let size = object_header.read_uint_le(length_size)
    guard size <= 0x7FFFFFFFUL else {
      raise InvalidField("global heap object is too large")
    }
    cursor = cursor + object_header_size.to_uint64()
    if index == 0 {
      break
    }
    guard size <= collection_size - cursor else {
      raise InvalidField("global heap object extends past its collection")
    }
    let padding = (8 - size.to_int() % 8) % 8
    guard padding.to_uint64() <= collection_size - cursor - size else {
      raise InvalidField("global heap padding extends past its collection")
    }
    if index == object_index {
      return source.read_at(offset + cursor, size.to_int())
    }
    cursor = cursor + size + padding.to_uint64()
  }
  raise InvalidField("global heap object not found")
}
