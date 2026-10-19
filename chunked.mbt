///|
fn product(values : Array[Int64]) -> Int64 raise Hdf5Error {
  let mut result = 1L
  for value in values {
    guard value > 0L && value <= 0x7FFFFFFFL else {
      raise InvalidField("invalid chunk dimension")
    }
    result = result * value
  }
  result
}

///|
fn copy_chunk(
  chunk_data : Bytes,
  chunk_size : Int32,
  offsets : Array[UInt64],
  chunk_dimensions : Array[UInt32],
  dataspace_dimensions : Array[UInt64],
  element_size : Int32,
  output : FixedArray[Byte],
) -> Unit raise Hdf5Error {
  let ndims = chunk_dimensions.length()
  let chunk_dimensions_i64 = chunk_dimensions.map(value => value.to_int64())
  let chunk_count = product(chunk_dimensions_i64)
  guard chunk_count * element_size.to_int64() <= chunk_size.to_int64() else {
    raise InvalidField("chunk is smaller than its declared shape")
  }
  let coordinates : Array[Int64] = Array::make(ndims, 0L)
  for linear in 0..<chunk_count.to_int() {
    let mut remaining = linear.to_int64()
    let mut dimension = ndims
    while dimension > 0 {
      dimension = dimension - 1
      let size = chunk_dimensions_i64[dimension]
      coordinates[dimension] = remaining % size
      remaining = remaining / size
    }
    let mut destination = 0L
    let mut outside = false
    for i in 0..<ndims {
      guard offsets[i] <= dataspace_dimensions[i] else {
        raise InvalidField("chunk offset is outside the dataspace")
      }
      let target = offsets[i].reinterpret_as_int64() + coordinates[i]
      if target >= dataspace_dimensions[i].reinterpret_as_int64() {
        outside = true
      }
      destination = destination * dataspace_dimensions[i].reinterpret_as_int64() +
        target
    }
    if outside {
      continue
    }
    guard destination >= 0L && destination <= 0x7FFFFFFFL else {
      raise InvalidField("chunk destination index is out of range")
    }
    let source_start = linear * element_size
    let destination_start = destination.to_int() * element_size
    output.blit_from_bytes(
      destination_start, chunk_data, source_start, element_size,
    )
  }
}

///|
fn read_chunk_btree(
  bytes : Bytes,
  node_address : UInt64,
  expected_level : Int32,
  chunk_dimensions : Array[UInt32],
  dataspace_dimensions : Array[UInt64],
  element_size : Int32,
  offset_size : Int32,
  base_address : UInt64,
  output : FixedArray[Byte],
) -> Unit raise Hdf5Error {
  let offset = address_to_offset(node_address, base_address)
  let reader = Reader::at(bytes, offset)
  guard reader.read_bytes(4) == b"TREE" else {
    raise InvalidField("expected chunk B-tree node")
  }
  guard reader.read_u8() == b'\x01' else {
    raise InvalidField("B-tree node is not a raw-data chunk node")
  }
  let node_level = reader.read_u8().to_int()
  if expected_level >= 0 {
    guard node_level == expected_level else {
      raise InvalidField("unexpected chunk B-tree level")
    }
  }
  let entries_used = reader.read_u16_le().to_int()
  let _left_sibling = reader.read_uint_le(offset_size)
  let _right_sibling = reader.read_uint_le(offset_size)
  let ndims = chunk_dimensions.length()
  let key_dimensions = ndims + 1
  for _ in 0..<entries_used {
    let chunk_size = reader.read_u32_le().reinterpret_as_int()
    let filter_mask = reader.read_u32_le()
    let offsets : Array[UInt64] = []
    for dimension in 0..<key_dimensions {
      let offset = reader.read_u64_le()
      if dimension < ndims {
        offsets.push(offset)
      }
    }
    let child = reader.read_uint_le(offset_size)
    guard filter_mask == 0U else {
      raise UnsupportedFeature("chunk filters are not supported yet")
    }
    if node_level > 0 {
      read_chunk_btree(
        bytes,
        child,
        node_level - 1,
        chunk_dimensions,
        dataspace_dimensions,
        element_size,
        offset_size,
        base_address,
        output,
      )
    } else {
      guard chunk_size >= 0 else { raise InvalidField("negative chunk size") }
      let chunk_data = Reader::at(bytes, address_to_offset(child, base_address)).read_bytes(
        chunk_size,
      )
      copy_chunk(
        chunk_data, chunk_size, offsets, chunk_dimensions, dataspace_dimensions,
        element_size, output,
      )
    }
  }
  let _final_size = reader.read_u32_le()
  let _final_filter_mask = reader.read_u32_le()
  for _ in 0..<key_dimensions {
    let _final_offset = reader.read_u64_le()
  }
}

///|
/// Reconstructs an unfiltered chunked dataset into a contiguous buffer.
pub fn read_chunked_payload(
  bytes : Bytes,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  base_address : UInt64,
  fill_value~ : Bytes = b"",
) -> Bytes raise Hdf5Error {
  guard dataspace.kind == Simple && dataspace.dimensions.length() > 0 else {
    raise InvalidField("chunked datasets require a simple dataspace")
  }
  guard layout.chunk_dimensions.length() == dataspace.dimensions.length() else {
    raise InvalidField("chunk rank does not match the dataspace rank")
  }
  let count = dataspace.element_count()
  guard count >= 0 && count <= 0x7FFFFFFFL else {
    raise InvalidField("chunked dataset is too large")
  }
  guard datatype.size > 0 && count <= 0x7FFFFFFFL / datatype.size.to_int64() else {
    raise InvalidField("chunked dataset size overflow")
  }
  let total_size = count.to_int() * datatype.size
  guard fill_value.length() == 0 || fill_value.length() == datatype.size else {
    raise InvalidField("fill value size does not match datatype")
  }
  let output = FixedArray::make(total_size, b'\x00')
  if fill_value.length() != 0 {
    for element in 0..<count.to_int() {
      output.blit_from_bytes(
        element * datatype.size,
        fill_value,
        0,
        datatype.size,
      )
    }
  }
  guard offset_size == 1 ||
    offset_size == 2 ||
    offset_size == 4 ||
    offset_size == 8 else {
    raise UnsupportedAddressSize(offset_size)
  }
  let undefined = if offset_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (offset_size * 8)) - 1UL
  }
  if layout.data_address == undefined {
    return output.unsafe_reinterpret_as_bytes()
  }
  read_chunk_btree(
    bytes,
    layout.data_address,
    -1,
    layout.chunk_dimensions,
    dataspace.dimensions,
    datatype.size,
    offset_size,
    base_address,
    output,
  )
  output.unsafe_reinterpret_as_bytes()
}
