///|
fn product(values : Array[Int64]) -> Int64 raise Hdf5Error {
  let mut result = 1L
  for value in values {
    guard value > 0L && value <= 0x7FFFFFFFL else {
      raise InvalidField("invalid chunk dimension")
    }
    guard result <= 0x7FFFFFFFL / value else {
      raise InvalidField("chunk element count is too large")
    }
    result = result * value
  }
  result
}

///|
fn chunk_byte_size(
  dimensions : Array[UInt32],
  element_size : Int32,
) -> Int32 raise Hdf5Error {
  let count = product(dimensions.map(size => size.to_int64()))
  guard element_size > 0 && count <= 0x7FFFFFFFL / element_size.to_int64() else {
    raise InvalidField("chunk byte size is too large")
  }
  count.to_int() * element_size
}

///|
fn read_filtered_chunk(
  bytes : Bytes,
  address : UInt64,
  base_address : UInt64,
  stored_size : UInt64,
  filters : Array[Filter],
  mask : UInt32,
  element_size : Int32,
  expected_size : Int32,
) -> Bytes raise Hdf5Error {
  guard stored_size > 0UL && stored_size <= 0x7FFFFFFFUL else {
    raise InvalidField("invalid stored chunk size")
  }
  let data = Reader::at(bytes, address_to_offset(address, base_address)).read_bytes(
    stored_size.to_int(),
  )
  decode_filters(data, filters, mask, element_size, expected_size)
}

///|
fn selected_chunk_index(
  index : UInt64,
  offsets : Array[UInt64],
  chunk_dimensions : Array[UInt32],
  dataspace_dimensions : Array[UInt64],
) -> UInt64? raise Hdf5Error {
  let mut remaining = index
  let mut chunk_index = 0UL
  let mut multiplier = 1UL
  let mut dimension = dataspace_dimensions.length()
  while dimension > 0 {
    dimension = dimension - 1
    let dataspace_size = dataspace_dimensions[dimension]
    guard dataspace_size > 0UL else {
      raise InvalidField("selected dataset dimension is empty")
    }
    let coordinate = remaining % dataspace_size
    remaining = remaining / dataspace_size
    if coordinate < offsets[dimension] ||
      coordinate - offsets[dimension] >= chunk_dimensions[dimension].to_uint64() {
      return None
    }
    chunk_index = chunk_index + (coordinate - offsets[dimension]) * multiplier
    multiplier = multiplier * chunk_dimensions[dimension].to_uint64()
  }
  Some(chunk_index)
}

///|
fn chunk_contains_selection(
  offsets : Array[UInt64],
  chunk_dimensions : Array[UInt32],
  dataspace_dimensions : Array[UInt64],
  indices : Array[UInt64]?,
) -> Bool raise Hdf5Error {
  guard indices is Some(selected) else { return true }
  for index in selected {
    if selected_chunk_index(
        index, offsets, chunk_dimensions, dataspace_dimensions,
      )
      is Some(_) {
      return true
    }
  }
  false
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
  selection : Array[UInt64]?,
) -> Unit raise Hdf5Error {
  let ndims = chunk_dimensions.length()
  let chunk_dimensions_i64 = chunk_dimensions.map(value => value.to_int64())
  let chunk_count = product(chunk_dimensions_i64)
  guard element_size > 0 &&
    chunk_size >= 0 &&
    chunk_size <= chunk_data.length() &&
    chunk_count <= chunk_size.to_int64() / element_size.to_int64() else {
    raise InvalidField("chunk is smaller than its declared shape")
  }
  if selection is Some(indices) {
    for selected in 0..<indices.length() {
      if selected_chunk_index(
          indices[selected],
          offsets,
          chunk_dimensions,
          dataspace_dimensions,
        )
        is Some(chunk_index) {
        guard chunk_index <= 0x7FFFFFFFUL &&
          chunk_index.reinterpret_as_int64() < chunk_count else {
          raise InvalidField("selected chunk index is out of range")
        }
        output.blit_from_bytes(
          selected * element_size,
          chunk_data,
          chunk_index.to_int() * element_size,
          element_size,
        )
      }
    }
    return
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
        break
      }
      guard dataspace_dimensions[i] <= 0x7FFFFFFFUL &&
        destination <=
        (0x7FFFFFFFL - target) / dataspace_dimensions[i].reinterpret_as_int64() else {
        raise InvalidField("chunk destination index is out of range")
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
    guard element_size <= output.length() &&
      destination <=
      (output.length() - element_size).to_int64() / element_size.to_int64() else {
      raise InvalidField("chunk destination exceeds output buffer")
    }
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
  filters : Array[Filter],
  selection : Array[UInt64]?,
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
        filters,
        selection,
      )
    } else {
      guard chunk_size >= 0 else { raise InvalidField("negative chunk size") }
      if !chunk_contains_selection(
          offsets, chunk_dimensions, dataspace_dimensions, selection,
        ) {
        continue
      }
      let decoded = read_filtered_chunk(
        bytes,
        child,
        base_address,
        chunk_size.to_uint64(),
        filters,
        filter_mask,
        element_size,
        chunk_byte_size(chunk_dimensions, element_size),
      )
      copy_chunk(
        decoded,
        decoded.length(),
        offsets,
        chunk_dimensions,
        dataspace_dimensions,
        element_size,
        output,
        selection,
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
  length_size~ : Int32 = 8,
) -> Bytes raise Hdf5Error {
  read_chunked_with_filters(
    bytes,
    layout,
    dataspace,
    datatype,
    offset_size,
    base_address,
    fill_value,
    [],
    length_size,
  )
}

///|
fn read_chunked_with_filters(
  bytes : Bytes,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  base_address : UInt64,
  fill_value : Bytes,
  filters : Array[Filter],
  length_size : Int32,
  selection~ : Array[UInt64]? = None,
) -> Bytes raise Hdf5Error {
  guard dataspace.kind == Simple && dataspace.dimensions.length() > 0 else {
    raise InvalidField("chunked datasets require a simple dataspace")
  }
  guard layout.chunk_dimensions.length() == dataspace.dimensions.length() else {
    raise InvalidField("chunk rank does not match the dataspace rank")
  }
  chunk_byte_size(layout.chunk_dimensions, datatype.size) |> ignore
  if layout.version == 4 {
    guard layout.element_size == datatype.size else {
      raise InvalidField("chunk element size does not match datatype")
    }
    match layout.chunk_index {
      Single(_, _) => {
        guard dataspace.maximum_dimensions == dataspace.dimensions &&
          dataspace.dimensions ==
          layout.chunk_dimensions.map(size => size.to_uint64()) else {
          raise InvalidField("single chunk dimensions must match dataspace")
        }
      }
      Implicit => {
        guard filters.length() == 0 else {
          raise InvalidField("implicit chunks cannot have filters")
        }
      }
      BTreeV1 => raise InvalidField("v4 layout cannot use a v1 B-tree index")
      FixedArray(_) | ExtensibleArray(_) | BTreeV2(_) => ()
    }
  }
  let count = dataspace.element_count()
  guard count >= 0 && datatype.size > 0 else {
    raise InvalidField("invalid chunked dataset size")
  }
  if selection is None && count > 0x7FFFFFFFL / datatype.size.to_int64() {
    raise InvalidField("chunked dataset size overflow")
  }
  let output_count = match selection {
    Some(indices) => indices.length().to_int64()
    None => count
  }
  let output = filled_payload(output_count, datatype.size, fill_value)
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
  match layout.chunk_index {
    BTreeV1 =>
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
        filters,
        selection,
      )
    FixedArray(page_bits) =>
      read_fixed_array_chunks(
        bytes, layout, dataspace, datatype, offset_size, length_size, base_address,
        filters, output, undefined, page_bits, selection,
      )
    ExtensibleArray(config) =>
      read_extensible_array_chunks(
        bytes, layout, dataspace, datatype, offset_size, length_size, base_address,
        filters, output, undefined, config, selection,
      )
    BTreeV2(config) =>
      read_btree_v2_chunks(
        bytes, layout, dataspace, datatype, offset_size, length_size, base_address,
        filters, output, undefined, config, selection,
      )
    _ =>
      read_direct_chunks(
        bytes, layout, dataspace, datatype, base_address, filters, output, selection,
      )
  }
  output.unsafe_reinterpret_as_bytes()
}
