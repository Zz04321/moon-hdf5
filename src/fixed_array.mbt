///|
fn read_fixed_array_chunks(
  bytes : Bytes,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  filters : Array[Filter],
  output : FixedArray[Byte],
  undefined : UInt64,
  page_bits : Int32,
  selection : Array[UInt64]?,
) -> Unit raise Hdf5Error {
  guard page_bits > 0 && page_bits <= 31 else {
    raise InvalidField("invalid fixed array page bits")
  }
  let rank = dataspace.dimensions.length()
  guard dataspace.maximum_dimensions.length() == rank else {
    raise InvalidField("fixed array requires maximum dimensions")
  }
  let current_grid : Array[Int64] = []
  let maximum_grid : Array[Int64] = []
  for dimension in 0..<rank {
    let current = dataspace.dimensions[dimension]
    let maximum = dataspace.maximum_dimensions[dimension]
    let width = layout.chunk_dimensions[dimension].to_int64()
    guard current <= maximum && maximum <= 0x7FFFFFFFUL && width > 0 else {
      raise InvalidField("invalid fixed array dimensions")
    }
    current_grid.push((current.reinterpret_as_int64() + width - 1) / width)
    maximum_grid.push((maximum.reinterpret_as_int64() + width - 1) / width)
  }
  let maximum_total = product(maximum_grid)
  let chunk_size = chunk_byte_size(layout.chunk_dimensions, datatype.size)
  let header_offset = address_to_offset(layout.data_address, base_address)
  let header = Reader::at(bytes, header_offset)
  guard header.read_bytes(4) == b"FAHD" && header.read_u8() == b'\x00' else {
    raise InvalidField("invalid fixed array header")
  }
  let client = header.read_u8().to_int()
  let entry_size = header.read_u8().to_int()
  let size_width = entry_size - offset_size - 4
  guard header.read_u8().to_int() == page_bits &&
    (
      (client == 0 && filters.length() == 0 && entry_size == offset_size) ||
      (
        client == 1 &&
        filters.length() > 0 &&
        size_width >= 1 &&
        size_width <= 8
      )
    ) else {
    raise InvalidField("invalid fixed array entry encoding")
  }
  guard header.read_uint_le(length_size) ==
    maximum_total.reinterpret_as_uint64() else {
    raise InvalidField(
      "fixed array entry count does not match maximum dimensions",
    )
  }
  let block_address = header.read_uint_le(offset_size)
  verify_metadata_checksum(
    bytes,
    header_offset,
    header.position(),
    "fixed array header",
  )
  header.skip(4)
  if block_address == undefined {
    return
  }
  let block_offset = address_to_offset(block_address, base_address)
  let block = Reader::at(bytes, block_offset)
  guard block.read_bytes(4) == b"FADB" &&
    block.read_u8() == b'\x00' &&
    block.read_u8().to_int() == client &&
    block.read_uint_le(offset_size) == layout.data_address else {
    raise InvalidField("invalid fixed array data block")
  }
  let page_entries = 1L << page_bits
  let paged = maximum_total > page_entries
  let page_count = (maximum_total + page_entries - 1) / page_entries
  let bitmap = if paged {
    let bitmap = block.read_bytes(((page_count + 7) / 8).to_int())
    verify_metadata_checksum(
      bytes,
      block_offset,
      block.position(),
      "fixed array data block",
    )
    block.skip(4)
    bitmap
  } else {
    b""
  }
  let entries_start = block.position().to_int64()
  if !paged {
    verify_metadata_checksum(
      bytes,
      block_offset,
      (entries_start + maximum_total * entry_size.to_int64()).to_int(),
      "fixed array data block",
    )
  }
  let storage_size = maximum_total * entry_size.to_int64() +
    (if paged { page_count } else { 1L }) * 4
  guard storage_size <= bytes.length().to_int64() - entries_start else {
    raise InvalidField("fixed array entries extend beyond input")
  }
  if output.length() == 0 {
    return
  }
  let total = product(current_grid)
  for linear in 0..<total.to_int() {
    let coordinates : Array[Int64] = Array::make(rank, 0L)
    let offsets : Array[UInt64] = Array::make(rank, 0UL)
    let mut remaining = linear.to_int64()
    let mut dimension = rank
    while dimension > 0 {
      dimension = dimension - 1
      coordinates[dimension] = remaining % current_grid[dimension]
      remaining = remaining / current_grid[dimension]
      offsets[dimension] = (coordinates[dimension] *
      layout.chunk_dimensions[dimension].to_int64()).reinterpret_as_uint64()
    }
    let mut stored_index = 0L
    for dimension in 0..<rank {
      stored_index = stored_index * maximum_grid[dimension] +
        coordinates[dimension]
    }
    let page = stored_index / page_entries
    if paged &&
      (bitmap[page.to_int() / 8].to_int() & (0x80 >> (page.to_int() % 8))) == 0 {
      continue
    }
    if paged {
      let page_elements = if page + 1 == page_count {
        maximum_total - page * page_entries
      } else {
        page_entries
      }
      let page_offset = entries_start +
        page * page_entries * entry_size.to_int64() +
        page * 4L
      verify_metadata_checksum(
        bytes,
        page_offset.to_int(),
        (page_offset + page_elements * entry_size.to_int64()).to_int(),
        "fixed array data block page",
      )
    }
    let position = entries_start +
      stored_index * entry_size.to_int64() +
      (if paged { page * 4 } else { 0L })
    let entry = Reader::at(bytes, position.to_int())
    let address = entry.read_uint_le(offset_size)
    if address == undefined {
      continue
    }
    let mut stored_size = chunk_size.to_uint64()
    let mut mask = 0U
    if client == 1 {
      stored_size = entry.read_var_uint_le(size_width)
      mask = entry.read_u32_le()
    }
    if !chunk_contains_selection(
        offsets,
        layout.chunk_dimensions,
        dataspace.dimensions,
        selection,
      ) {
      continue
    }
    let decoded = read_filtered_chunk(
      bytes,
      address,
      base_address,
      stored_size,
      filters,
      mask,
      datatype.size,
      chunk_size,
    )
    copy_chunk(
      decoded,
      decoded.length(),
      offsets,
      layout.chunk_dimensions,
      dataspace.dimensions,
      datatype.size,
      output,
      selection,
    )
  }
}

///|
fn read_source_fixed_array_chunks(
  source : &Hdf5Source,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  filters : Array[Filter],
  output : FixedArray[Byte],
  undefined : UInt64,
  page_bits : Int32,
  selection : Array[UInt64]?,
) -> Unit raise Hdf5Error {
  guard page_bits > 0 && page_bits <= 31 else {
    raise InvalidField("invalid fixed array page bits")
  }
  let rank = dataspace.dimensions.length()
  guard dataspace.maximum_dimensions.length() == rank else {
    raise InvalidField("fixed array requires maximum dimensions")
  }
  let current_grid : Array[Int64] = []
  let maximum_grid : Array[Int64] = []
  for dimension in 0..<rank {
    let current = dataspace.dimensions[dimension]
    let maximum = dataspace.maximum_dimensions[dimension]
    let width = layout.chunk_dimensions[dimension].to_int64()
    guard current <= maximum && maximum <= 0x7FFFFFFFUL && width > 0 else {
      raise InvalidField("invalid fixed array dimensions")
    }
    current_grid.push((current.reinterpret_as_int64() + width - 1) / width)
    maximum_grid.push((maximum.reinterpret_as_int64() + width - 1) / width)
  }
  let maximum_total = product(maximum_grid)
  let chunk_size = chunk_byte_size(layout.chunk_dimensions, datatype.size)
  let header_offset = address_to_offset(layout.data_address, base_address)
  let header_content = 8 + length_size + offset_size
  let header = Reader::new(
    source_read_checksummed(
      source,
      header_offset.to_uint64(),
      header_content,
      "fixed array header",
    ),
  )
  guard header.read_bytes(4) == b"FAHD" && header.read_u8() == b'\x00' else {
    raise InvalidField("invalid fixed array header")
  }
  let client = header.read_u8().to_int()
  let entry_size = header.read_u8().to_int()
  let size_width = entry_size - offset_size - 4
  guard header.read_u8().to_int() == page_bits &&
    (
      (client == 0 && filters.length() == 0 && entry_size == offset_size) ||
      (
        client == 1 &&
        filters.length() > 0 &&
        size_width >= 1 &&
        size_width <= 8
      )
    ) else {
    raise InvalidField("invalid fixed array entry encoding")
  }
  guard header.read_uint_le(length_size) ==
    maximum_total.reinterpret_as_uint64() else {
    raise InvalidField(
      "fixed array entry count does not match maximum dimensions",
    )
  }
  let block_address = header.read_uint_le(offset_size)
  if block_address == undefined || output.length() == 0 {
    return
  }
  let block_offset = address_to_offset(block_address, base_address)
  let block_prefix_size = 6 + offset_size
  let prefix = Reader::new(
    source.read_at(block_offset.to_uint64(), block_prefix_size),
  )
  guard prefix.read_bytes(4) == b"FADB" &&
    prefix.read_u8() == b'\x00' &&
    prefix.read_u8().to_int() == client &&
    prefix.read_uint_le(offset_size) == layout.data_address else {
    raise InvalidField("invalid fixed array data block")
  }
  let page_entries = 1L << page_bits
  let paged = maximum_total > page_entries
  let page_count = (maximum_total + page_entries - 1) / page_entries
  let bitmap_size = if paged { ((page_count + 7) / 8).to_int() } else { 0 }
  let block_header = if paged {
    source_read_checksummed(
      source,
      block_offset.to_uint64(),
      block_prefix_size + bitmap_size,
      "fixed array data block",
    )
  } else {
    b""
  }
  let bitmap = if paged {
    block_header
    .get_view(start=block_prefix_size, end=block_prefix_size + bitmap_size)
    .unwrap()
    .to_owned()
  } else {
    b""
  }
  let entries_start = block_offset.to_uint64() +
    block_prefix_size.to_uint64() +
    (if paged { bitmap_size + 4 } else { 0 }).to_uint64()
  let unpaged = if paged {
    b""
  } else {
    guard maximum_total <= 0x7FFFFFFFL / entry_size.to_int64() else {
      raise InvalidField("fixed array entries extend beyond input")
    }
    source_read_checksummed(
      source,
      block_offset.to_uint64(),
      block_prefix_size + maximum_total.to_int() * entry_size,
      "fixed array data block",
    )
  }
  let total = product(current_grid)
  let mut cached_page = -1L
  let mut cached_entries = b""
  for linear in 0..<total.to_int() {
    let coordinates : Array[Int64] = Array::make(rank, 0L)
    let offsets : Array[UInt64] = Array::make(rank, 0UL)
    let mut remaining = linear.to_int64()
    let mut dimension = rank
    while dimension > 0 {
      dimension = dimension - 1
      coordinates[dimension] = remaining % current_grid[dimension]
      remaining = remaining / current_grid[dimension]
      offsets[dimension] = (coordinates[dimension] *
      layout.chunk_dimensions[dimension].to_int64()).reinterpret_as_uint64()
    }
    let mut stored_index = 0L
    for dimension in 0..<rank {
      stored_index = stored_index * maximum_grid[dimension] +
        coordinates[dimension]
    }
    let page = stored_index / page_entries
    if paged &&
      (bitmap[page.to_int() / 8].to_int() & (0x80 >> (page.to_int() % 8))) == 0 {
      continue
    }
    if !chunk_contains_selection(
        offsets,
        layout.chunk_dimensions,
        dataspace.dimensions,
        selection,
      ) {
      continue
    }
    let entry = if paged {
      if cached_page != page {
        let page_elements = if page + 1 == page_count {
          maximum_total - page * page_entries
        } else {
          page_entries
        }
        guard page_elements <= 0x7FFFFFFFL / entry_size.to_int64() else {
          raise InvalidField("fixed array page is too large")
        }
        let page_offset = entries_start +
          page.reinterpret_as_uint64() *
          (page_entries * entry_size.to_int64() + 4L).reinterpret_as_uint64()
        cached_entries = source_read_checksummed(
          source,
          page_offset,
          page_elements.to_int() * entry_size,
          "fixed array data block page",
        )
        cached_page = page
      }
      Reader::at(
        cached_entries,
        ((stored_index - page * page_entries) * entry_size.to_int64()).to_int(),
      )
    } else {
      Reader::at(
        unpaged,
        block_prefix_size + stored_index.to_int() * entry_size,
      )
    }
    let address = entry.read_uint_le(offset_size)
    if address == undefined {
      continue
    }
    let stored_size = if client == 1 {
      entry.read_var_uint_le(size_width)
    } else {
      chunk_size.to_uint64()
    }
    let mask = if client == 1 { entry.read_u32_le() } else { 0U }
    let decoded = read_source_filtered_chunk(
      source,
      address,
      base_address,
      stored_size,
      filters,
      mask,
      datatype.size,
      chunk_size,
    )
    copy_chunk(
      decoded,
      decoded.length(),
      offsets,
      layout.chunk_dimensions,
      dataspace.dimensions,
      datatype.size,
      output,
      selection,
    )
  }
}
