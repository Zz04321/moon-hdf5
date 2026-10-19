///|
/// One direct block in a fractal heap's linear address space.
priv struct FractalHeapBlock {
  heap_offset : UInt64
  address : UInt64
  size : Int32
  data : Bytes
}

///|
/// One record in the v2 B-tree that indexes indirectly accessed huge objects.
priv struct HugeObjectRecord {
  address : UInt64
  stored_size : UInt64
  filter_mask : UInt32
  object_size : UInt64
  id : UInt64
}

///|
/// A parsed fractal heap used by dense groups and dense attributes.
priv struct FractalHeap {
  bytes : Bytes
  header_address : UInt64
  offset_size : Int32
  length_size : Int32
  base_address : UInt64
  id_length : Int32
  flags : Int32
  table_width : Int32
  start_block_size : UInt64
  managed_offset_size : Int32
  managed_length_size : Int32
  block_offset_size : Int32
  max_direct_rows : Int32
  indirect_rows_base : Int32
  managed_space_size : UInt64
  huge_btree_address : UInt64
  filter_len : Int32
  filters : Array[Filter]
  huge_id_size : Int32
  huge_ids_direct : Bool
  blocks : Array[FractalHeapBlock]
}

///|
fn heap_undefined_address(offset_size : Int32) -> UInt64 {
  if offset_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (offset_size * 8)) - 1UL
  }
}

///|
fn heap_log2(value : UInt64, field : String) -> Int32 raise Hdf5Error {
  guard value > 0UL else { raise InvalidField(field + " is zero") }
  let mut remaining = value
  let mut bits = 0
  while remaining > 1UL {
    guard (remaining & 1UL) == 0UL else {
      raise InvalidField(field + " is not a power of two")
    }
    remaining = remaining >> 1
    bits = bits + 1
  }
  bits
}

///|
fn heap_byte_width(value : UInt64) -> Int32 {
  let mut bits = 0
  let mut remaining = value
  while remaining > 0UL {
    bits = bits + 1
    remaining = remaining >> 1
  }
  if bits == 0 {
    1
  } else {
    (bits + 7) / 8
  }
}

///|
fn FractalHeap::read_file_object(
  self : FractalHeap,
  address : UInt64,
  size : UInt64,
  label : String,
) -> Bytes raise Hdf5Error {
  guard size > 0UL && size <= 0x7FFFFFFFUL else {
    raise InvalidField("invalid " + label + " size")
  }
  let start = address_to_offset(address, self.base_address)
  guard start.to_int64() + size.reinterpret_as_int64() <=
    self.bytes.length().to_int64() else {
    raise UnexpectedEof(start.to_int64())
  }
  self.bytes.get_view(start~, end=start + size.to_int()).unwrap().to_owned()
}

///|
fn FractalHeap::decode_stored_object(
  self : FractalHeap,
  data : Bytes,
  mask : UInt32,
  object_size : UInt64,
) -> Bytes raise Hdf5Error {
  guard object_size > 0UL && object_size <= 0x7FFFFFFFUL else {
    raise InvalidField("invalid fractal heap huge object size")
  }
  decode_filters(data, self.filters, mask, 0, object_size.to_int())
}

///|
fn parse_huge_btree_record(
  data : Bytes,
  node_type : Int32,
  offset_size : Int32,
  length_size : Int32,
) -> HugeObjectRecord raise Hdf5Error {
  let reader = Reader::new(data)
  let address = reader.read_uint_le(offset_size)
  let stored_size = reader.read_uint_le(length_size)
  let filter_mask = match node_type {
    2 | 4 => reader.read_u32_le()
    _ => 0U
  }
  let object_size = match node_type {
    2 | 4 => reader.read_uint_le(length_size)
    _ => stored_size
  }
  let id = match node_type {
    1 | 2 => reader.read_uint_le(length_size)
    _ => 0UL
  }
  guard reader.remaining() == 0 else {
    raise InvalidField("fractal heap huge object record has trailing bytes")
  }
  guard stored_size > 0UL &&
    stored_size <= 0x7FFFFFFFUL &&
    object_size > 0UL &&
    object_size <= 0x7FFFFFFFUL else {
    raise InvalidField("invalid fractal heap huge object record size")
  }
  { address, stored_size, filter_mask, object_size, id, }
}

///|
fn FractalHeap::huge_record(
  self : FractalHeap,
  id : UInt64,
) -> HugeObjectRecord raise Hdf5Error {
  guard self.huge_btree_address != heap_undefined_address(self.offset_size) else {
    raise InvalidField("fractal heap has no huge object B-tree")
  }
  let expected_type = if self.filter_len > 0 { 2 } else { 1 }
  let contents = read_v2_btree_records(
    self.bytes,
    self.huge_btree_address,
    self.offset_size,
    self.length_size,
    self.base_address,
    [expected_type],
    "fractal heap huge object v2 B-tree",
  )
  for raw in contents.records {
    let record = parse_huge_btree_record(
      raw,
      contents.node_type,
      self.offset_size,
      self.length_size,
    )
    if record.id == id {
      return record
    }
  }
  raise InvalidField("fractal heap huge object ID was not found")
}

///|
fn FractalHeap::block_size_at(
  self : FractalHeap,
  index : Int32,
) -> UInt64 raise Hdf5Error {
  guard index >= 0 && self.table_width > 0 else {
    raise InvalidField("invalid fractal heap block index")
  }
  let row = index / self.table_width
  let shift = if row > 1 { row - 1 } else { 0 }
  guard shift <= 62 else {
    raise InvalidField("fractal heap block size is too large")
  }
  let size = self.start_block_size << shift
  guard size >= self.start_block_size && size <= 0x7FFFFFFFUL else {
    raise InvalidField("fractal heap block size is too large")
  }
  size
}

///|
fn FractalHeap::add_direct_block(
  self : FractalHeap,
  address : UInt64,
  block_size : UInt64,
  stored_size : UInt64,
  filter_mask : UInt32,
  depth : Int32,
) -> Unit raise Hdf5Error {
  guard depth < 64 && block_size > 0UL && block_size <= 0x7FFFFFFFUL else {
    raise InvalidField("invalid fractal heap direct block size")
  }
  let undefined = heap_undefined_address(self.offset_size)
  guard address != undefined else { return }
  for block in self.blocks {
    if block.address == address {
      return
    }
  }
  let offset = address_to_offset(address, self.base_address)
  let raw_size = if self.filter_len > 0 { stored_size } else { block_size }
  guard raw_size > 0UL &&
    raw_size <= 0x7FFFFFFFUL &&
    offset.to_int64() + raw_size.reinterpret_as_int64() <=
    self.bytes.length().to_int64() else {
    raise InvalidField("invalid fractal heap direct block size")
  }
  let stored = self.bytes
    .get_view(start=offset, end=offset + raw_size.to_int())
    .unwrap()
    .to_owned()
  let data = if self.filter_len > 0 {
    decode_filters(stored, self.filters, filter_mask, 0, block_size.to_int())
  } else {
    stored
  }
  let reader = Reader::new(data)
  guard reader.read_bytes(4) == b"FHDB" && reader.read_u8() == b'\x00' else {
    raise InvalidField("invalid fractal heap direct block")
  }
  guard reader.read_uint_le(self.offset_size) == self.header_address else {
    raise InvalidField("fractal heap direct block has the wrong header address")
  }
  let heap_offset = reader.read_var_uint_le(self.block_offset_size)
  if (self.flags & 2) != 0 {
    verify_metadata_checksum_zeroed(
      data,
      0,
      block_size.to_int(),
      reader.position(),
      "fractal heap direct block",
    )
    reader.skip(4)
  }
  guard heap_offset <= self.managed_space_size &&
    block_size <= self.managed_space_size - heap_offset else {
    raise InvalidField("fractal heap direct block is outside managed space")
  }
  guard reader.position() <= reader.limit else {
    raise InvalidField("fractal heap direct block header is truncated")
  }
  self.blocks.push({ heap_offset, address, size: block_size.to_int(), data, })
}

///|
fn FractalHeap::add_indirect_block(
  self : FractalHeap,
  address : UInt64,
  nrows : Int32,
  depth : Int32,
  visited : Array[UInt64],
) -> Unit raise Hdf5Error {
  guard depth < 64 && nrows > 0 && self.table_width > 0 else {
    raise InvalidField("invalid fractal heap indirect block")
  }
  let undefined = heap_undefined_address(self.offset_size)
  guard address != undefined else { return }
  guard !visited.contains(address) else {
    raise InvalidField("fractal heap contains an indirect block cycle")
  }
  visited.push(address)
  let offset = address_to_offset(address, self.base_address)
  let reader = Reader::at(self.bytes, offset)
  guard reader.read_bytes(4) == b"FHIB" && reader.read_u8() == b'\x00' else {
    raise InvalidField("invalid fractal heap indirect block")
  }
  guard reader.read_uint_le(self.offset_size) == self.header_address else {
    raise InvalidField(
      "fractal heap indirect block has the wrong header address",
    )
  }
  let _block_offset = reader.read_var_uint_le(self.block_offset_size)
  guard nrows <= 0x10000 / self.table_width else {
    raise InvalidField("fractal heap indirect block has too many rows")
  }
  let entries = nrows * self.table_width
  let direct_limit = self.max_direct_rows * self.table_width
  let direct_count = if entries < direct_limit { entries } else { direct_limit }
  let indirect_count = entries - direct_count
  let direct_entries : Array[(UInt64, UInt64, UInt32)] = []
  for _ in 0..<direct_count {
    let address = reader.read_uint_le(self.offset_size)
    let stored_size = if self.filter_len > 0 {
      reader.read_uint_le(self.length_size)
    } else {
      0UL
    }
    let filter_mask = if self.filter_len > 0 {
      reader.read_u32_le()
    } else {
      0U
    }
    direct_entries.push((address, stored_size, filter_mask))
  }
  let indirect_addresses : Array[UInt64] = []
  for _ in 0..<indirect_count {
    indirect_addresses.push(reader.read_uint_le(self.offset_size))
  }
  verify_metadata_checksum(
    self.bytes,
    offset,
    reader.position(),
    "fractal heap indirect block",
  )
  reader.skip(4)
  for index in 0..<direct_count {
    let block_size = self.block_size_at(index)
    let (address, stored_size, filter_mask) = direct_entries[index]
    self.add_direct_block(
      address,
      block_size,
      if self.filter_len > 0 {
        stored_size
      } else {
        block_size
      },
      filter_mask,
      depth + 1,
    )
  }
  for index in 0..<indirect_count {
    let block_size = self.block_size_at(direct_count + index)
    let child_rows = heap_log2(block_size, "fractal heap block size") -
      self.indirect_rows_base
    guard child_rows > 0 && child_rows <= 0x10000 else {
      raise InvalidField("invalid fractal heap indirect block row count")
    }
    self.add_indirect_block(
      indirect_addresses[index],
      child_rows,
      depth + 1,
      visited,
    )
  }
}

///|
fn parse_fractal_heap(
  bytes : Bytes,
  address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> FractalHeap raise Hdf5Error {
  let offset = address_to_offset(address, base_address)
  let reader = Reader::at(bytes, offset)
  guard reader.read_bytes(4) == b"FRHP" && reader.read_u8() == b'\x00' else {
    raise InvalidField("invalid fractal heap header")
  }
  let id_length = reader.read_u16_le().to_int()
  let filter_info_size = reader.read_u16_le().to_int()
  let flags = reader.read_u8().to_int()
  let max_managed_size = reader.read_u32_le().to_uint64()
  let _next_huge_id = reader.read_uint_le(length_size)
  let huge_btree_address = reader.read_uint_le(offset_size)
  let _free_space = reader.read_uint_le(length_size)
  let _free_space_manager = reader.read_uint_le(offset_size)
  let managed_space_size = reader.read_uint_le(length_size)
  let _allocated_managed_space = reader.read_uint_le(length_size)
  let _next_direct_block = reader.read_uint_le(length_size)
  let _managed_objects = reader.read_uint_le(length_size)
  let _huge_object_size = reader.read_uint_le(length_size)
  let _huge_objects = reader.read_uint_le(length_size)
  let _tiny_object_size = reader.read_uint_le(length_size)
  let _tiny_objects = reader.read_uint_le(length_size)
  let table_width = reader.read_u16_le().to_int()
  let start_block_size = reader.read_uint_le(length_size)
  let max_direct_block_size = reader.read_uint_le(length_size)
  let max_heap_bits = reader.read_u16_le().to_int()
  let _starting_rows = reader.read_u16_le().to_int()
  let root_address = reader.read_uint_le(offset_size)
  let current_rows = reader.read_u16_le().to_int()
  let mut filtered_root_size = 0UL
  let mut filtered_root_mask = 0U
  let mut encoded_filters = b""
  if filter_info_size != 0 {
    filtered_root_size = reader.read_uint_le(length_size)
    filtered_root_mask = reader.read_u32_le()
    encoded_filters = reader.read_bytes(filter_info_size)
  }
  verify_metadata_checksum(
    bytes,
    offset,
    reader.position(),
    "fractal heap header",
  )
  reader.skip(4)
  guard id_length >= 1 && id_length <= 255 else {
    raise InvalidField("invalid fractal heap ID length")
  }
  guard (flags & 0xFC) == 0 else {
    raise InvalidField("invalid fractal heap flags")
  }
  guard table_width > 0 && (table_width & (table_width - 1)) == 0 else {
    raise InvalidField("invalid fractal heap table width")
  }
  guard max_heap_bits > 0 && max_heap_bits <= 64 else {
    raise InvalidField("invalid fractal heap maximum size")
  }
  let filters : Array[Filter] = if filter_info_size != 0 {
    parse_filters({
      msg_type: msg_filter_pipeline,
      flags: b'\x00',
      data_offset: 0,
      data: encoded_filters,
    })
  } else {
    []
  }
  let huge_ids_direct = if filter_info_size != 0 {
    id_length - 1 >= offset_size + length_size * 2 + 4
  } else {
    id_length - 1 >= offset_size + length_size
  }
  let huge_id_size = if huge_ids_direct {
    if filter_info_size != 0 {
      offset_size + length_size * 2
    } else {
      offset_size + length_size
    }
  } else if id_length - 1 < 8 {
    id_length - 1
  } else {
    8
  }
  let start_bits = heap_log2(
    start_block_size, "fractal heap starting block size",
  )
  let direct_bits = heap_log2(
    max_direct_block_size, "fractal heap maximum direct block size",
  )
  guard direct_bits >= start_bits else {
    raise InvalidField("fractal heap direct block size is too small")
  }
  let offset_bits = (max_heap_bits + 7) / 8
  let managed_length_size = heap_byte_width(
    if max_direct_block_size < max_managed_size {
      max_direct_block_size
    } else {
      max_managed_size
    },
  )
  let table_bits = heap_log2(
    table_width.to_uint64(),
    "fractal heap table width",
  )
  let heap : FractalHeap = {
    bytes,
    header_address: address,
    offset_size,
    length_size,
    base_address,
    id_length,
    flags,
    table_width,
    start_block_size,
    managed_offset_size: offset_bits,
    managed_length_size,
    block_offset_size: offset_bits,
    max_direct_rows: direct_bits - start_bits + 2,
    indirect_rows_base: table_bits + start_bits - 1,
    managed_space_size,
    huge_btree_address,
    filter_len: filter_info_size,
    filters,
    huge_id_size,
    huge_ids_direct,
    blocks: [],
  }
  let undefined = heap_undefined_address(offset_size)
  if root_address != undefined {
    if current_rows == 0 {
      heap.add_direct_block(
        root_address,
        start_block_size,
        if filter_info_size != 0 {
          filtered_root_size
        } else {
          start_block_size
        },
        filtered_root_mask,
        0,
      )
    } else {
      heap.add_indirect_block(root_address, current_rows, 0, [])
    }
  }
  heap
}

///|
fn FractalHeap::find_block(
  self : FractalHeap,
  offset : UInt64,
) -> FractalHeapBlock? {
  for block in self.blocks {
    if offset >= block.heap_offset &&
      offset < block.heap_offset + block.size.to_uint64() {
      return Some(block)
    }
  }
  None
}

///|
fn FractalHeap::get_data(
  self : FractalHeap,
  heap_id : Bytes,
) -> Bytes raise Hdf5Error {
  guard heap_id.length() >= self.id_length else {
    raise InvalidField("fractal heap ID is truncated")
  }
  let first = heap_id.unsafe_get(0).to_int()
  let version = first >> 6
  guard version == 0 else {
    raise UnsupportedFeature("unsupported fractal heap ID version")
  }
  let id_type = (first >> 4) & 3
  let reader = Reader::new(heap_id)
  reader.skip(1)
  match id_type {
    0 => {
      let offset = reader.read_var_uint_le(self.managed_offset_size)
      let size = reader.read_var_uint_le(self.managed_length_size)
      guard size > 0UL && size <= 0x7FFFFFFFUL else {
        raise InvalidField("invalid fractal heap object size")
      }
      let block = match self.find_block(offset) {
        Some(block) => block
        None =>
          raise InvalidField("fractal heap object offset is outside blocks")
      }
      let relative = offset - block.heap_offset
      guard relative + size <= block.size.to_uint64() else {
        raise InvalidField("fractal heap object extends past its direct block")
      }
      let start = relative.to_int()
      guard start >= 0 && start + size.to_int() <= block.data.length() else {
        raise UnexpectedEof(start.to_int64())
      }
      block.data.get_view(start~, end=start + size.to_int()).unwrap().to_owned()
    }
    1 =>
      if self.huge_ids_direct {
        let address = reader.read_uint_le(self.offset_size)
        if self.filter_len > 0 {
          let stored_size = reader.read_uint_le(self.length_size)
          let filter_mask = reader.read_u32_le()
          let object_size = reader.read_uint_le(self.length_size)
          let stored = self.read_file_object(
            address, stored_size, "fractal heap huge object",
          )
          self.decode_stored_object(stored, filter_mask, object_size)
        } else {
          let size = reader.read_uint_le(self.length_size)
          self.read_file_object(address, size, "fractal heap huge object")
        }
      } else {
        guard self.huge_id_size > 0 && reader.remaining() >= self.huge_id_size else {
          raise InvalidField("fractal heap huge object ID is truncated")
        }
        let id = reader.read_var_uint_le(self.huge_id_size)
        let record = self.huge_record(id)
        let stored = self.read_file_object(
          record.address,
          record.stored_size,
          "fractal heap huge object",
        )
        if self.filter_len > 0 {
          self.decode_stored_object(
            stored,
            record.filter_mask,
            record.object_size,
          )
        } else {
          stored
        }
      }
    3 => raise InvalidField("invalid fractal heap ID type")
    2 => {
      let extended = self.id_length > 18
      let encoded = if extended {
        guard self.id_length >= 2 else {
          raise InvalidField("fractal heap tiny ID is truncated")
        }
        ((first & 0x0F) << 8) | heap_id.unsafe_get(1).to_int()
      } else {
        first & 0x0F
      }
      let size = encoded + 1
      let start = if extended { 2 } else { 1 }
      guard start + size <= self.id_length else {
        raise InvalidField("fractal heap tiny object exceeds its ID")
      }
      heap_id.get_view(start~, end=start + size).unwrap().to_owned()
    }
    _ => raise InvalidField("invalid fractal heap ID type")
  }
}
