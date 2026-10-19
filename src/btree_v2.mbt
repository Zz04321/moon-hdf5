///|
fn validate_btree_v2_config(config : BTreeV2Config) -> Unit raise Hdf5Error {
  guard config.node_size > 10 &&
    config.split_percent > 0 &&
    config.split_percent <= 100 &&
    config.merge_percent > 0 &&
    config.merge_percent <= 100 &&
    config.merge_percent * 2 < config.split_percent else {
    raise InvalidField("invalid v2 B-tree configuration")
  }
}

///|
fn btree_integer_width(value : UInt64) -> Int32 {
  let mut width = 1
  let mut remaining = value
  while remaining > 255UL {
    remaining = remaining >> 8
    width = width + 1
  }
  width
}

///|
priv struct ChunkBTreeV2 {
  bytes : Bytes
  layout : DataLayout
  dataspace : Dataspace
  datatype : Datatype
  offset_size : Int32
  base_address : UInt64
  undefined : UInt64
  filters : Array[Filter]
  output : FixedArray[Byte]
  config : BTreeV2Config
  client : Int32
  record_size : Int32
  size_width : Int32
  chunk_size : Int32
  count_width : Int32
  capacities : Array[Int32]
  total_widths : Array[Int32]
  mut previous : Array[UInt64]?
  mut remaining_records : UInt64
}

///|
fn ChunkBTreeV2::record(
  self : ChunkBTreeV2,
  reader : Reader,
) -> Unit raise Hdf5Error {
  guard self.remaining_records > 0UL else {
    raise InvalidField("v2 B-tree exceeds declared record count")
  }
  self.remaining_records = self.remaining_records - 1UL
  let address = reader.read_uint_le(self.offset_size)
  guard address != self.undefined else {
    raise InvalidField("undefined v2 B-tree chunk address")
  }
  let stored_size = if self.client == 11 {
    reader.read_var_uint_le(self.size_width)
  } else {
    self.chunk_size.to_uint64()
  }
  let mask = if self.client == 11 { reader.read_u32_le() } else { 0U }
  let coordinates : Array[UInt64] = []
  let offsets : Array[UInt64] = []
  for dimension in 0..<self.layout.chunk_dimensions.length() {
    let coordinate = reader.read_u64_le()
    let width = self.layout.chunk_dimensions[dimension].to_uint64()
    guard self.dataspace.dimensions[dimension] > 0UL &&
      coordinate <= (self.dataspace.dimensions[dimension] - 1UL) / width else {
      raise InvalidField("v2 B-tree chunk coordinate is outside dataspace")
    }
    coordinates.push(coordinate)
    offsets.push(coordinate * width)
  }
  if self.previous is Some(previous) {
    let mut greater = false
    for dimension in 0..<coordinates.length() {
      if coordinates[dimension] != previous[dimension] {
        greater = coordinates[dimension] > previous[dimension]
        break
      }
    }
    guard greater else {
      raise InvalidField("v2 B-tree chunk records are not strictly ordered")
    }
  }
  self.previous = Some(coordinates)
  let decoded = read_filtered_chunk(
    self.bytes,
    address,
    self.base_address,
    stored_size,
    self.filters,
    mask,
    self.datatype.size,
    self.chunk_size,
  )
  copy_chunk(
    decoded,
    decoded.length(),
    offsets,
    self.layout.chunk_dimensions,
    self.dataspace.dimensions,
    self.datatype.size,
    self.output,
  )
}

///|
fn ChunkBTreeV2::node(
  self : ChunkBTreeV2,
  address : UInt64,
  depth : Int32,
  count : Int32,
  total : UInt64,
) -> Unit raise Hdf5Error {
  guard address != self.undefined &&
    count > 0 &&
    count <= self.capacities[depth] &&
    count.to_uint64() <= total &&
    total <= self.remaining_records else {
    raise InvalidField("invalid v2 B-tree node record count")
  }
  let start = address_to_offset(address, self.base_address)
  let reader = Reader::new(self.bytes).sub(start, self.config.node_size)
  guard reader.read_bytes(4) == (if depth == 0 { b"BTLF" } else { b"BTIN" }) &&
    reader.read_u8() == b'\x00' &&
    reader.read_u8().to_int() == self.client else {
    raise InvalidField("invalid v2 B-tree node")
  }
  let records_start = reader.position()
  reader.skip(count * self.record_size)
  let children : Array[(UInt64, Int32, UInt64)] = []
  let mut computed_total = count.to_uint64()
  if depth > 0 {
    for _ in 0..<(count + 1) {
      let child_address = reader.read_uint_le(self.offset_size)
      let child_count = reader.read_var_uint_le(self.count_width)
      guard child_count <= self.capacities[depth - 1].to_uint64() else {
        raise InvalidField("v2 B-tree child count exceeds capacity")
      }
      let child_total = if depth > 1 {
        reader.read_var_uint_le(self.total_widths[depth - 1])
      } else {
        child_count
      }
      guard computed_total <= total && child_total <= total - computed_total else {
        raise InvalidField("invalid v2 B-tree subtree count")
      }
      computed_total = computed_total + child_total
      children.push((child_address, child_count.to_int(), child_total))
    }
  }
  reader.skip(4)
  guard computed_total == total else {
    raise InvalidField("v2 B-tree subtree count mismatch")
  }
  for record_index in 0..<count {
    if depth > 0 {
      let (child_address, child_count, child_total) = children[record_index]
      self.node(child_address, depth - 1, child_count, child_total)
    }
    self.record(
      Reader::at(self.bytes, records_start + record_index * self.record_size),
    )
  }
  if depth > 0 {
    let (child_address, child_count, child_total) = children[count]
    self.node(child_address, depth - 1, child_count, child_total)
  }
}

///|
fn read_btree_v2_chunks(
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
  config : BTreeV2Config,
) -> Unit raise Hdf5Error {
  validate_btree_v2_config(config)
  let header = Reader::at(
    bytes,
    address_to_offset(layout.data_address, base_address),
  )
  guard header.read_bytes(4) == b"BTHD" && header.read_u8() == b'\x00' else {
    raise InvalidField("invalid v2 B-tree header")
  }
  let client = header.read_u8().to_int()
  guard client == (if filters.length() == 0 { 10 } else { 11 }) &&
    header.read_u32_le() == config.node_size.reinterpret_as_uint() else {
    raise InvalidField("invalid v2 B-tree type or node size")
  }
  let record_size = header.read_u16_le().to_int()
  let size_width = record_size -
    offset_size -
    8 * layout.chunk_dimensions.length() -
    4
  guard (client == 10 && size_width == -4) ||
    (client == 11 && size_width >= 1 && size_width <= 8) else {
    raise InvalidField("invalid v2 B-tree record size")
  }
  let depth = header.read_u16_le().to_int()
  guard depth <= 32 &&
    header.read_u8().to_int() == config.split_percent &&
    header.read_u8().to_int() == config.merge_percent else {
    raise InvalidField("invalid v2 B-tree depth or thresholds")
  }
  let root = header.read_uint_le(offset_size)
  let count = header.read_u16_le().to_int()
  let total = header.read_uint_le(length_size)
  header.skip(4)
  guard total <= (bytes.length() / record_size).to_uint64() else {
    raise InvalidField("v2 B-tree record count exceeds input")
  }
  if root == undefined {
    guard total == 0UL && count == 0 && depth == 0 else {
      raise InvalidField("invalid empty v2 B-tree")
    }
    return
  }
  let leaf_capacity = (config.node_size - 10) / record_size
  guard leaf_capacity > 0 && leaf_capacity <= 65535 else {
    raise InvalidField("invalid v2 B-tree leaf capacity")
  }
  let count_width = btree_integer_width(leaf_capacity.to_uint64())
  let capacities = [leaf_capacity]
  let total_widths = [0]
  let mut cumulative = leaf_capacity.to_uint64()
  for level in 1..<(depth + 1) {
    let pointer_size = offset_size + count_width + total_widths[level - 1]
    let capacity = (config.node_size - 10 - pointer_size) /
      (record_size + pointer_size)
    guard capacity > 0 &&
      cumulative <=
      (0xFFFFFFFFFFFFFFFFUL - capacity.to_uint64()) / (capacity + 1).to_uint64() else {
      raise InvalidField("invalid v2 B-tree internal capacity")
    }
    cumulative = cumulative * (capacity + 1).to_uint64() + capacity.to_uint64()
    capacities.push(capacity)
    total_widths.push(btree_integer_width(cumulative))
  }
  let context : ChunkBTreeV2 = {
    bytes,
    layout,
    dataspace,
    datatype,
    offset_size,
    base_address,
    undefined,
    filters,
    output,
    config,
    client,
    record_size,
    size_width,
    chunk_size: chunk_byte_size(layout.chunk_dimensions, datatype.size),
    count_width,
    capacities,
    total_widths,
    previous: None,
    remaining_records: total,
  }
  context.node(root, depth, count, total)
  guard context.remaining_records == 0UL else {
    raise InvalidField("v2 B-tree record count mismatch")
  }
}
