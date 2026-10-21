///|
/// A record from a metadata v2 B-tree. Group records only carry a heap ID,
/// while attribute records also carry the object-header message flags.
priv struct MetadataBTreeRecord {
  heap_id : Bytes
  flags : Byte
}

///|
fn metadata_count_width(value : UInt64) -> Int32 {
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
fn parse_metadata_btree_record(
  data : Bytes,
  node_type : Int32,
  record_size : Int32,
) -> MetadataBTreeRecord raise Hdf5Error {
  let reader = Reader::new(data)
  match node_type {
    5 => {
      guard record_size >= 5 else {
        raise InvalidField("group-name B-tree record is too small")
      }
      reader.skip(4)
      { heap_id: reader.read_bytes(record_size - 4), flags: b'\x00', }
    }
    6 => {
      guard record_size >= 9 else {
        raise InvalidField("group-order B-tree record is too small")
      }
      reader.skip(8)
      { heap_id: reader.read_bytes(record_size - 8), flags: b'\x00', }
    }
    8 | 9 => {
      guard record_size >= 9 else {
        raise InvalidField("attribute B-tree record is too small")
      }
      let heap_id = reader.read_bytes(8)
      let flags = reader.read_u8()
      { heap_id, flags, }
    }
    _ => raise UnsupportedFeature("unsupported metadata B-tree record type")
  }
}

///|
/// Shared traversal state for a v2 B-tree.
priv struct V2BTreeReader {
  source : &Hdf5Source
  base_address : UInt64
  node_size : Int32
  record_size : Int32
  node_type : Int32
  offset_size : Int32
  label : String
  capacities : Array[Int32]
  count_widths : Array[Int32]
  total_widths : Array[Int32]
  visited : Array[UInt64]
  records : Array[Bytes]
}

///|
fn V2BTreeReader::node(
  self : V2BTreeReader,
  address : UInt64,
  depth : Int32,
  count : Int32,
  total : UInt64,
) -> Unit raise Hdf5Error {
  guard address != heap_undefined_address(self.offset_size) &&
    depth >= 0 &&
    depth < self.capacities.length() &&
    count > 0 &&
    count <= self.capacities[depth] &&
    count.to_uint64() <= total else {
    raise InvalidField("invalid " + self.label + " node count")
  }
  guard !self.visited.contains(address) else {
    raise InvalidField(self.label + " contains a node cycle")
  }
  self.visited.push(address)
  let start = source_address_to_offset(address, self.base_address)
  let node = self.source.read_at(start, self.node_size)
  let reader = Reader::new(node)
  guard reader.read_bytes(4) == (if depth == 0 { b"BTLF" } else { b"BTIN" }) &&
    reader.read_u8() == b'\x00' &&
    reader.read_u8().to_int() == self.node_type else {
    raise InvalidField("invalid " + self.label + " node")
  }
  let records : Array[Bytes] = []
  for _ in 0..<count {
    records.push(reader.read_bytes(self.record_size))
  }
  let children : Array[(UInt64, Int32, UInt64)] = []
  let mut computed_total = count.to_uint64()
  if depth > 0 {
    let count_width = self.count_widths[depth]
    let total_width = self.total_widths[depth]
    for _ in 0..<(count + 1) {
      let child_address = reader.read_uint_le(self.offset_size)
      let child_count = reader.read_var_uint_le(count_width)
      guard child_count <= self.capacities[depth - 1].to_uint64() else {
        raise InvalidField(self.label + " child count exceeds capacity")
      }
      let child_total = if total_width > 0 {
        reader.read_var_uint_le(total_width)
      } else {
        child_count
      }
      guard child_total <= total - computed_total else {
        raise InvalidField("invalid " + self.label + " subtree count")
      }
      computed_total = computed_total + child_total
      children.push((child_address, child_count.to_int(), child_total))
    }
  }
  verify_metadata_checksum(node, 0, reader.position(), self.label + " node")
  reader.skip(4)
  guard computed_total == total else {
    raise InvalidField(self.label + " subtree count mismatch")
  }
  for index in 0..<count {
    if depth > 0 {
      let (child_address, child_count, child_total) = children[index]
      self.node(child_address, depth - 1, child_count, child_total)
    }
    self.records.push(records[index])
  }
  if depth > 0 {
    let (child_address, child_count, child_total) = children[count]
    self.node(child_address, depth - 1, child_count, child_total)
  }
}

///|
/// Raw records and layout information from a v2 B-tree.
priv struct V2BTreeContents {
  node_type : Int32
  record_size : Int32
  records : Array[Bytes]
}

///|
/// Reads the raw records of a v2 B-tree in key order.
fn read_v2_btree_records(
  bytes : Bytes,
  address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  expected_types : Array[Int32],
  label : String,
) -> V2BTreeContents raise Hdf5Error {
  let sources : Array[&Hdf5Source] = [bytes]
  read_source_v2_btree_records(
    sources[0],
    address,
    offset_size,
    length_size,
    base_address,
    expected_types,
    label,
  )
}

///|
fn read_source_v2_btree_records(
  source : &Hdf5Source,
  address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  expected_types : Array[Int32],
  label : String,
) -> V2BTreeContents raise Hdf5Error {
  let header_offset = source_address_to_offset(address, base_address)
  let header_content = 18 + offset_size + length_size
  let header = Reader::new(
    source_read_checksummed(
      source,
      header_offset,
      header_content,
      label + " header",
    ),
  )
  guard header.read_bytes(4) == b"BTHD" && header.read_u8() == b'\x00' else {
    raise InvalidField("invalid " + label + " header")
  }
  let node_type = header.read_u8().to_int()
  guard expected_types.contains(node_type) else {
    raise InvalidField(label + " type does not match its index")
  }
  let raw_node_size = header.read_u32_le()
  guard raw_node_size <= 0x7FFFFFFFU else {
    raise InvalidField(label + " node is too large")
  }
  let node_size = raw_node_size.reinterpret_as_int()
  let record_size = header.read_u16_le().to_int()
  let depth = header.read_u16_le().to_int()
  let split_percent = header.read_u8().to_int()
  let merge_percent = header.read_u8().to_int()
  let root_address = header.read_uint_le(offset_size)
  let root_count = header.read_u16_le().to_int()
  let total = header.read_uint_le(length_size)
  guard header.position() == header_content else {
    raise InvalidField(label + " header size mismatch")
  }
  header.skip(4)
  guard node_size > 10 && record_size > 0 && depth <= 32 else {
    raise InvalidField("invalid " + label + " header")
  }
  guard split_percent > 0 &&
    split_percent <= 100 &&
    merge_percent > 0 &&
    merge_percent <= 100 &&
    merge_percent * 2 < split_percent else {
    raise InvalidField("invalid " + label + " thresholds")
  }
  if root_address == heap_undefined_address(offset_size) {
    guard total == 0UL && root_count == 0 && depth == 0 else {
      raise InvalidField("invalid empty " + label)
    }
    return { node_type, record_size, records: [], }
  }
  let leaf_capacity = (node_size - 10) / record_size
  guard leaf_capacity > 0 && leaf_capacity <= 0xFFFF else {
    raise InvalidField(label + " leaf capacity is zero")
  }
  let capacities : Array[Int32] = [leaf_capacity]
  let count_widths : Array[Int32] = [0]
  let total_widths : Array[Int32] = [0]
  let subtree_totals : Array[UInt64] = [leaf_capacity.to_uint64()]
  for level in 1..<(depth + 1) {
    let count_width = metadata_count_width(capacities[level - 1].to_uint64())
    let total_width = if level > 1 {
      metadata_count_width(subtree_totals[level - 1])
    } else {
      0
    }
    let pointer_size = offset_size + count_width + total_width
    let capacity = (node_size - 10 - pointer_size) /
      (record_size + pointer_size)
    guard capacity > 0 && capacity <= 0xFFFF else {
      raise InvalidField("invalid " + label + " internal capacity")
    }
    capacities.push(capacity)
    count_widths.push(count_width)
    total_widths.push(total_width)
    let child_factor = (capacity + 1).to_uint64()
    guard subtree_totals[level - 1] <=
      (0xFFFFFFFFFFFFFFFFUL - capacity.to_uint64()) / child_factor else {
      raise InvalidField(label + " subtree count overflow")
    }
    subtree_totals.push(
      subtree_totals[level - 1] * child_factor + capacity.to_uint64(),
    )
  }
  guard total <= 0x7FFFFFFFUL else {
    raise InvalidField(label + " has too many records")
  }
  let context : V2BTreeReader = {
    source,
    base_address,
    node_size,
    record_size,
    node_type,
    offset_size,
    label,
    capacities,
    count_widths,
    total_widths,
    visited: [],
    records: [],
  }
  context.node(root_address, depth, root_count, total)
  guard context.records.length().to_uint64() == total else {
    raise InvalidField(label + " record count mismatch")
  }
  { node_type, record_size, records: context.records, }
}

///|
/// Reads heap-ID records from a group or attribute metadata v2 B-tree.
fn read_metadata_btree_records(
  bytes : Bytes,
  address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  expected_types : Array[Int32],
) -> Array[MetadataBTreeRecord] raise Hdf5Error {
  let contents = read_v2_btree_records(
    bytes, address, offset_size, length_size, base_address, expected_types, "metadata v2 B-tree",
  )
  contents.records.map(record => {
    parse_metadata_btree_record(
      record,
      contents.node_type,
      contents.record_size,
    )
  })
}

///|
fn read_source_metadata_btree_records(
  source : &Hdf5Source,
  address : UInt64,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
  expected_types : Array[Int32],
) -> Array[MetadataBTreeRecord] raise Hdf5Error {
  let contents = read_source_v2_btree_records(
    source, address, offset_size, length_size, base_address, expected_types, "metadata v2 B-tree",
  )
  contents.records.map(record => {
    parse_metadata_btree_record(
      record,
      contents.node_type,
      contents.record_size,
    )
  })
}
