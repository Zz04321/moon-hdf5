///|
fn write_signature(buf : @buffer.Buffer) -> Unit {
  for byte in hdf5_signature {
    buf.write_byte(byte)
  }
}

///|
fn write_v0_superblock(buf : @buffer.Buffer) -> Unit {
  write_signature(buf)
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x08')
  buf.write_byte(b'\x08')
  buf.write_byte(b'\x00')
  buf.write_uint16_le(4)
  buf.write_uint16_le(16)
  buf.write_uint_le(0U)
  buf.write_uint64_le(0UL)
  buf.write_uint64_le(0UL)
  buf.write_uint64_le(256UL)
  buf.write_uint64_le(0UL)
  buf.write_uint64_le(0UL)
  buf.write_uint64_le(0x40UL)
  buf.write_uint_le(0U)
  buf.write_uint_le(0U)
  for _ in 0..<16 {
    buf.write_byte(b'\x00')
  }
}

///|
test "parse v0 superblock" {
  let buf = @buffer.Buffer::Buffer(size_hint=128)
  write_v0_superblock(buf)
  let superblock = parse_superblock(buf.to_bytes())
  assert_eq(superblock.version, 0)
  assert_eq(superblock.offset_size, 8)
  assert_eq(superblock.length_size, 8)
  assert_eq(superblock.group_leaf_node_k, 4)
  assert_eq(superblock.end_of_file_address, 256UL)
  assert_eq(superblock.root_group_object_header, 0x40UL)
  assert_eq(superblock.superblock_offset, 0)
  assert_eq(superblock.superblock_end, 96)
}

///|
test "parse v3 superblock" {
  let buf = @buffer.Buffer::Buffer(size_hint=64)
  write_signature(buf)
  buf.write_byte(b'\x03')
  buf.write_byte(b'\x08')
  buf.write_byte(b'\x08')
  buf.write_byte(b'\x00')
  buf.write_uint64_le(0UL)
  buf.write_uint64_le(0UL)
  buf.write_uint64_le(512UL)
  buf.write_uint64_le(0x30UL)
  buf.write_uint_le(0xDEADBEEFU)
  let superblock = parse_superblock(buf.to_bytes())
  assert_eq(superblock.version, 3)
  assert_eq(superblock.end_of_file_address, 512UL)
  assert_eq(superblock.root_group_object_header, 0x30UL)
  assert_eq(superblock.superblock_checksum, 0xDEADBEEFU)
  assert_eq(superblock.superblock_end, 48)
}

///|
test "find superblock after user block" {
  let buf = @buffer.Buffer::Buffer(size_hint=640)
  for _ in 0..<512 {
    buf.write_byte(b'\x00')
  }
  write_v0_superblock(buf)
  let superblock = parse_superblock(buf.to_bytes())
  assert_eq(superblock.superblock_offset, 512)
  assert_eq(superblock.root_group_object_header, 0x40UL)
}

///|
test "reject invalid signature" {
  let error = try {
    parse_superblock(b"not an hdf5 file") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(InvalidSignature))
}

///|
test "reject unsupported superblock version" {
  let buf = @buffer.Buffer::Buffer(size_hint=64)
  write_signature(buf)
  buf.write_byte(b'\x09')
  for _ in 0..<32 {
    buf.write_byte(b'\x00')
  }
  let error = try {
    parse_superblock(buf.to_bytes()) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(UnsupportedSuperblockVersion(9)))
}

///|
fn write_v1_object_header(buf : @buffer.Buffer) -> Unit {
  buf.write_byte(b'\x01')
  buf.write_byte(b'\x00')
  buf.write_uint16_le(1)
  buf.write_uint_le(1U)
  buf.write_uint_le(24U)
  buf.write_uint_le(0U)
  buf.write_uint16_le(0x0011)
  buf.write_uint16_le(16)
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_uint64_le(0x100UL)
  buf.write_uint64_le(0x200UL)
}

///|
test "parse v1 object header symbol table message" {
  let buf = @buffer.Buffer::Buffer(size_hint=64)
  write_v1_object_header(buf)
  let header = parse_object_header(buf.to_bytes(), 0UL, 8, 8, 0UL)
  assert_eq(header.version, 1)
  assert_eq(header.messages.length(), 1)
  let message = header.messages[0]
  assert_eq(message.msg_type, msg_symbol_table)
  assert_eq(message.data.length(), 16)
  let data = Reader::new(message.data)
  assert_eq(data.read_uint_le(8), 0x100UL)
  assert_eq(data.read_uint_le(8), 0x200UL)
}

///|
test "parse v2 object header link message" {
  let buf = @buffer.Buffer::Buffer(size_hint=32)
  for byte in b"OHDR" {
    buf.write_byte(byte)
  }
  buf.write_byte(b'\x02')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x0c')
  buf.write_byte(b'\x06')
  buf.write_uint16_le(8)
  buf.write_byte(b'\x00')
  buf.write_uint64_le(0x30UL)
  buf.write_uint_le(0U)
  let header = parse_object_header(buf.to_bytes(), 0UL, 8, 8, 0UL)
  assert_eq(header.version, 2)
  assert_eq(header.messages.length(), 1)
  assert_eq(header.messages[0].msg_type, msg_link)
  assert_eq(header.messages[0].data.length(), 8)
}

///|
test "follow v1 object header continuation" {
  let buf = @buffer.Buffer::Buffer(size_hint=320)
  buf.write_byte(b'\x01')
  buf.write_byte(b'\x00')
  buf.write_uint16_le(1)
  buf.write_uint_le(1U)
  buf.write_uint_le(24U)
  buf.write_uint_le(0U)
  buf.write_uint16_le(0x0010)
  buf.write_uint16_le(16)
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_uint64_le(0x100UL)
  buf.write_uint64_le(16UL)
  for _ in 0..<216 {
    buf.write_byte(b'\x00')
  }
  buf.write_uint16_le(0x000C)
  buf.write_uint16_le(8)
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_byte(b'\x00')
  buf.write_uint64_le(0xCAFEBABEUL)
  let header = parse_object_header(buf.to_bytes(), 0UL, 8, 8, 0UL)
  assert_eq(header.messages.length(), 2)
  assert_eq(header.messages[0].msg_type, msg_object_continuation)
  assert_eq(header.messages[1].msg_type, msg_attribute)
  assert_eq(Reader::new(header.messages[1].data).read_uint_le(8), 0xCAFEBABEUL)
}

///|
test "parse real v0 superblock and root object header" {
  let bytes = @fs.read_file_to_bytes("testdata/basic_v0.h5")
  let superblock = parse_superblock(bytes)
  assert_eq(superblock.version, 0)
  assert_eq(superblock.root_group_object_header, 96UL)
  let header = parse_object_header(
    bytes,
    superblock.root_group_object_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  assert_eq(header.version, 1)
  assert_true(header.first(msg_symbol_table) is Some(_))
}

///|
test "parse real user-block file" {
  let bytes = @fs.read_file_to_bytes("testdata/userblock.h5")
  let superblock = parse_superblock(bytes)
  assert_eq(superblock.superblock_offset, 512)
  assert_eq(superblock.base_address, 512UL)
  let header = parse_object_header(
    bytes,
    superblock.root_group_object_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  assert_eq(header.version, 1)
}

///|
test "parse real v3 superblock and v2 object header" {
  let bytes = @fs.read_file_to_bytes("testdata/latest_v3.h5")
  let superblock = parse_superblock(bytes)
  assert_eq(superblock.version, 3)
  let header = parse_object_header(
    bytes,
    superblock.root_group_object_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  assert_eq(header.version, 2)
  assert_true(header.messages.length() > 0)
}

///|
test "read real v0 group links" {
  let bytes = @fs.read_file_to_bytes("testdata/basic_v0.h5")
  let superblock = parse_superblock(bytes)
  let root = parse_object_header(
    bytes,
    superblock.root_group_object_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let symbol_table_message = root.first(msg_symbol_table).unwrap()
  let symbol_table = parse_symbol_table_message(
    symbol_table_message,
    superblock.offset_size,
    superblock.length_size,
  )
  let links = read_group_links_v1(
    bytes,
    symbol_table.btree_address,
    symbol_table.heap_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let names = links.map(link => link.name)
  assert_true(names.contains("group"))
  assert_true(names.contains("matrix"))
  let group_index = names.search("group").unwrap()
  let group_link = links[group_index]
  let group = parse_object_header(
    bytes,
    group_link.object_header_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let group_symbol_table_message = group.first(msg_symbol_table).unwrap()
  let group_symbol_table = parse_symbol_table_message(
    group_symbol_table_message,
    superblock.offset_size,
    superblock.length_size,
  )
  let group_links = read_group_links_v1(
    bytes,
    group_symbol_table.btree_address,
    group_symbol_table.heap_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let group_names = group_links.map(link => link.name)
  assert_true(group_names.contains("values"))
}

///|
test "read real v2 link messages" {
  let bytes = @fs.read_file_to_bytes("testdata/latest_v3.h5")
  let superblock = parse_superblock(bytes)
  let root = parse_object_header(
    bytes,
    superblock.root_group_object_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let links = read_group_links_v2(root, superblock.offset_size)
  let names = links.map(link => link.name)
  assert_true(names.contains("group"))
  let group_index = names.search("group").unwrap()
  let group = parse_object_header(
    bytes,
    links[group_index].object_header_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let group_links = read_group_links_v2(group, superblock.offset_size)
  assert_true(group_links.map(link => link.name).contains("values"))
}

///|
fn values_integers(values : DatasetValues) -> Array[Int64] raise Hdf5Error {
  match values {
    Integers(values) => values
    _ => raise InvalidField("expected integer dataset")
  }
}

///|
fn integer_values(dataset : Dataset) -> Array[Int64] raise Hdf5Error {
  values_integers(dataset.values)
}

///|
fn unsigned_values(dataset : Dataset) -> Array[UInt64] raise Hdf5Error {
  match dataset.values {
    UnsignedIntegers(values) => values
    _ => raise InvalidField("expected unsigned integer dataset")
  }
}

///|
fn float_values(dataset : Dataset) -> Array[Double] raise Hdf5Error {
  match dataset.values {
    Floats(values) => values
    _ => raise InvalidField("expected float dataset")
  }
}

///|
fn values_strings(values : DatasetValues) -> Array[String] raise Hdf5Error {
  match values {
    Strings(values) => values
    _ => raise InvalidField("expected string dataset")
  }
}

///|
fn string_values(dataset : Dataset) -> Array[String] raise Hdf5Error {
  values_strings(dataset.values)
}

///|
fn open_root_link(bytes : Bytes, name : String) -> GroupLink raise Hdf5Error {
  let superblock = parse_superblock(bytes)
  let root = parse_object_header(
    bytes,
    superblock.root_group_object_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let links = if root.version == 1 {
    let symbol_table = parse_symbol_table_message(
      root.first(msg_symbol_table).unwrap(),
      superblock.offset_size,
      superblock.length_size,
    )
    read_group_links_v1(
      bytes,
      symbol_table.btree_address,
      symbol_table.heap_address,
      superblock.offset_size,
      superblock.length_size,
      superblock.base_address,
    )
  } else {
    read_group_links_v2(root, superblock.offset_size)
  }
  match links.search_by(link => link.name == name) {
    Some(index) => links[index]
    None => raise InvalidField("link not found")
  }
}

///|
fn open_link(
  bytes : Bytes,
  header : ObjectHeader,
  name : String,
) -> GroupLink raise Hdf5Error {
  let superblock = parse_superblock(bytes)
  let links = if header.version == 1 {
    let symbol_table = parse_symbol_table_message(
      header.first(msg_symbol_table).unwrap(),
      superblock.offset_size,
      superblock.length_size,
    )
    read_group_links_v1(
      bytes,
      symbol_table.btree_address,
      symbol_table.heap_address,
      superblock.offset_size,
      superblock.length_size,
      superblock.base_address,
    )
  } else {
    read_group_links_v2(header, superblock.offset_size)
  }
  match links.search_by(link => link.name == name) {
    Some(index) => links[index]
    None => raise InvalidField("link not found")
  }
}

///|
test "read contiguous int and float datasets" {
  let bytes = @fs.read_file_to_bytes("testdata/basic_v0.h5")
  let superblock = parse_superblock(bytes)
  let matrix_link = open_root_link(bytes, "matrix")
  let matrix_header = parse_object_header(
    bytes,
    matrix_link.object_header_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let matrix = read_dataset(
    bytes,
    matrix_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  assert_eq(matrix.dataspace.dimensions[0], 2UL)
  assert_eq(matrix.dataspace.dimensions[1], 3UL)
  assert_eq(matrix.datatype.class, 1)
  assert_eq(matrix.datatype.size, 4)
  let matrix_values = float_values(matrix)
  assert_eq(matrix_values.length(), 6)
  assert_eq(matrix_values[0], 0.0)
  assert_eq(matrix_values[5], 5.0)

  let group_link = open_root_link(bytes, "group")
  let group_header = parse_object_header(
    bytes,
    group_link.object_header_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let values_link = open_link(bytes, group_header, "values")
  let values_header = parse_object_header(
    bytes,
    values_link.object_header_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let values = read_dataset(
    bytes,
    values_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let integers = integer_values(values)
  assert_eq(integers.length(), 10)
  assert_eq(integers[0], 0L)
  assert_eq(integers[9], 9L)
}

///|
test "read all primitive dataset types" {
  let bytes = @fs.read_file_to_bytes("testdata/types.h5")
  let superblock = parse_superblock(bytes)
  let read_named = fn(name : String) -> Dataset raise Hdf5Error {
    let link = open_root_link(bytes, name)
    let header = parse_object_header(
      bytes,
      link.object_header_address,
      superblock.offset_size,
      superblock.length_size,
      superblock.base_address,
    )
    read_dataset(
      bytes,
      header,
      superblock.offset_size,
      superblock.length_size,
      superblock.base_address,
    )
  }
  let i8 = integer_values(read_named("i8"))
  assert_eq(i8, [-1L, 2L])
  let u16 = unsigned_values(read_named("u16"))
  assert_eq(u16, [1UL, 65535UL])
  let i64 = integer_values(read_named("i64"))
  assert_eq(i64, [-1L, 1099511627776L])
  let f64 = float_values(read_named("f64"))
  assert_eq(f64, [1.5, -2.25])
  let scalar = read_named("scalar")
  assert_true(scalar.dataspace.kind is Scalar)
  assert_eq(float_values(scalar), [3.5])
  let text = string_values(read_named("text"))
  assert_eq(text, ["abc", "de"])
}

///|
test "read dataset behind user block" {
  let bytes = @fs.read_file_to_bytes("testdata/userblock.h5")
  let superblock = parse_superblock(bytes)
  assert_eq(superblock.base_address, 512UL)
  let link = open_root_link(bytes, "x")
  let header = parse_object_header(
    bytes,
    link.object_header_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let dataset = read_dataset(
    bytes,
    header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  assert_eq(integer_values(dataset), [0L, 1L, 2L, 3L])
}

///|
test "read v3 contiguous dataset" {
  let bytes = @fs.read_file_to_bytes("testdata/latest_v3.h5")
  let superblock = parse_superblock(bytes)
  let group_link = open_root_link(bytes, "group")
  let group_header = parse_object_header(
    bytes,
    group_link.object_header_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let values_link = open_link(bytes, group_header, "values")
  let values_header = parse_object_header(
    bytes,
    values_link.object_header_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let dataset = read_dataset(
    bytes,
    values_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let values = integer_values(dataset)
  assert_eq(values.length(), 8)
  assert_eq(values[0], 0L)
  assert_eq(values[7], 7L)
}

///|
test "read uncompressed chunked dataset" {
  let bytes = @fs.read_file_to_bytes("testdata/compressed.h5")
  let file = Hdf5File::parse(bytes)
  let dataset = file.open_dataset("/chunked")
  let values = integer_values(dataset)
  assert_eq(values.length(), 120)
  assert_eq(values[0], 0L)
  assert_eq(values[11], 11L)
  assert_eq(values[12], 12L)
  assert_eq(values[119], 119L)
}

///|
test "chunked edge chunks are decoded correctly" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/chunked_edge.h5"))
  let dataset = file.open_dataset("/edge")
  let values = integer_values(dataset)
  assert_eq(values.length(), 35)
  for i in 0..<35 {
    assert_eq(values[i], i.to_int64())
  }
}

///|
test "deflate chunked datasets are decoded" {
  let bytes = @fs.read_file_to_bytes("testdata/compressed.h5")
  let file = Hdf5File::parse(bytes)
  let values = integer_values(file.open_dataset("/gzip"))
  assert_eq(values.length(), 256)
  for index in 0..<256 {
    assert_eq(values[index], index.to_int64())
  }
}

///|
fn find_attribute(
  attributes : Array[Attribute],
  name : String,
) -> Attribute raise Hdf5Error {
  match attributes.search_by(attribute => attribute.name == name) {
    Some(index) => attributes[index]
    None => raise InvalidField("attribute not found")
  }
}

///|
test "read numeric and string attributes" {
  let bytes = @fs.read_file_to_bytes("testdata/attrs.h5")
  let superblock = parse_superblock(bytes)
  let root = parse_object_header(
    bytes,
    superblock.root_group_object_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let attributes = read_attributes(
    bytes,
    root,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  assert_eq(attributes.length(), 4)
  let answer = find_attribute(attributes, "answer")
  assert_true(answer.dataspace.kind is Scalar)
  assert_eq(values_integers(answer.values.unwrap()), [42L])
  let numbers = find_attribute(attributes, "numbers")
  assert_eq(numbers.dataspace.dimensions, [3UL])
  assert_eq(values_integers(numbers.values.unwrap()), [1L, 2L, 3L])
  let fixed = find_attribute(attributes, "fixed")
  assert_eq(values_strings(fixed.values.unwrap()), ["hello"])
  let title = find_attribute(attributes, "title")
  assert_eq(values_strings(title.values.unwrap()), ["hello"])

  let data_link = open_root_link(bytes, "data")
  let data_header = parse_object_header(
    bytes,
    data_link.object_header_address,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let data_attributes = read_attributes(
    bytes,
    data_header,
    superblock.offset_size,
    superblock.length_size,
    superblock.base_address,
  )
  let gain = find_attribute(data_attributes, "gain")
  let gain_values = match gain.values {
    Some(Floats(values)) => values
    _ => raise InvalidField("expected float attribute")
  }
  assert_eq(gain_values, [1.5])
  assert_eq(
    values_strings(find_attribute(data_attributes, "unit").values.unwrap()),
    ["count"],
  )
}

///|
test "high-level file API resolves paths and walks groups" {
  let bytes = @fs.read_file_to_bytes("testdata/basic_v0.h5")
  let file = Hdf5File::parse(bytes)
  assert_eq(file.resolve("/").kind, Group)
  assert_eq(file.resolve("/group").kind, Group)
  assert_eq(file.resolve("/group/values").kind, Dataset)
  let dataset = file.open_dataset("/group/values")
  assert_eq(integer_values(dataset), [0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L])
  let root_attributes = file.attributes("/")
  assert_eq(find_attribute(root_attributes, "title").name, "title")
  let nodes = file.walk()
  let paths = nodes.map(node => node.path)
  assert_true(paths.contains("/group"))
  assert_true(paths.contains("/group/values"))
  assert_true(paths.contains("/matrix"))
}

///|
test "high-level API works with user block and v3 files" {
  let userblock = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/userblock.h5"),
  )
  assert_eq(integer_values(userblock.open_dataset("/x")), [0L, 1L, 2L, 3L])
  let latest = Hdf5File::parse(@fs.read_file_to_bytes("testdata/latest_v3.h5"))
  assert_eq(integer_values(latest.open_dataset("/group/values")), [
    0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L,
  ])
}

///|
test "high-level API follows soft links and rejects external links" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/links.h5"))
  let alias_node = file.resolve("/alias")
  assert_eq(alias_node.kind, Group)
  assert_eq(integer_values(file.open_dataset("/alias/values")), [0L, 1L, 2L])
  let nodes = file.walk()
  let external = nodes.search_by(node => node.path == "/external").unwrap()
  assert_eq(nodes[external].kind, Unknown)
  let error = try {
    file.resolve("/external") |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_true(error is Some(UnsupportedFeature(_)))
}

///|
test "compact and empty groups remain readable" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/group_storage.h5"),
  )
  let empty = file.resolve("/empty")
  assert_eq(file.links(file.object_header_at(empty.address)).length(), 0)
  for path in ["/compact", "/compact_ordered"] {
    let node = file.resolve(path)
    let links = file.links(file.object_header_at(node.address))
    assert_eq(links.map(link => link.name), ["value_00", "value_01"])
    assert_eq(integer_values(file.open_dataset(path + "/value_00")), [0L, 100L])
    assert_eq(integer_values(file.open_dataset(path + "/value_01")), [1L, 101L])
  }
}

///|
test "dense groups report unsupported storage instead of missing links" {
  let file = Hdf5File::parse(
    @fs.read_file_to_bytes("testdata/group_storage.h5"),
  )
  let expected = Some(
    UnsupportedFeature("dense group storage is not supported yet"),
  )
  for path in ["/dense", "/dense_ordered"] {
    let node = file.resolve(path)
    assert_eq(node.kind, Group)
    let error = try {
      file.links(file.object_header_at(node.address)) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_eq(error, expected)
    let lookup_error = try {
      file.open_dataset(path + "/value_00") |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_eq(lookup_error, expected)
  }
  let walk_error = try {
    file.walk() |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(walk_error, expected)
}

///|
test "link info uses address-width-specific undefined addresses" {
  for width in [1, 2, 4, 8] {
    for flags in [0, 1, 3] {
      for allocated in [false, true] {
        let buffer = @buffer.Buffer::Buffer(size_hint=40)
        buffer.write_byte(b'\x00')
        buffer.write_byte(flags.to_byte())
        if (flags & 1) != 0 {
          buffer.write_uint64_le(42UL)
        }
        let addresses = if flags == 3 { 3 } else { 2 }
        for address_index in 0..<addresses {
          for _ in 0..<width {
            buffer.write_byte(
              if allocated && address_index == 0 {
                b'\x00'
              } else {
                b'\xff'
              },
            )
          }
        }
        let header : ObjectHeader = {
          version: 2,
          messages: [
            {
              msg_type: msg_link_info,
              flags: b'\x00',
              data_offset: 0,
              data: buffer.to_bytes(),
            },
          ],
        }
        let error = try {
          read_group_links_v2(header, width) |> ignore
          None
        } catch {
          error => Some(error)
        }
        if allocated {
          assert_eq(
            error,
            Some(UnsupportedFeature("dense group storage is not supported yet")),
          )
        } else {
          assert_eq(error, None)
          assert_eq(read_group_links_v2(header, width).length(), 0)
        }
      }
    }
  }
}

///|
test "invalid link info is not treated as an empty group" {
  for
    data in [
      b"\x00", b"\x00\x00\xff", b"\x00\x01", b"\x00\x02", b"\x00\x04", b"\x01\x00",
    ] {
    let header : ObjectHeader = {
      version: 2,
      messages: [
        { msg_type: msg_link_info, flags: b'\x00', data_offset: 0, data, },
      ],
    }
    let error = try {
      read_group_links_v2(header, 8) |> ignore
      None
    } catch {
      error => Some(error)
    }
    assert_true(error is Some(_))
  }
}

///|
test "read latest-format attributes" {
  let file = Hdf5File::parse(@fs.read_file_to_bytes("testdata/attrs_v3.h5"))
  let attributes = file.attributes("/")
  assert_eq(
    values_integers(find_attribute(attributes, "answer").values.unwrap()),
    [7L],
  )
  assert_eq(
    values_strings(find_attribute(attributes, "title").values.unwrap()),
    ["modern"],
  )
  assert_eq(integer_values(file.open_dataset("/data")), [0L, 1L, 2L])
  let data_attributes = file.attributes("/data")
  let scale = match find_attribute(data_attributes, "scale").values.unwrap() {
    Floats(values) => values
    _ => raise InvalidField("expected float attribute")
  }
  assert_eq(scale, [2.5])
}
