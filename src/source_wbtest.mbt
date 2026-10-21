///|
priv struct CountingSource {
  bytes : Bytes
  mut reads : Int32
  mut requested : UInt64
  mut largest_read : Int32
} derive(@debug.Debug)

///|
impl Hdf5Source for CountingSource with fn length(self : CountingSource) -> UInt64 {
  self.bytes.length().to_uint64()
}

///|
impl Hdf5Source for CountingSource with fn read_at(
  self : CountingSource,
  offset : UInt64,
  length : Int32,
) -> Bytes raise Hdf5Error {
  self.reads = self.reads + 1
  self.requested = self.requested + length.to_uint64()
  if length > self.largest_read {
    self.largest_read = length
  }
  Hdf5Source::read_at(self.bytes, offset, length)
}

///|
priv struct SparseSource {
  window_start : UInt64
  bytes : Bytes
  virtual_length : UInt64
} derive(@debug.Debug)

///|
impl Hdf5Source for SparseSource with fn length(self : SparseSource) -> UInt64 {
  self.virtual_length
}

///|
impl Hdf5Source for SparseSource with fn read_at(
  self : SparseSource,
  offset : UInt64,
  length : Int32,
) -> Bytes raise Hdf5Error {
  guard length >= 0 && offset >= self.window_start else {
    raise UnexpectedEof(offset.reinterpret_as_int64())
  }
  let relative = offset - self.window_start
  let available = self.bytes.length().to_uint64()
  guard relative <= available && length.to_uint64() <= available - relative else {
    raise UnexpectedEof(offset.reinterpret_as_int64())
  }
  Hdf5Source::read_at(self.bytes, relative, length)
}

///|
test "Hdf5File accepts a random-access source" {
  let source = CountingSource::{
    bytes: @fs.read_file_to_bytes("testdata/basic_v0.h5"),
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let file = Hdf5File::parse_source(sources[0])
  assert_eq(file.open_dataset_metadata("/matrix").dataspace.dimensions, [
    2UL, 3UL,
  ])
  assert_true(source.reads > 1)
  assert_true(source.requested > 0UL)
  assert_true(source.largest_read < source.bytes.length())
}

///|
test "version 2 object headers use bounded source reads" {
  let source = CountingSource::{
    bytes: @fs.read_file_to_bytes("testdata/latest_v3.h5"),
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let file = Hdf5File::parse_source(sources[0])
  assert_eq(file.object_header_at(file.root_address()).version, 2)
  assert_true(source.reads > 1)
  assert_true(source.largest_read < source.bytes.length())
}

///|
test "contiguous selections use bounded source reads" {
  let source = CountingSource::{
    bytes: @fs.read_file_to_bytes("testdata/basic_v0.h5"),
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let file = Hdf5File::parse_source(sources[0])
  let points = file.read_dataset_selection(
    "/matrix",
    Points([[1UL, 2UL], [0UL, 1UL], [1UL, 2UL]]),
  )
  match points.values {
    Floats(values) => assert_eq(values, [5.0, 1.0, 5.0])
    _ => fail("expected floating-point points")
  }
  let slab = file.read_dataset_selection(
    "/matrix",
    Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 2UL]),
  )
  match slab.values {
    Floats(values) => assert_eq(values, [0.0, 2.0, 3.0, 5.0])
    _ => fail("expected floating-point hyperslab")
  }
  assert_true(source.largest_read < source.bytes.length())
  assert_true(source.requested < source.bytes.length().to_uint64())
}

///|
test "open_dataset reads values without materializing the source" {
  let source = CountingSource::{
    bytes: @fs.read_file_to_bytes("testdata/basic_v0.h5"),
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let file = Hdf5File::parse_source(sources[0])
  let dataset = file.open_dataset("/matrix")
  match dataset.values {
    Floats(values) => assert_eq(values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    _ => fail("expected floating-point dataset")
  }
  assert_true(source.largest_read < source.bytes.length())
  assert_true(source.requested < source.bytes.length().to_uint64())
}

///|
test "compact selections need no payload source reads" {
  let bytes = @fs.read_file_to_bytes("testdata/interop_earliest.h5")
  let parsed = Hdf5File::parse(bytes)
  let node = parsed.resolve("/data/compact")
  let header = parsed.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
  )
  let source = CountingSource::{
    bytes,
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let selected = read_source_dataset_selection(
    sources[0],
    header,
    metadata,
    Points([[1UL]]),
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
    parsed.superblock.base_address,
  )
  match selected.values {
    Integers(values) => assert_eq(values, [40L])
    _ => fail("expected compact integer selection")
  }
  assert_eq(source.reads, 0)
}

///|
test "v1 B-tree chunk selections use bounded source reads" {
  let bytes = @fs.read_file_to_bytes("testdata/interop_earliest.h5")
  let parsed = Hdf5File::parse(bytes)
  let node = parsed.resolve("/data/chunked_raw")
  let header = parsed.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
  )
  assert_true(metadata.layout.chunk_index is BTreeV1)
  let source = CountingSource::{
    bytes,
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let selected = read_source_dataset_selection(
    sources[0],
    header,
    metadata,
    Hyperslab([0UL, 1UL], [3UL, 3UL], [2UL, 2UL]),
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
    parsed.superblock.base_address,
  )
  match selected.values {
    Integers(values) =>
      assert_eq(values, [1L, 3L, 5L, 15L, 17L, 19L, 29L, 31L, 33L])
    _ => fail("expected v1 B-tree integer selection")
  }
  let filtered_node = parsed.resolve("/data/chunked_filtered")
  let filtered_header = parsed.object_header_at(filtered_node.address)
  let filtered_metadata = read_dataset_metadata(
    filtered_header,
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
  )
  let filtered = read_source_dataset_selection(
    sources[0],
    filtered_header,
    filtered_metadata,
    Points([[0UL, 0UL], [4UL, 6UL], [2UL, 2UL], [0UL, 0UL]]),
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
    parsed.superblock.base_address,
  )
  match filtered.values {
    Integers(values) => assert_eq(values, [100L, 134L, 116L, 100L])
    _ => fail("expected filtered v1 B-tree integer selection")
  }
  assert_true(source.largest_read < source.bytes.length())
  assert_true(source.requested < source.bytes.length().to_uint64())
}

///|
test "direct chunk selections use bounded source reads" {
  let bytes = @fs.read_file_to_bytes("testdata/layout_v4.h5")
  let parsed = Hdf5File::parse(bytes)
  for
    (path, selection, expected) in [
      ("/single/filtered", Points([[2UL, 3UL], [0UL, 0UL]]), [11L, 0L]),
      (
        "/implicit/edge",
        Points([[4UL, 6UL], [0UL, 0UL], [2UL, 3UL]]),
        [34L, 0L, 17L],
      ),
    ] {
    let node = parsed.resolve(path)
    let header = parsed.object_header_at(node.address)
    let metadata = read_dataset_metadata(
      header,
      parsed.superblock.offset_size,
      parsed.superblock.length_size,
    )
    let source = CountingSource::{
      bytes,
      reads: 0,
      requested: 0UL,
      largest_read: 0,
    }
    let sources : Array[&Hdf5Source] = [source]
    let selected = read_source_dataset_selection(
      sources[0],
      header,
      metadata,
      selection,
      parsed.superblock.offset_size,
      parsed.superblock.length_size,
      parsed.superblock.base_address,
    )
    match selected.values {
      Integers(values) => assert_eq(values, expected)
      _ => fail("expected direct chunk integer selection")
    }
    assert_true(source.largest_read < source.bytes.length())
    assert_true(source.requested < source.bytes.length().to_uint64())
  }
}

///|
test "fixed-array chunk selections use bounded source reads" {
  let bytes = @fs.read_file_to_bytes("testdata/fixed_array.h5")
  let parsed = Hdf5File::parse(bytes)
  for
    (path, points, expected) in [
      ("/filtered", [[4UL, 6UL], [0UL, 0UL], [2UL, 3UL]], [34L, 0L, 17L]),
      (
        "/paged_filtered",
        [[0UL], [1023UL], [2048UL], [4098UL], [1UL]],
        [0L, 1023L, 2048L, 4098L, -7L],
      ),
    ] {
    let node = parsed.resolve(path)
    let header = parsed.object_header_at(node.address)
    let metadata = read_dataset_metadata(
      header,
      parsed.superblock.offset_size,
      parsed.superblock.length_size,
    )
    assert_true(metadata.layout.chunk_index is FixedArray(_))
    let source = CountingSource::{
      bytes,
      reads: 0,
      requested: 0UL,
      largest_read: 0,
    }
    let sources : Array[&Hdf5Source] = [source]
    let selected = read_source_dataset_selection(
      sources[0],
      header,
      metadata,
      Points(points),
      parsed.superblock.offset_size,
      parsed.superblock.length_size,
      parsed.superblock.base_address,
    )
    match selected.values {
      Integers(values) => assert_eq(values, expected)
      _ => fail("expected fixed-array integer selection")
    }
    assert_true(source.largest_read < source.bytes.length())
    assert_true(source.requested < source.bytes.length().to_uint64())
  }
}

///|
test "v2 B-tree chunk selections use bounded source reads" {
  let bytes = @fs.read_file_to_bytes("testdata/btree_v2.h5")
  let parsed = Hdf5File::parse(bytes)
  let node = parsed.resolve("/filtered/edge")
  let header = parsed.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
  )
  assert_true(metadata.layout.chunk_index is BTreeV2(_))
  let source = CountingSource::{
    bytes,
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let selected = read_source_dataset_selection(
    sources[0],
    header,
    metadata,
    Points([[4UL, 6UL], [0UL, 0UL], [2UL, 3UL]]),
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
    parsed.superblock.base_address,
  )
  match selected.values {
    Integers(values) => assert_eq(values, [34L, 0L, 17L])
    _ => fail("expected v2 B-tree integer selection")
  }
  assert_true(source.largest_read < source.bytes.length())
  assert_true(source.requested < source.bytes.length().to_uint64())
}

///|
test "extensible-array chunk selections use bounded source reads" {
  let bytes = @fs.read_file_to_bytes("testdata/extensible_array.h5")
  let parsed = Hdf5File::parse(bytes)
  let node = parsed.resolve("/filtered/paged")
  let header = parsed.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
  )
  assert_true(metadata.layout.chunk_index is ExtensibleArray(_))
  let source = CountingSource::{
    bytes,
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let selected = read_source_dataset_selection(
    sources[0],
    header,
    metadata,
    Points([[199999UL], [65524UL], [1UL]]),
    parsed.superblock.offset_size,
    parsed.superblock.length_size,
    parsed.superblock.base_address,
  )
  match selected.values {
    Integers(values) => assert_eq(values, [199999L, 65524L, -7L])
    _ => fail("expected extensible-array integer selection")
  }
  assert_true(source.largest_read < source.bytes.length())
  assert_true(source.requested < source.bytes.length().to_uint64())
}

///|
test "dense v2 groups use bounded source reads" {
  let source = CountingSource::{
    bytes: @fs.read_file_to_bytes("testdata/attribute_storage.h5"),
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let file = Hdf5File::parse_source(sources[0])
  let dense = file.resolve("/dense")
  assert_eq(dense.kind, Dataset)
  assert_true(source.largest_read < source.bytes.length())
  assert_true(source.requested < source.bytes.length().to_uint64())
}

///|
test "dense attributes use bounded source reads" {
  let source = CountingSource::{
    bytes: @fs.read_file_to_bytes("testdata/attribute_storage.h5"),
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let file = Hdf5File::parse_source(sources[0])
  let attributes = file.attributes("/dense")
  assert_eq(attributes.length(), 12)
  assert_true(source.largest_read < source.bytes.length())
  assert_true(source.requested < source.bytes.length().to_uint64())
}

///|
test "random-access sources support value reads during parser migration" {
  let bytes = @fs.read_file_to_bytes("testdata/basic_v0.h5")
  let sources : Array[&Hdf5Source] = [bytes]
  let dataset = Hdf5File::parse_source(sources[0]).open_dataset("/group/values")
  match dataset.values {
    Integers(values) =>
      assert_eq(values, [0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L])
    _ => fail("expected signed integer values")
  }
}

///|
test "global heap values use bounded source reads" {
  let source = CountingSource::{
    bytes: @fs.read_file_to_bytes("testdata/datatypes_v200.h5"),
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let file = Hdf5File::parse_source(sources[0])
  assert_eq(string_values(file.open_dataset("/utf8")), ["月亮", "HDF5", ""])
  match
    file.read_dataset_selection("/sequences", Points([[2UL], [0UL], [2UL]])).values {
    Structured(values) =>
      assert_eq(values, [
        Sequence([SignedInteger(300L), SignedInteger(400L)]),
        Sequence([SignedInteger(1L), SignedInteger(-2L), SignedInteger(3L)]),
        Sequence([SignedInteger(300L), SignedInteger(400L)]),
      ])
    _ => fail("expected variable-length sequences")
  }
  assert_true(source.largest_read < source.bytes.length())
}

///|
test "random-access addresses can exceed two GiB" {
  let buffer = @buffer.Buffer::Buffer(size_hint=40)
  for byte in b"GCOL\x01\x00\x00\x00" {
    buffer.write_byte(byte)
  }
  buffer.write_uint64_le(40UL)
  buffer.write_uint16_le(1)
  buffer.write_uint16_le(1)
  buffer.write_uint_le(0U)
  buffer.write_uint64_le(5UL)
  for byte in b"large\x00\x00\x00" {
    buffer.write_byte(byte)
  }
  let address = 0x80000000UL
  let source = SparseSource::{
    window_start: address,
    bytes: buffer.to_bytes(),
    virtual_length: address + 40UL,
  }
  let sources : Array[&Hdf5Source] = [source]
  assert_eq(
    read_source_global_heap_object(sources[0], address, 1, 8, 0UL),
    b"large",
  )
}

///|
test "variable-length attributes use bounded source reads" {
  let writer = Hdf5Writer::new()
  writer.add_dataset("/names", [2UL], VariableStringValues(["moon", "月亮"]))
  writer.add_attribute(
    "/names",
    "description",
    [],
    VariableStringValues(["UTF-8 标签"]),
  )
  let source = CountingSource::{
    bytes: writer.encode(),
    reads: 0,
    requested: 0UL,
    largest_read: 0,
  }
  let sources : Array[&Hdf5Source] = [source]
  let attributes = Hdf5File::parse_source(sources[0]).attributes("/names")
  match attributes[0].values {
    Some(Strings(values)) => assert_eq(values, ["UTF-8 标签"])
    _ => fail("expected variable-length string attribute")
  }
  assert_true(source.largest_read < source.bytes.length())
}

///|
test "Bytes random-access reads validate their range" {
  assert_eq(Hdf5Source::read_at(b"abcdef", 2UL, 3), b"cde")
  let error : Hdf5Error? = try {
    Hdf5Source::read_at(b"abcdef", 5UL, 2) |> ignore
    None
  } catch {
    error => Some(error)
  }
  assert_eq(error, Some(UnexpectedEof(5L)))
}
