///|
/// Values accepted by the initial HDF5 writer.
pub(all) enum Hdf5WriteValues {
  Int64Values(Array[Int64])
  UInt64Values(Array[UInt64])
  Float64Values(Array[Double])
  FixedStringValues(Array[String], Int32)
} derive(@debug.Debug)

///|
pub extend Hdf5WriteValues with @debug.Debug::{to_repr}

///|
/// Filters available for chunked writer datasets.
pub(all) enum Hdf5WriteFilter {
  Shuffle
  Deflate(Int32)
  Fletcher32
} derive(Eq, @debug.Debug)

///|
pub extend Hdf5WriteFilter with Eq::{not_equal, equal}

///|
pub extend Hdf5WriteFilter with @debug.Debug::{to_repr}

///|
priv enum WritableStorage {
  WritableContiguous
  WritableChunked(Array[UInt32], Array[Hdf5WriteFilter])
}

///|
priv struct WritableDataset {
  path : String
  dimensions : Array[UInt64]
  values : Hdf5WriteValues
  storage : WritableStorage
  fill_value : Hdf5WriteValues?
}

///|
priv struct WritableAttribute {
  object_path : String
  name : String
  dimensions : Array[UInt64]
  values : Hdf5WriteValues
}

///|
/// In-memory builder for a new HDF5 file.
pub struct Hdf5Writer {
  priv groups : Array[String]
  priv datasets : Array[WritableDataset]
  priv attributes : Array[WritableAttribute]
}

///|
priv enum WritableObjectKind {
  WritableGroup
  WritableDatasetObject(Int32)
}

///|
priv struct WritableObject {
  path : String
  kind : WritableObjectKind
  address : UInt64
  raw_address : UInt64
}

///|
priv struct EncodedWriteValues {
  datatype : Bytes
  payload : Bytes
  element_size : Int32
}

///|
priv struct EncodedChunkEntry {
  allocated : Bool
  data : Bytes
  address : UInt64
}

///|
pub fn Hdf5Writer::new() -> Hdf5Writer {
  { groups: ["/"], datasets: [], attributes: [], }
}

///|
fn writer_write_bytes(buffer : @buffer.Buffer, bytes : Bytes) -> Unit {
  buffer.write_bytes(bytes.get_view().unwrap())
}

///|
fn writer_write_zeros(buffer : @buffer.Buffer, count : Int32) -> Unit {
  for _ in 0..<count {
    buffer.write_byte(b'\x00')
  }
}

///|
fn writer_align(value : Int32) -> Int32 {
  (value + 7) / 8 * 8
}

///|
fn writer_path_segments(path : String) -> Array[String] raise Hdf5Error {
  guard path.has_prefix("/") else {
    raise InvalidField("writer paths must be absolute")
  }
  let segments : Array[String] = []
  for raw in path.split("/") {
    let segment = raw.to_owned()
    if segment != "" {
      guard segment != "." && segment != ".." else {
        raise InvalidField("writer paths cannot contain dot segments")
      }
      let encoded = @utf8.encode(segment)
      guard encoded.length() <= 65535 else {
        raise InvalidField("writer path component is too long")
      }
      for byte in encoded {
        guard byte != b'\x00' else {
          raise InvalidField("writer paths cannot contain NUL")
        }
      }
      segments.push(segment)
    }
  }
  segments
}

///|
fn writer_path(segments : Array[String]) -> String {
  let mut path = ""
  for segment in segments {
    path = path + "/" + segment
  }
  if path == "" {
    "/"
  } else {
    path
  }
}

///|
fn canonical_writer_path(path : String) -> String raise Hdf5Error {
  writer_path(writer_path_segments(path))
}

///|
fn writer_parent_path(path : String) -> String raise Hdf5Error {
  let segments = writer_path_segments(path)
  guard segments.length() > 0 else { raise InvalidField("root has no parent") }
  segments.pop() |> ignore
  writer_path(segments)
}

///|
fn writer_basename(path : String) -> String raise Hdf5Error {
  let segments = writer_path_segments(path)
  guard segments.length() > 0 else { raise InvalidField("root has no name") }
  segments[segments.length() - 1]
}

///|
fn writer_value_count(values : Hdf5WriteValues) -> Int32 {
  match values {
    Int64Values(items) => items.length()
    UInt64Values(items) => items.length()
    Float64Values(items) => items.length()
    FixedStringValues(items, _) => items.length()
  }
}

///|
fn copy_write_values(values : Hdf5WriteValues) -> Hdf5WriteValues {
  match values {
    Int64Values(items) => Int64Values(items.copy())
    UInt64Values(items) => UInt64Values(items.copy())
    Float64Values(items) => Float64Values(items.copy())
    FixedStringValues(items, width) => FixedStringValues(items.copy(), width)
  }
}

///|
fn writer_shape_count(dimensions : Array[UInt64]) -> Int32 raise Hdf5Error {
  let mut count = 1UL
  for dimension in dimensions {
    guard dimension <= 0x7FFFFFFFUL else {
      raise InvalidField("writer dimension is too large")
    }
  }
  if dimensions.contains(0UL) {
    return 0
  }
  for dimension in dimensions {
    guard count <= 0x7FFFFFFFUL / dimension else {
      raise InvalidField("writer dataspace is too large")
    }
    count = count * dimension
  }
  count.to_int()
}

///|
fn validate_write_values(
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
) -> Unit raise Hdf5Error {
  guard dimensions.length() <= 32 else {
    raise InvalidField("writer dataspace rank exceeds 32")
  }
  guard writer_shape_count(dimensions) == writer_value_count(values) else {
    raise InvalidField("writer value count does not match dataspace")
  }
  match values {
    FixedStringValues(items, width) => {
      guard width > 0 && width <= 65535 else {
        raise InvalidField("fixed string width is invalid")
      }
      for item in items {
        guard @utf8.encode(item).length() <= width else {
          raise InvalidField("fixed string value exceeds its width")
        }
      }
      guard items.length() <= 0x7FFFFFFF / width else {
        raise InvalidField("writer dataset byte size is too large")
      }
    }
    Int64Values(items) => {
      guard items.length() <= 0x7FFFFFFF / 8 else {
        raise InvalidField("writer dataset byte size is too large")
      }
    }
    UInt64Values(items) => {
      guard items.length() <= 0x7FFFFFFF / 8 else {
        raise InvalidField("writer dataset byte size is too large")
      }
    }
    Float64Values(items) => {
      guard items.length() <= 0x7FFFFFFF / 8 else {
        raise InvalidField("writer dataset byte size is too large")
      }
    }
  }
}

///|
fn Hdf5Writer::has_object(self : Hdf5Writer, path : String) -> Bool {
  self.groups.contains(path) ||
  self.datasets.search_by(dataset => dataset.path == path) is Some(_)
}

///|
fn Hdf5Writer::ensure_groups(
  self : Hdf5Writer,
  path : String,
) -> Unit raise Hdf5Error {
  let segments = writer_path_segments(path)
  let mut current = ""
  for segment in segments {
    current = current + "/" + segment
    guard self.datasets.search_by(dataset => dataset.path == current) is None else {
      raise InvalidField("writer path conflicts with a dataset")
    }
    if !self.groups.contains(current) {
      self.groups.push(current)
    }
  }
}

///|
/// Adds a group and any missing parent groups.
pub fn Hdf5Writer::add_group(
  self : Hdf5Writer,
  path : String,
) -> Unit raise Hdf5Error {
  self.ensure_groups(canonical_writer_path(path))
}

///|
/// Adds a contiguous dataset. Empty `dimensions` denotes a scalar dataspace.
pub fn Hdf5Writer::add_dataset(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
) -> Unit raise Hdf5Error {
  let normalized = canonical_writer_path(path)
  guard normalized != "/" else {
    raise InvalidField("a dataset cannot replace the root group")
  }
  guard !self.has_object(normalized) else {
    raise InvalidField("writer object path already exists")
  }
  validate_write_values(dimensions, values)
  self.ensure_groups(writer_parent_path(normalized))
  self.datasets.push({
    path: normalized,
    dimensions: dimensions.copy(),
    values: copy_write_values(values),
    storage: WritableContiguous,
    fill_value: None,
  })
}

///|
fn same_write_datatype(left : Hdf5WriteValues, right : Hdf5WriteValues) -> Bool {
  match (left, right) {
    (Int64Values(_), Int64Values(_)) => true
    (UInt64Values(_), UInt64Values(_)) => true
    (Float64Values(_), Float64Values(_)) => true
    (FixedStringValues(_, left_width), FixedStringValues(_, right_width)) =>
      left_width == right_width
    _ => false
  }
}

///|
fn validate_write_filters(
  filters : Array[Hdf5WriteFilter],
) -> Unit raise Hdf5Error {
  guard filters.length() <= 32 else { raise InvalidField("too many filters") }
  let mut saw_shuffle = false
  let mut saw_deflate = false
  let mut saw_fletcher = false
  for filter in filters {
    match filter {
      Shuffle => {
        guard !saw_shuffle && !saw_deflate && !saw_fletcher else {
          raise InvalidField("shuffle must be the first writer filter")
        }
        saw_shuffle = true
      }
      Deflate(level) => {
        guard !saw_deflate && !saw_fletcher && level >= 0 && level <= 9 else {
          raise InvalidField("invalid deflate writer filter")
        }
        saw_deflate = true
      }
      Fletcher32 => {
        guard !saw_fletcher else {
          raise InvalidField("duplicate Fletcher32 writer filter")
        }
        saw_fletcher = true
      }
    }
  }
}

///|
/// Adds a finite chunked dataset using a fixed-array chunk index.
///
/// Chunks containing only the fill value are left unallocated. Filters are
/// applied in the supplied HDF5 pipeline order.
pub fn Hdf5Writer::add_chunked_dataset(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  chunk_dimensions : Array[UInt32],
  values : Hdf5WriteValues,
  fill_value? : Hdf5WriteValues,
  filters~ : Array[Hdf5WriteFilter] = [],
) -> Unit raise Hdf5Error {
  let normalized = canonical_writer_path(path)
  guard normalized != "/" else {
    raise InvalidField("a dataset cannot replace the root group")
  }
  guard !self.has_object(normalized) else {
    raise InvalidField("writer object path already exists")
  }
  validate_write_values(dimensions, values)
  guard dimensions.length() > 0 &&
    chunk_dimensions.length() == dimensions.length() else {
    raise InvalidField("chunk dimensions must match a simple dataspace")
  }
  for axis in 0..<chunk_dimensions.length() {
    let dimension = chunk_dimensions[axis]
    guard dimensions[axis] > 0UL && dimension > 0U && dimension <= 0x7FFFFFFFU else {
      raise InvalidField("chunk dimension is invalid")
    }
  }
  let (_, total_chunks) = chunk_grid(dimensions, chunk_dimensions)
  guard total_chunks <= 1024 else {
    raise InvalidField("writer fixed-array index exceeds 1024 chunks")
  }
  let element_size = encode_write_values(values).element_size
  let mut chunk_bytes = element_size.to_int64()
  for dimension in chunk_dimensions {
    guard chunk_bytes <= 0x7FFFFFFFL / dimension.to_int64() else {
      raise InvalidField("writer chunk is too large")
    }
    chunk_bytes = chunk_bytes * dimension.to_int64()
  }
  validate_write_filters(filters)
  let copied_fill = match fill_value {
    Some(fill) => {
      guard writer_value_count(fill) == 1 && same_write_datatype(values, fill) else {
        raise InvalidField("fill value must be one value of the dataset type")
      }
      Some(copy_write_values(fill))
    }
    None => None
  }
  self.ensure_groups(writer_parent_path(normalized))
  self.datasets.push({
    path: normalized,
    dimensions: dimensions.copy(),
    values: copy_write_values(values),
    storage: WritableChunked(chunk_dimensions.copy(), filters.copy()),
    fill_value: copied_fill,
  })
}

///|
/// Adds an inline attribute to an existing group or dataset.
pub fn Hdf5Writer::add_attribute(
  self : Hdf5Writer,
  object_path : String,
  name : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
) -> Unit raise Hdf5Error {
  let normalized = canonical_writer_path(object_path)
  guard self.has_object(normalized) else {
    raise InvalidField("attribute target does not exist")
  }
  let encoded_name = @utf8.encode(name)
  guard encoded_name.length() > 0 && encoded_name.length() < 65535 else {
    raise InvalidField("attribute name is empty or too long")
  }
  for byte in encoded_name {
    guard byte != b'\x00' else {
      raise InvalidField("attribute names cannot contain NUL")
    }
  }
  guard self.attributes.search_by(attribute => {
      attribute.object_path == normalized && attribute.name == name
    })
    is None else {
    raise InvalidField("attribute name already exists")
  }
  validate_write_values(dimensions, values)
  self.attributes.push({
    object_path: normalized,
    name,
    dimensions: dimensions.copy(),
    values: copy_write_values(values),
  })
}

///|
fn encode_integer_datatype(signed : Bool) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=12)
  buffer.write_byte(b'\x10')
  buffer.write_byte(if signed { b'\x08' } else { b'\x00' })
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(8U)
  buffer.write_uint16_le(0)
  buffer.write_uint16_le(64)
  buffer.to_bytes()
}

///|
fn encode_float64_datatype() -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=20)
  buffer.write_byte(b'\x11')
  buffer.write_byte(b'\x20')
  buffer.write_byte(b'\x3f')
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(8U)
  buffer.write_uint16_le(0)
  buffer.write_uint16_le(64)
  buffer.write_byte(b'\x34')
  buffer.write_byte(b'\x0b')
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x34')
  buffer.write_uint_le(1023U)
  buffer.to_bytes()
}

///|
fn encode_string_datatype(width : Int32) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=8)
  buffer.write_byte(b'\x13')
  buffer.write_byte(b'\x11')
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(width.reinterpret_as_uint())
  buffer.to_bytes()
}

///|
fn encode_write_values(values : Hdf5WriteValues) -> EncodedWriteValues {
  let payload = @buffer.Buffer::Buffer()
  let (datatype, element_size) = match values {
    Int64Values(items) => {
      for item in items {
        payload.write_int64_le(item)
      }
      (encode_integer_datatype(true), 8)
    }
    UInt64Values(items) => {
      for item in items {
        payload.write_uint64_le(item)
      }
      (encode_integer_datatype(false), 8)
    }
    Float64Values(items) => {
      for item in items {
        payload.write_double_le(item)
      }
      (encode_float64_datatype(), 8)
    }
    FixedStringValues(items, width) => {
      for item in items {
        let encoded = @utf8.encode(item)
        writer_write_bytes(payload, encoded)
        writer_write_zeros(payload, width - encoded.length())
      }
      (encode_string_datatype(width), width)
    }
  }
  { datatype, payload: payload.to_bytes(), element_size, }
}

///|
fn encode_dataspace(dimensions : Array[UInt64]) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=4 + dimensions.length() * 8)
  buffer.write_byte(b'\x02')
  buffer.write_byte(dimensions.length().to_byte())
  buffer.write_byte(b'\x00')
  buffer.write_byte(if dimensions.length() == 0 { b'\x00' } else { b'\x01' })
  for dimension in dimensions {
    buffer.write_uint64_le(dimension)
  }
  buffer.to_bytes()
}

///|
fn encode_contiguous_layout(address : UInt64, size : Int32) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=18)
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x01')
  buffer.write_uint64_le(address)
  buffer.write_uint64_le(size.to_uint64())
  buffer.to_bytes()
}

///|
fn encode_chunked_layout(
  address : UInt64,
  chunk_dimensions : Array[UInt32],
  element_size : Int32,
) -> Bytes {
  let mut largest = element_size.to_uint64()
  for dimension in chunk_dimensions {
    if dimension.to_uint64() > largest {
      largest = dimension.to_uint64()
    }
  }
  let mut width = 1
  let mut remaining = largest >> 8
  while remaining > 0UL {
    width = width + 1
    remaining = remaining >> 8
  }
  let buffer = @buffer.Buffer::Buffer(
    size_hint=8 + (chunk_dimensions.length() + 1) * width + 8,
  )
  buffer.write_byte(b'\x04')
  buffer.write_byte(b'\x02')
  buffer.write_byte(b'\x00')
  buffer.write_byte((chunk_dimensions.length() + 1).to_byte())
  buffer.write_byte(width.to_byte())
  for dimension in chunk_dimensions {
    for shift in 0..<width {
      buffer.write_byte(
        ((dimension.to_uint64() >> (shift * 8)) & 0xFFUL).to_byte(),
      )
    }
  }
  for shift in 0..<width {
    buffer.write_byte(
      ((element_size.to_uint64() >> (shift * 8)) & 0xFFUL).to_byte(),
    )
  }
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x0a')
  buffer.write_uint64_le(address)
  buffer.to_bytes()
}

///|
fn encode_fill_value(fill : Hdf5WriteValues) -> Bytes {
  let encoded = encode_write_values(fill)
  let buffer = @buffer.Buffer::Buffer(size_hint=6 + encoded.element_size)
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x22')
  buffer.write_uint_le(encoded.element_size.reinterpret_as_uint())
  writer_write_bytes(buffer, encoded.payload)
  buffer.to_bytes()
}

///|
fn encode_filter_pipeline(
  filters : Array[Hdf5WriteFilter],
  element_size : Int32,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer()
  buffer.write_byte(b'\x02')
  buffer.write_byte(filters.length().to_byte())
  for filter in filters {
    match filter {
      Shuffle => {
        buffer.write_uint16_le(2)
        buffer.write_uint16_le(0)
        buffer.write_uint16_le(1)
        buffer.write_uint_le(element_size.reinterpret_as_uint())
      }
      Deflate(level) => {
        buffer.write_uint16_le(1)
        buffer.write_uint16_le(0)
        buffer.write_uint16_le(1)
        buffer.write_uint_le(level.reinterpret_as_uint())
      }
      Fletcher32 => {
        buffer.write_uint16_le(3)
        buffer.write_uint16_le(0)
        buffer.write_uint16_le(0)
      }
    }
  }
  buffer.to_bytes()
}

///|
fn shuffle_chunk(data : Bytes, element_size : Int32) -> Bytes {
  let elements = data.length() / element_size
  let output = FixedArray::make(data.length(), b'\x00')
  for byte_index in 0..<element_size {
    for element in 0..<elements {
      output[byte_index * elements + element] = data[element * element_size +
        byte_index]
    }
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn add_fletcher32(data : Bytes) -> Bytes {
  let mut first = 0U
  let mut second = 0U
  let mut offset = 0
  while offset < data.length() {
    let high = data[offset].to_uint() << 8
    let low = if offset + 1 < data.length() {
      data[offset + 1].to_uint()
    } else {
      0U
    }
    first = first + high + low
    first = (first & 0xFFFFU) + (first >> 16)
    second = second + first
    second = (second & 0xFFFFU) + (second >> 16)
    offset = offset + 2
  }
  first = (first & 0xFFFFU) + (first >> 16)
  second = (second & 0xFFFFU) + (second >> 16)
  let checksum = (second << 16) | first
  let legacy = ((checksum & 0x00FF00FFU) << 8) | ((checksum & 0xFF00FF00U) >> 8)
  let buffer = @buffer.Buffer::Buffer(size_hint=data.length() + 4)
  writer_write_bytes(buffer, data)
  buffer.write_uint_le(legacy)
  buffer.to_bytes()
}

///|
fn encode_chunk_filters(
  data : Bytes,
  filters : Array[Hdf5WriteFilter],
  element_size : Int32,
) -> Bytes {
  let mut output = data
  for filter in filters {
    output = match filter {
      Shuffle => shuffle_chunk(output, element_size)
      Deflate(level) => @zlib.zlib(output[:], level~)
      Fletcher32 => add_fletcher32(output)
    }
  }
  output
}

///|
fn write_repeated_element(output : FixedArray[Byte], element : Bytes) -> Unit {
  let count = output.length() / element.length()
  for index in 0..<count {
    output.blit_from_bytes(
      index * element.length(),
      element,
      0,
      element.length(),
    )
  }
}

///|
fn element_differs(payload : Bytes, offset : Int32, fill : Bytes) -> Bool {
  for byte_index in 0..<fill.length() {
    if payload[offset + byte_index] != fill[byte_index] {
      return true
    }
  }
  false
}

///|
fn chunk_grid(
  dimensions : Array[UInt64],
  chunk_dimensions : Array[UInt32],
) -> (Array[Int32], Int32) raise Hdf5Error {
  let grid : Array[Int32] = []
  let mut total = 1L
  for axis in 0..<dimensions.length() {
    let dimension = dimensions[axis]
    let width = chunk_dimensions[axis].to_uint64()
    let count = (dimension + width - 1UL) / width
    guard count <= 0x7FFFFFFFUL &&
      total <= 0x7FFFFFFFL / count.reinterpret_as_int64() else {
      raise InvalidField("writer chunk grid is too large")
    }
    grid.push(count.to_int())
    total = total * count.reinterpret_as_int64()
  }
  (grid, total.to_int())
}

///|
fn encode_dataset_chunks(
  dataset : WritableDataset,
) -> Array[EncodedChunkEntry] raise Hdf5Error {
  let (chunk_dimensions, filters) = match dataset.storage {
    WritableChunked(dimensions, filters) => (dimensions, filters)
    WritableContiguous => raise InvalidField("expected chunked writer dataset")
  }
  let encoded = encode_write_values(dataset.values)
  let fill = match dataset.fill_value {
    Some(value) => encode_write_values(value).payload
    None => Bytes::make(encoded.element_size, b'\x00')
  }
  let (grid, chunk_count) = chunk_grid(dataset.dimensions, chunk_dimensions)
  let mut chunk_elements = 1L
  for dimension in chunk_dimensions {
    guard chunk_elements <= 0x7FFFFFFFL / dimension.to_int64() else {
      raise InvalidField("writer chunk is too large")
    }
    chunk_elements = chunk_elements * dimension.to_int64()
  }
  guard chunk_elements <= 0x7FFFFFFFL / encoded.element_size.to_int64() else {
    raise InvalidField("writer chunk is too large")
  }
  let chunk_size = (chunk_elements * encoded.element_size.to_int64()).to_int()
  let chunk_buffers : Array[FixedArray[Byte]] = []
  let allocated = Array::make(chunk_count, false)
  for _ in 0..<chunk_count {
    let chunk = FixedArray::make(chunk_size, b'\x00')
    write_repeated_element(chunk, fill)
    chunk_buffers.push(chunk)
  }
  let element_count = writer_shape_count(dataset.dimensions)
  let rank = dataset.dimensions.length()
  for linear in 0..<element_count {
    let coordinates = Array::make(rank, 0)
    let mut remaining = linear.to_int64()
    let mut axis = rank
    while axis > 0 {
      axis = axis - 1
      let dimension = dataset.dimensions[axis].reinterpret_as_int64()
      coordinates[axis] = (remaining % dimension).to_int()
      remaining = remaining / dimension
    }
    let mut chunk_index = 0
    let mut within_index = 0
    for dimension in 0..<rank {
      let width = chunk_dimensions[dimension].reinterpret_as_int()
      chunk_index = chunk_index * grid[dimension] +
        coordinates[dimension] / width
      within_index = within_index * width + coordinates[dimension] % width
    }
    let source_offset = linear * encoded.element_size
    let destination_offset = within_index * encoded.element_size
    chunk_buffers[chunk_index].blit_from_bytes(
      destination_offset,
      encoded.payload,
      source_offset,
      encoded.element_size,
    )
    if element_differs(encoded.payload, source_offset, fill) {
      allocated[chunk_index] = true
    }
  }
  let entries : Array[EncodedChunkEntry] = []
  for index in 0..<chunk_count {
    let data = if allocated[index] {
      encode_chunk_filters(
        chunk_buffers[index].unsafe_reinterpret_as_bytes(),
        filters,
        encoded.element_size,
      )
    } else {
      b""
    }
    entries.push({ allocated: allocated[index], data, address: 0UL, })
  }
  entries
}

///|
fn encode_fixed_array_storage(
  start : UInt64,
  dataset : WritableDataset,
) -> Bytes raise Hdf5Error {
  let (_, filters) = match dataset.storage {
    WritableChunked(dimensions, filters) => (dimensions, filters)
    WritableContiguous => raise InvalidField("expected chunked writer dataset")
  }
  let entries = encode_dataset_chunks(dataset)
  let filtered = filters.length() > 0
  let entry_size = if filtered { 14 } else { 8 }
  let header_length = 28
  let block_address = writer_align(start.to_int() + header_length).to_uint64()
  let block_length = 14 + entries.length() * entry_size + 4
  let mut chunk_address = writer_align(block_address.to_int() + block_length)
  for index in 0..<entries.length() {
    if entries[index].allocated {
      entries[index] = {
        allocated: true,
        data: entries[index].data,
        address: chunk_address.to_uint64(),
      }
      chunk_address = writer_align(chunk_address + entries[index].data.length())
    }
  }

  let header = @buffer.Buffer::Buffer(size_hint=header_length)
  writer_write_bytes(header, b"FAHD")
  header.write_byte(b'\x00')
  header.write_byte(if filtered { b'\x01' } else { b'\x00' })
  header.write_byte(entry_size.to_byte())
  header.write_byte(b'\x0a')
  header.write_uint64_le(entries.length().to_uint64())
  header.write_uint64_le(block_address)
  header.write_uint_le(metadata_checksum(header.to_bytes(), 0, header.length()))

  let block = @buffer.Buffer::Buffer(size_hint=block_length)
  writer_write_bytes(block, b"FADB")
  block.write_byte(b'\x00')
  block.write_byte(if filtered { b'\x01' } else { b'\x00' })
  block.write_uint64_le(start)
  for entry in entries {
    block.write_uint64_le(
      if entry.allocated {
        entry.address
      } else {
        0xFFFFFFFFFFFFFFFFUL
      },
    )
    if filtered {
      guard entry.data.length() <= 65535 else {
        raise InvalidField("filtered writer chunk is too large")
      }
      block.write_uint16_le(entry.data.length().to_uint16())
      block.write_uint_le(0U)
    }
  }
  block.write_uint_le(metadata_checksum(block.to_bytes(), 0, block.length()))

  let output = @buffer.Buffer::Buffer(size_hint=chunk_address - start.to_int())
  writer_write_bytes(output, header.to_bytes())
  writer_write_zeros(
    output,
    block_address.to_int() - start.to_int() - output.length(),
  )
  writer_write_bytes(output, block.to_bytes())
  for entry in entries {
    if entry.allocated {
      writer_write_zeros(
        output,
        entry.address.to_int() - start.to_int() - output.length(),
      )
      writer_write_bytes(output, entry.data)
    }
  }
  writer_write_zeros(output, chunk_address - start.to_int() - output.length())
  output.to_bytes()
}

///|
fn encode_link_info() -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=18)
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x00')
  buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  buffer.to_bytes()
}

///|
fn encode_hard_link(name : String, address : UInt64) -> Bytes {
  let encoded = @utf8.encode(name)
  let buffer = @buffer.Buffer::Buffer(size_hint=12 + encoded.length())
  buffer.write_byte(b'\x01')
  buffer.write_byte(b'\x11')
  buffer.write_byte(b'\x01')
  buffer.write_uint16_le(encoded.length().to_uint16())
  writer_write_bytes(buffer, encoded)
  buffer.write_uint64_le(address)
  buffer.to_bytes()
}

///|
fn encode_attribute(attribute : WritableAttribute) -> Bytes {
  let encoded = encode_write_values(attribute.values)
  let name = @utf8.encode(attribute.name)
  let dataspace = encode_dataspace(attribute.dimensions)
  let buffer = @buffer.Buffer::Buffer(
    size_hint=9 +
      name.length() +
      encoded.datatype.length() +
      dataspace.length() +
      encoded.payload.length(),
  )
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x00')
  buffer.write_uint16_le((name.length() + 1).to_uint16())
  buffer.write_uint16_le(encoded.datatype.length().to_uint16())
  buffer.write_uint16_le(dataspace.length().to_uint16())
  buffer.write_byte(b'\x01')
  writer_write_bytes(buffer, name)
  buffer.write_byte(b'\x00')
  writer_write_bytes(buffer, encoded.datatype)
  writer_write_bytes(buffer, dataspace)
  writer_write_bytes(buffer, encoded.payload)
  buffer.to_bytes()
}

///|
fn write_header_message(
  buffer : @buffer.Buffer,
  message_type : Int32,
  data : Bytes,
) -> Unit raise Hdf5Error {
  guard data.length() <= 65535 else {
    raise InvalidField("object header message is too large")
  }
  buffer.write_byte(message_type.to_byte())
  buffer.write_uint16_le(data.length().to_uint16())
  buffer.write_byte(b'\x00')
  writer_write_bytes(buffer, data)
}

///|
fn encode_object_header(payload : Bytes) -> Bytes raise Hdf5Error {
  guard payload.length() <= 65535 else {
    raise InvalidField("object header is too large")
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=12 + payload.length())
  writer_write_bytes(buffer, b"OHDR")
  buffer.write_byte(b'\x02')
  buffer.write_byte(b'\x01')
  buffer.write_uint16_le(payload.length().to_uint16())
  writer_write_bytes(buffer, payload)
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  buffer.to_bytes()
}

///|
fn Hdf5Writer::object_attributes(
  self : Hdf5Writer,
  path : String,
) -> Array[WritableAttribute] {
  self.attributes.filter(attribute => attribute.object_path == path)
}

///|
fn find_writable_object(
  objects : Array[WritableObject],
  path : String,
) -> WritableObject raise Hdf5Error {
  match objects.search_by(object => object.path == path) {
    Some(index) => objects[index]
    None => raise InvalidField("writer object was not assigned an address")
  }
}

///|
fn Hdf5Writer::object_payload(
  self : Hdf5Writer,
  object : WritableObject,
  objects : Array[WritableObject],
) -> Bytes raise Hdf5Error {
  let buffer = @buffer.Buffer::Buffer()
  match object.kind {
    WritableGroup => {
      write_header_message(buffer, msg_link_info, encode_link_info())
      for child in objects {
        if child.path != "/" && writer_parent_path(child.path) == object.path {
          write_header_message(
            buffer,
            msg_link,
            encode_hard_link(writer_basename(child.path), child.address),
          )
        }
      }
    }
    WritableDatasetObject(index) => {
      let dataset = self.datasets[index]
      let encoded = encode_write_values(dataset.values)
      write_header_message(
        buffer,
        msg_dataspace,
        encode_dataspace(dataset.dimensions),
      )
      write_header_message(buffer, msg_datatype, encoded.datatype)
      match dataset.fill_value {
        Some(fill) =>
          write_header_message(buffer, msg_fill_value, encode_fill_value(fill))
        None => ()
      }
      match dataset.storage {
        WritableContiguous => {
          let address = if encoded.payload.length() == 0 {
            0xFFFFFFFFFFFFFFFFUL
          } else {
            object.raw_address
          }
          write_header_message(
            buffer,
            msg_data_layout,
            encode_contiguous_layout(address, encoded.payload.length()),
          )
        }
        WritableChunked(chunk_dimensions, filters) => {
          if filters.length() > 0 {
            write_header_message(
              buffer,
              msg_filter_pipeline,
              encode_filter_pipeline(filters, encoded.element_size),
            )
          }
          write_header_message(
            buffer,
            msg_data_layout,
            encode_chunked_layout(
              object.raw_address,
              chunk_dimensions,
              encoded.element_size,
            ),
          )
        }
      }
    }
  }
  for attribute in self.object_attributes(object.path) {
    write_header_message(buffer, msg_attribute, encode_attribute(attribute))
  }
  buffer.to_bytes()
}

///|
fn initial_writable_objects(writer : Hdf5Writer) -> Array[WritableObject] {
  let objects : Array[WritableObject] = []
  for group in writer.groups {
    objects.push({
      path: group,
      kind: WritableGroup,
      address: 0UL,
      raw_address: 0UL,
    })
  }
  for index in 0..<writer.datasets.length() {
    objects.push({
      path: writer.datasets[index].path,
      kind: WritableDatasetObject(index),
      address: 0UL,
      raw_address: 0UL,
    })
  }
  objects
}

///|
fn assign_writable_addresses(
  writer : Hdf5Writer,
  templates : Array[WritableObject],
) -> Array[WritableObject] raise Hdf5Error {
  let lengths : Array[Int32] = []
  for object in templates {
    lengths.push(
      encode_object_header(writer.object_payload(object, templates)).length(),
    )
  }
  let objects : Array[WritableObject] = []
  let mut offset = 48
  for index in 0..<templates.length() {
    let object = templates[index]
    objects.push({
      path: object.path,
      kind: object.kind,
      address: offset.to_uint64(),
      raw_address: 0UL,
    })
    offset = writer_align(offset + lengths[index])
  }
  for index in 0..<objects.length() {
    match objects[index].kind {
      WritableDatasetObject(dataset_index) => {
        let dataset = writer.datasets[dataset_index]
        let has_storage = match dataset.storage {
          WritableContiguous =>
            encode_write_values(dataset.values).payload.length() > 0
          WritableChunked(_, _) => true
        }
        if has_storage {
          objects[index] = {
            path: objects[index].path,
            kind: objects[index].kind,
            address: objects[index].address,
            raw_address: offset.to_uint64(),
          }
          offset = match dataset.storage {
            WritableContiguous =>
              writer_align(
                offset + encode_write_values(dataset.values).payload.length(),
              )
            WritableChunked(_, _) =>
              writer_align(
                offset +
                encode_fixed_array_storage(offset.to_uint64(), dataset).length(),
              )
          }
        }
      }
      WritableGroup => ()
    }
  }
  objects
}

///|
fn encode_superblock(root_address : UInt64, eof : UInt64) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=48)
  writer_write_bytes(buffer, b"\x89HDF\r\n\x1a\n")
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x08')
  buffer.write_byte(b'\x08')
  buffer.write_byte(b'\x00')
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  buffer.write_uint64_le(eof)
  buffer.write_uint64_le(root_address)
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  buffer.to_bytes()
}

///|
/// Encodes the builder as a complete HDF5 file.
pub fn Hdf5Writer::encode(self : Hdf5Writer) -> Bytes raise Hdf5Error {
  let objects = assign_writable_addresses(self, initial_writable_objects(self))
  let encoded_headers : Array[Bytes] = []
  let mut eof = 48
  for object in objects {
    let header = encode_object_header(self.object_payload(object, objects))
    encoded_headers.push(header)
    eof = writer_align(object.address.to_int() + header.length())
  }
  let raw_payloads : Array[(UInt64, Bytes)] = []
  for object in objects {
    match object.kind {
      WritableDatasetObject(index) => {
        let dataset = self.datasets[index]
        let payload = match dataset.storage {
          WritableContiguous => encode_write_values(dataset.values).payload
          WritableChunked(_, _) =>
            encode_fixed_array_storage(object.raw_address, dataset)
        }
        if payload.length() > 0 {
          raw_payloads.push((object.raw_address, payload))
          eof = writer_align(object.raw_address.to_int() + payload.length())
        }
      }
      WritableGroup => ()
    }
  }
  let output = @buffer.Buffer::Buffer(size_hint=eof)
  writer_write_bytes(
    output,
    encode_superblock(
      find_writable_object(objects, "/").address,
      eof.to_uint64(),
    ),
  )
  for index in 0..<objects.length() {
    writer_write_zeros(
      output,
      objects[index].address.to_int() - output.length(),
    )
    writer_write_bytes(output, encoded_headers[index])
  }
  for (address, payload) in raw_payloads {
    writer_write_zeros(output, address.to_int() - output.length())
    writer_write_bytes(output, payload)
  }
  writer_write_zeros(output, eof - output.length())
  output.to_bytes()
}
