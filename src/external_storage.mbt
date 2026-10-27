///|
priv struct ExternalDataSlot {
  name : String
  file_offset : UInt64
  data_size : UInt64
}

///|
priv struct ResolvedExternalDataSlot {
  source : &Hdf5Source
  file_offset : UInt64
  data_size : UInt64
}

///|
fn external_storage_file_name(name : String) -> String raise Hdf5Error {
  let mut value = name
  if value.has_prefix("file://localhost/") {
    value = "/" + value.replace(old="file://localhost/", new="")
  } else if value.has_prefix("file://") {
    value = value.replace(old="file://", new="")
    guard value.has_prefix("/") else {
      raise UnsupportedFeature(
        "remote external data file hosts are not supported",
      )
    }
  } else if value.has_prefix("file:") {
    value = value.replace(old="file:", new="")
  } else if value.contains("://") {
    raise UnsupportedFeature("external data URL schemes are not supported")
  }
  guard value != "" else {
    raise InvalidField("external data file name is empty")
  }
  value
}

///|
fn parse_external_data_slots(
  source : &Hdf5Source,
  header : ObjectHeader,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Array[ExternalDataSlot] raise Hdf5Error {
  let message = match header.first(msg_external_data_files) {
    Some(message) => message
    None => raise InvalidField("dataset has no external data file message")
  }
  guard (message.flags.to_int() & 0x02) == 0 else {
    raise UnsupportedFeature(
      "shared external data file messages are not supported",
    )
  }
  let reader = Reader::new(message.data)
  let version = reader.read_u8().to_int()
  guard version == 1 else {
    raise UnsupportedFeature("unsupported external data file message version")
  }
  guard reader.read_bytes(3) == b"\x00\x00\x00" else {
    raise InvalidField("external data file message reserved bytes are not zero")
  }
  let allocated = reader.read_u16_le().to_int()
  let used = reader.read_u16_le().to_int()
  guard used > 0 && used <= allocated else {
    raise InvalidField("invalid external data file slot counts")
  }
  let heap_address = reader.read_uint_le(offset_size)
  let heap = read_source_local_heap(
    source, heap_address, offset_size, length_size, base_address,
  )
  let slots : Array[ExternalDataSlot] = []
  for _ in 0..<used {
    let name_offset = reader.read_uint_le(offset_size)
    let file_offset = reader.read_uint_le(length_size)
    let data_size = reader.read_uint_le(length_size)
    guard data_size > 0UL else {
      raise InvalidField("external data file slot is empty")
    }
    let name = heap.get_string(name_offset)
    guard name != "" else {
      raise InvalidField("external data file name is empty")
    }
    slots.push({ name, file_offset, data_size, })
  }
  slots
}

///|
fn external_storage_product(
  count : Int64,
  size : Int32,
  description : String,
) -> Int32 raise Hdf5Error {
  guard size > 0 && count >= 0 && count <= 0x7FFFFFFFL / size.to_int64() else {
    raise InvalidField(description + " is too large")
  }
  count.to_int() * size
}

///|
fn external_storage_total_size(
  slots : Array[ResolvedExternalDataSlot],
) -> UInt64 raise Hdf5Error {
  let mut total = 0UL
  for slot in slots {
    guard total <= 0xFFFFFFFFFFFFFFFFUL - slot.data_size else {
      raise InvalidField("external data storage size overflow")
    }
    total = total + slot.data_size
  }
  total
}

///|
fn external_read_bytes(
  source : &Hdf5Source,
  offset : UInt64,
  length : Int32,
) -> Bytes raise Hdf5Error {
  guard length >= 0 else {
    raise InvalidField("external read length is negative")
  }
  let output = FixedArray::make(length, b'\x00')
  let source_length = source.length()
  if offset < source_length {
    let available = source_length - offset
    let count = if available < length.to_uint64() {
      available.to_int()
    } else {
      length
    }
    if count > 0 {
      let bytes = source.read_at(offset, count)
      output.blit_from_bytes(0, bytes, 0, count)
    }
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn external_copy_range(
  slots : Array[ResolvedExternalDataSlot],
  total : UInt64,
  logical_start : UInt64,
  length : Int32,
  output : FixedArray[Byte],
  destination : Int32,
) -> Unit raise Hdf5Error {
  guard length >= 0 &&
    destination >= 0 &&
    destination <= output.length() - length else {
    raise InvalidField("external data destination is out of range")
  }
  guard logical_start <= total && length.to_uint64() <= total - logical_start else {
    raise InvalidField("external data storage is smaller than the dataset")
  }
  let mut skip = logical_start
  let mut remaining = length
  let mut target = destination
  for slot in slots {
    if remaining == 0 {
      break
    }
    if skip >= slot.data_size {
      skip = skip - slot.data_size
      continue
    }
    let local_offset = skip
    let available = slot.data_size - local_offset
    let take = if available < remaining.to_uint64() {
      available.to_int()
    } else {
      remaining
    }
    guard slot.file_offset <= 0xFFFFFFFFFFFFFFFFUL - local_offset else {
      raise InvalidField("external data file offset overflow")
    }
    let bytes = external_read_bytes(
      slot.source,
      slot.file_offset + local_offset,
      take,
    )
    output.blit_from_bytes(target, bytes, 0, take)
    target = target + take
    remaining = remaining - take
    skip = 0UL
  }
  guard remaining == 0 else {
    raise InvalidField("external data storage is smaller than the dataset")
  }
}

///|
fn external_full_payload(
  slots : Array[ResolvedExternalDataSlot],
  size : Int32,
) -> Bytes raise Hdf5Error {
  guard size >= 0 else {
    raise InvalidField("external dataset size is negative")
  }
  let output = FixedArray::make(size, b'\x00')
  if size > 0 {
    let total = external_storage_total_size(slots)
    external_copy_range(slots, total, 0UL, size, output, 0)
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn external_selection_payload(
  slots : Array[ResolvedExternalDataSlot],
  datatype_size : Int32,
  indices : Array[UInt64],
  full_count : Int64,
) -> Bytes raise Hdf5Error {
  let output_size = external_storage_product(
    indices.length().to_int64(),
    datatype_size,
    "external selection size",
  )
  let output = FixedArray::make(output_size, b'\x00')
  let total = external_storage_total_size(slots)
  let max_index = full_count.reinterpret_as_uint64()
  let mut selected = 0
  while selected < indices.length() {
    let index = indices[selected]
    guard index < max_index else {
      raise InvalidField("selection coordinate is outside the dataset")
    }
    let mut run = 1
    while selected + run < indices.length() &&
          indices[selected + run] == index + run.to_uint64() {
      run = run + 1
    }
    guard index <= 0xFFFFFFFFFFFFFFFFUL / datatype_size.to_uint64() else {
      raise InvalidField("external dataset offset overflow")
    }
    let logical_start = index * datatype_size.to_uint64()
    let byte_length = external_storage_product(
      run.to_int64(),
      datatype_size,
      "external selection run",
    )
    guard logical_start <= total &&
      byte_length.to_uint64() <= total - logical_start else {
      raise InvalidField("external data storage is smaller than the dataset")
    }
    let destination = selected * datatype_size
    external_copy_range(
      slots, total, logical_start, byte_length, output, destination,
    )
    selected = selected + run
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn Hdf5Resolver::external_data_slots(
  self : Hdf5Resolver,
  referring_file : String,
  file : Hdf5File,
  header : ObjectHeader,
) -> Array[ResolvedExternalDataSlot] raise Hdf5Error {
  let source = match self.source_index(referring_file) {
    Some(index) => self.sources[index].source
    None => raise InvalidField("source is not registered: " + referring_file)
  }
  let parsed = parse_external_data_slots(
    source,
    header,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
  )
  let slots : Array[ResolvedExternalDataSlot] = []
  for slot in parsed {
    let name = external_storage_file_name(slot.name)
    let source_name = self.external_source_name(referring_file, name)
    let raw_source = match self.source_index(source_name) {
      Some(index) => self.sources[index].source
      None => raise InvalidField("source is not registered: " + source_name)
    }
    slots.push({
      source: raw_source,
      file_offset: slot.file_offset,
      data_size: slot.data_size,
    })
  }
  slots
}

///|
fn Hdf5Resolver::read_external_dataset(
  self : Hdf5Resolver,
  referring_file : String,
  file : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
) -> Dataset raise Hdf5Error {
  guard metadata.layout.layout_class == 1 else {
    raise UnsupportedFeature(
      "external data storage requires a contiguous layout",
    )
  }
  if header.first(msg_filter_pipeline) is Some(_) {
    raise UnsupportedFeature("external data storage cannot be filtered")
  }
  let slots = self.external_data_slots(referring_file, file, header)
  let count = metadata.dataspace.element_count()
  let size = external_storage_product(
    count,
    metadata.datatype.size,
    "external dataset size",
  )
  let payload = external_full_payload(slots, size)
  let source = match self.source_index(referring_file) {
    Some(index) => self.sources[index].source
    None => raise InvalidField("source is not registered: " + referring_file)
  }
  let values = decode_values(
    b"",
    payload,
    count,
    metadata.datatype,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
    source=Some(source),
  )
  {
    dataspace: metadata.dataspace,
    datatype: metadata.datatype,
    layout: metadata.layout,
    values,
  }
}

///|
fn Hdf5Resolver::read_external_dataset_selection(
  self : Hdf5Resolver,
  referring_file : String,
  file : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  selection : DatasetSelection,
) -> DatasetSelectionResult raise Hdf5Error {
  guard metadata.layout.layout_class == 1 else {
    raise UnsupportedFeature(
      "external data storage requires a contiguous layout",
    )
  }
  if header.first(msg_filter_pipeline) is Some(_) {
    raise UnsupportedFeature("external data storage cannot be filtered")
  }
  let slots = self.external_data_slots(referring_file, file, header)
  let (dimensions, indices) = selection_indices(metadata.dataspace, selection)
  let full_count = metadata.dataspace.element_count()
  let payload = external_selection_payload(
    slots,
    metadata.datatype.size,
    indices,
    full_count,
  )
  let source = match self.source_index(referring_file) {
    Some(index) => self.sources[index].source
    None => raise InvalidField("source is not registered: " + referring_file)
  }
  let values = decode_values(
    b"",
    payload,
    indices.length().to_int64(),
    metadata.datatype,
    file.superblock.offset_size,
    file.superblock.length_size,
    file.superblock.base_address,
    source=Some(source),
  )
  { dimensions, datatype: metadata.datatype, values, }
}
