///|
/// A selection of elements from a dataset.
pub(all) enum DatasetSelection {
  /// Selects `count` elements per axis from `start` at each `stride`.
  Hyperslab(Array[UInt64], Array[UInt64], Array[UInt64])
  /// Selects coordinates in the given order and preserves duplicates.
  Points(Array[Array[UInt64]])
} derive(@debug.Debug)

///|
pub extend DatasetSelection with @debug.Debug::{to_repr}

///|
/// Values and shape produced by a dataset selection. Hyperslabs use their
/// count as the shape; point selections use a one-dimensional point count.
pub(all) struct DatasetSelectionResult {
  dimensions : Array[UInt64]
  datatype : Datatype
  values : DatasetValues
} derive(@debug.Debug)

///|
pub extend DatasetSelectionResult with @debug.Debug::{to_repr}

///|
fn selection_element_count(dimensions : Array[UInt64]) -> Int64 raise Hdf5Error {
  { kind: Simple, dimensions, maximum_dimensions: dimensions.copy(), }.element_count()
}

///|
fn coordinate_linear_index(
  coordinate : Array[UInt64],
  dimensions : Array[UInt64],
) -> UInt64 raise Hdf5Error {
  guard coordinate.length() == dimensions.length() else {
    raise InvalidArgument(
      "selection coordinate rank does not match dataset rank",
    )
  }
  let mut linear = 0UL
  for axis in 0..<dimensions.length() {
    let dimension = dimensions[axis]
    let value = coordinate[axis]
    guard value < dimension else {
      raise InvalidArgument("selection coordinate is outside the dataset")
    }
    guard dimension == 0UL || linear <= 0xFFFFFFFFFFFFFFFFUL / dimension else {
      raise InvalidArgument("selection coordinate overflow")
    }
    linear = linear * dimension
    guard linear <= 0xFFFFFFFFFFFFFFFFUL - value else {
      raise InvalidArgument("selection coordinate overflow")
    }
    linear = linear + value
  }
  linear
}

///|
fn hyperslab_indices(
  dimensions : Array[UInt64],
  start : Array[UInt64],
  count : Array[UInt64],
  stride : Array[UInt64],
) -> (Array[UInt64], Array[UInt64]) raise Hdf5Error {
  let rank = dimensions.length()
  guard start.length() == rank &&
    count.length() == rank &&
    stride.length() == rank else {
    raise InvalidArgument("hyperslab rank does not match dataset rank")
  }
  for axis in 0..<rank {
    guard stride[axis] > 0UL else {
      raise InvalidArgument("hyperslab stride must be positive")
    }
    if count[axis] == 0UL {
      guard start[axis] <= dimensions[axis] else {
        raise InvalidArgument("hyperslab start is outside the dataset")
      }
    } else {
      guard start[axis] < dimensions[axis] else {
        raise InvalidArgument("hyperslab start is outside the dataset")
      }
      let steps = count[axis] - 1UL
      guard steps <= (0xFFFFFFFFFFFFFFFFUL - start[axis]) / stride[axis] else {
        raise InvalidArgument("hyperslab coordinate overflow")
      }
      guard start[axis] + steps * stride[axis] < dimensions[axis] else {
        raise InvalidArgument("hyperslab extends outside the dataset")
      }
    }
  }
  let total = selection_element_count(count)
  guard total <= 0x7FFFFFFFL else {
    raise InvalidArgument("selection contains too many elements")
  }
  let indices : Array[UInt64] = []
  if total == 0L {
    return (count.copy(), indices)
  }
  let positions = Array::make(rank, 0UL)
  for _ in 0..<total.to_int() {
    let coordinate = Array::make(rank, 0UL)
    for axis in 0..<rank {
      coordinate[axis] = start[axis] + positions[axis] * stride[axis]
    }
    indices.push(coordinate_linear_index(coordinate, dimensions))
    let mut axis = rank - 1
    while axis >= 0 {
      positions[axis] = positions[axis] + 1UL
      if positions[axis] < count[axis] {
        break
      }
      positions[axis] = 0UL
      axis = axis - 1
    }
  }
  (count.copy(), indices)
}

///|
fn selection_indices(
  dataspace : Dataspace,
  selection : DatasetSelection,
) -> (Array[UInt64], Array[UInt64]) raise Hdf5Error {
  match selection {
    Hyperslab(start, count, stride) =>
      hyperslab_indices(dataspace.dimensions, start, count, stride)
    Points(points) => {
      if dataspace.kind == Null && points.length() > 0 {
        raise InvalidArgument("null dataspaces have no selectable points")
      }
      let indices : Array[UInt64] = []
      for point in points {
        indices.push(coordinate_linear_index(point, dataspace.dimensions))
      }
      ([points.length().to_uint64()], indices)
    }
  }
}

///|
fn selected_payload(
  bytes : Bytes,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  indices : Array[UInt64],
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> Bytes raise Hdf5Error {
  let datatype = metadata.datatype
  guard datatype.size > 0 else { raise InvalidField("invalid datatype size") }
  if header.first(msg_external_data_files) is Some(_) {
    raise UnsupportedFeature("external dataset storage is not supported yet")
  }
  let filters = match header.first(msg_filter_pipeline) {
    Some(message) => parse_filters(message)
    None => []
  }
  if filters.length() > 0 && metadata.layout.layout_class != 2 {
    raise UnsupportedFeature("filters require chunked storage")
  }
  let full_count = metadata.dataspace.element_count()
  let layout = metadata.layout
  if layout.layout_class == 2 {
    return read_chunked_with_filters(
      bytes,
      layout,
      metadata.dataspace,
      datatype,
      offset_size,
      base_address,
      dataset_fill_value(header, datatype),
      filters,
      length_size,
      selection=Some(indices),
    )
  }
  guard full_count >= 0 && full_count <= 0x7FFFFFFFL / datatype.size.to_int64() else {
    raise InvalidField("dataset byte size is too large")
  }
  let full_size = full_count.to_int() * datatype.size
  let mut source = b""
  let mut source_offset = 0
  let mut fill = b""
  match layout.layout_class {
    0 => {
      guard layout.compact_data.length() >= full_size else {
        raise InvalidField("dataset storage is smaller than its declared shape")
      }
      source = layout.compact_data
    }
    1 => {
      let undefined = if offset_size == 8 {
        0xFFFFFFFFFFFFFFFFUL
      } else {
        (1UL << (offset_size * 8)) - 1UL
      }
      if layout.data_address == undefined {
        fill = dataset_fill_value(header, datatype)
      } else {
        let stored_size = if layout.data_size > 0UL {
          layout.data_size
        } else {
          full_size.to_uint64()
        }
        guard stored_size >= full_size.to_uint64() else {
          raise InvalidField(
            "dataset storage is smaller than its declared shape",
          )
        }
        source_offset = address_to_offset(layout.data_address, base_address)
        guard source_offset.to_int64() + full_size.to_int64() <=
          bytes.length().to_int64() else {
          raise UnexpectedEof(source_offset.to_int64())
        }
        source = bytes
      }
    }
    _ => raise UnsupportedFeature("unknown data layout class")
  }
  let output = filled_payload(indices.length().to_int64(), datatype.size, fill)
  for selected in 0..<indices.length() {
    let index = indices[selected]
    guard index < full_count.reinterpret_as_uint64() else {
      raise InvalidField("selection coordinate is outside the dataset")
    }
    let destination = selected * datatype.size
    if source.length() > 0 {
      output.blit_from_bytes(
        destination,
        source,
        source_offset + index.to_int() * datatype.size,
        datatype.size,
      )
    }
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn source_contiguous_payload(
  source : &Hdf5Source,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  indices : Array[UInt64],
  offset_size : Int32,
  base_address : UInt64,
) -> Bytes raise Hdf5Error {
  let datatype = metadata.datatype
  guard datatype.size > 0 else { raise InvalidField("invalid datatype size") }
  guard metadata.layout.layout_class == 1 else {
    raise InvalidField("expected contiguous dataset storage")
  }
  if header.first(msg_external_data_files) is Some(_) {
    raise UnsupportedFeature("external dataset storage is not supported yet")
  }
  if header.first(msg_filter_pipeline) is Some(_) {
    raise UnsupportedFeature("filters require chunked storage")
  }
  let full_count = metadata.dataspace.element_count()
  guard full_count >= 0 && full_count <= 0x7FFFFFFFL / datatype.size.to_int64() else {
    raise InvalidField("dataset byte size is too large")
  }
  let full_size = full_count.to_int() * datatype.size
  let layout = metadata.layout
  let undefined = if offset_size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (offset_size * 8)) - 1UL
  }
  let fill = if layout.data_address == undefined {
    dataset_fill_value(header, datatype)
  } else {
    b""
  }
  let output = filled_payload(indices.length().to_int64(), datatype.size, fill)
  if layout.data_address == undefined {
    return output.unsafe_reinterpret_as_bytes()
  }
  let stored_size = if layout.data_size > 0UL {
    layout.data_size
  } else {
    full_size.to_uint64()
  }
  guard stored_size >= full_size.to_uint64() else {
    raise InvalidField("dataset storage is smaller than its declared shape")
  }
  let source_offset = source_address_to_offset(
    layout.data_address,
    base_address,
  )
  let mut selected = 0
  while selected < indices.length() {
    let index = indices[selected]
    guard index < full_count.reinterpret_as_uint64() else {
      raise InvalidField("selection coordinate is outside the dataset")
    }
    let mut run_elements = 1
    while selected + run_elements < indices.length() &&
          indices[selected + run_elements] == index + run_elements.to_uint64() {
      run_elements = run_elements + 1
    }
    let relative = index * datatype.size.to_uint64()
    let byte_length = run_elements * datatype.size
    let data = source.read_at(source_offset + relative, byte_length)
    output.blit_from_bytes(selected * datatype.size, data, 0, byte_length)
    selected = selected + run_elements
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn read_source_dataset_selection(
  source : &Hdf5Source,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  selection : DatasetSelection,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> DatasetSelectionResult raise Hdf5Error {
  let (dimensions, indices) = selection_indices(metadata.dataspace, selection)
  let payload = match metadata.layout.layout_class {
    0 =>
      selected_payload(
        b"", header, metadata, indices, offset_size, length_size, base_address,
      )
    1 =>
      source_contiguous_payload(
        source, header, metadata, indices, offset_size, base_address,
      )
    2 => {
      let filters = match header.first(msg_filter_pipeline) {
        Some(message) => parse_filters(message)
        None => []
      }
      read_source_chunked_with_filters(
        source,
        metadata.layout,
        metadata.dataspace,
        metadata.datatype,
        offset_size,
        length_size,
        base_address,
        dataset_fill_value(header, metadata.datatype),
        filters,
        indices,
      )
    }
    _ => raise InvalidField("unsupported dataset storage layout")
  }
  let values = decode_values(
    b"",
    payload,
    indices.length().to_int64(),
    metadata.datatype,
    offset_size,
    length_size,
    base_address,
    source=Some(source),
  )
  { dimensions, datatype: metadata.datatype, values, }
}

///|
/// Reads selected values from a dataset.
fn read_dataset_selection(
  bytes : Bytes,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  selection : DatasetSelection,
  offset_size : Int32,
  length_size : Int32,
  base_address : UInt64,
) -> DatasetSelectionResult raise Hdf5Error {
  let (dimensions, indices) = selection_indices(metadata.dataspace, selection)
  let payload = selected_payload(
    bytes, header, metadata, indices, offset_size, length_size, base_address,
  )
  let values = decode_values(
    bytes,
    payload,
    indices.length().to_int64(),
    metadata.datatype,
    offset_size,
    length_size,
    base_address,
  )
  { dimensions, datatype: metadata.datatype, values, }
}
