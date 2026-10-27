///|
/// One inclusive block in a serialized HDF5 hyperslab selection.
pub(all) struct Hdf5RegionBlock {
  start : Array[UInt64]
  end : Array[UInt64]
} derive(Eq, @debug.Debug)

///|
pub extend Hdf5RegionBlock with Eq::{not_equal, equal}

///|
pub extend Hdf5RegionBlock with @debug.Debug::{to_repr}

///|
/// The dataspace selection stored in an HDF5 1.14 region reference.
pub(all) enum Hdf5RegionSelection {
  RegionEmpty
  RegionAll
  RegionPoints(Array[Array[UInt64]])
  RegionHyperslab(Array[UInt64], Array[UInt64], Array[UInt64], Array[UInt64])
  RegionBlocks(Array[Hdf5RegionBlock])
} derive(Eq, @debug.Debug)

///|
pub extend Hdf5RegionSelection with Eq::{not_equal, equal}

///|
pub extend Hdf5RegionSelection with @debug.Debug::{to_repr}

///|
/// The dataset and dataspace selection reached by a region reference.
pub(all) struct Hdf5RegionReferenceTarget {
  address : UInt64
  kind : Hdf5ObjectKind
  path : String?
  dataspace : Dataspace
  selection : Hdf5RegionSelection
} derive(@debug.Debug)

///|
pub extend Hdf5RegionReferenceTarget with @debug.Debug::{to_repr}

///|
fn validate_region_encoding_size(enc_size : Int32) -> Unit raise Hdf5Error {
  guard enc_size == 2 || enc_size == 4 || enc_size == 8 else {
    raise UnsupportedFeature(
      "region selection encoding width must be 2, 4 or 8 bytes",
    )
  }
}

///|
fn read_region_rank(reader : Reader) -> Int32 raise Hdf5Error {
  let rank = reader.read_u32_le().reinterpret_as_int()
  guard rank >= 0 && rank <= 32 else {
    raise InvalidField("region selection rank is outside 0..32")
  }
  rank
}

///|
fn read_region_vector(
  reader : Reader,
  rank : Int32,
  enc_size : Int32,
) -> Array[UInt64] raise Hdf5Error {
  let values : Array[UInt64] = []
  for _ in 0..<rank {
    values.push(reader.read_uint_le(enc_size))
  }
  values
}

///|
fn read_region_block(
  reader : Reader,
  rank : Int32,
  enc_size : Int32,
) -> Hdf5RegionBlock raise Hdf5Error {
  let start = read_region_vector(reader, rank, enc_size)
  let end = read_region_vector(reader, rank, enc_size)
  { start, end, }
}

///|
fn read_region_count(
  reader : Reader,
  enc_size : Int32,
) -> Int32 raise Hdf5Error {
  let value = reader.read_uint_le(enc_size)
  guard value <= 0x7FFFFFFFUL else {
    raise InvalidField("region selection contains too many entries")
  }
  value.to_int()
}

///|
fn parse_region_points(
  reader : Reader,
  version : Int32,
) -> Hdf5RegionSelection raise Hdf5Error {
  let enc_size = if version == 1 {
    reader.skip(8)
    4
  } else if version == 2 {
    reader.read_u8().to_int()
  } else {
    raise UnsupportedFeature("unsupported region point-selection version")
  }
  validate_region_encoding_size(enc_size)
  let rank = read_region_rank(reader)
  let point_count = read_region_count(reader, enc_size)
  let points : Array[Array[UInt64]] = []
  for _ in 0..<point_count {
    points.push(read_region_vector(reader, rank, enc_size))
  }
  RegionPoints(points)
}

///|
fn parse_region_hyperslab(
  reader : Reader,
  version : Int32,
) -> Hdf5RegionSelection raise Hdf5Error {
  if version == 1 {
    reader.skip(4)
    let _length = reader.read_u32_le()
    let rank = read_region_rank(reader)
    let block_count = reader.read_u32_le().reinterpret_as_int()
    guard block_count >= 0 else {
      raise InvalidField("region hyperslab block count is too large")
    }
    let blocks : Array[Hdf5RegionBlock] = []
    for _ in 0..<block_count {
      blocks.push(read_region_block(reader, rank, 4))
    }
    return RegionBlocks(blocks)
  }

  let flags = if version == 2 {
    let flags = reader.read_u8().to_int()
    let _length = reader.read_u32_le()
    flags
  } else if version == 3 {
    let flags = reader.read_u8().to_int()
    flags
  } else {
    raise UnsupportedFeature("unsupported region hyperslab version")
  }
  guard (flags & 0xFE) == 0 else {
    raise InvalidField("unsupported region hyperslab flags")
  }
  let enc_size = if version == 2 { 8 } else { reader.read_u8().to_int() }
  validate_region_encoding_size(enc_size)
  let rank = read_region_rank(reader)
  if (flags & 0x01) != 0 {
    let start : Array[UInt64] = []
    let stride : Array[UInt64] = []
    let count : Array[UInt64] = []
    let block : Array[UInt64] = []
    for _ in 0..<rank {
      start.push(reader.read_uint_le(enc_size))
      stride.push(reader.read_uint_le(enc_size))
      count.push(reader.read_uint_le(enc_size))
      block.push(reader.read_uint_le(enc_size))
    }
    return RegionHyperslab(start, stride, count, block)
  }
  let block_count = read_region_count(reader, enc_size)
  let blocks : Array[Hdf5RegionBlock] = []
  for _ in 0..<block_count {
    blocks.push(read_region_block(reader, rank, enc_size))
  }
  RegionBlocks(blocks)
}

///|
fn parse_region_empty_selection(
  reader : Reader,
  version : Int32,
  selection : Hdf5RegionSelection,
) -> Hdf5RegionSelection raise Hdf5Error {
  guard version == 1 else {
    raise UnsupportedFeature("unsupported region selection version")
  }
  let _padding = reader.read_u32_le()
  let length = reader.read_u32_le()
  guard length == 0U else {
    raise InvalidField("region all/none selection has trailing data")
  }
  selection
}

///|
fn validate_region_selection(
  dataspace : Dataspace,
  selection : Hdf5RegionSelection,
) -> Unit raise Hdf5Error {
  let dimensions = dataspace.dimensions
  let rank = dimensions.length()
  match selection {
    RegionEmpty | RegionAll => ()
    RegionPoints(points) =>
      for point in points {
        guard point.length() == rank else {
          raise InvalidField("region point rank does not match the dataset")
        }
        for axis in 0..<rank {
          guard point[axis] < dimensions[axis] else {
            raise InvalidField("region point is outside the dataset")
          }
        }
      }
    RegionHyperslab(start, stride, count, block) => {
      guard start.length() == rank &&
        stride.length() == rank &&
        count.length() == rank &&
        block.length() == rank else {
        raise InvalidField("region hyperslab rank does not match the dataset")
      }
      for axis in 0..<rank {
        if count[axis] == 0UL || block[axis] == 0UL {
          continue
        }
        guard stride[axis] > 0UL else {
          raise InvalidField("region hyperslab stride must be positive")
        }
        guard start[axis] < dimensions[axis] else {
          raise InvalidField("region hyperslab start is outside the dataset")
        }
        let steps = count[axis] - 1UL
        guard steps <= (0xFFFFFFFFFFFFFFFFUL - start[axis]) / stride[axis] else {
          raise InvalidField("region hyperslab coordinate overflow")
        }
        let stride_end = start[axis] + steps * stride[axis]
        guard block[axis] - 1UL <= 0xFFFFFFFFFFFFFFFFUL - stride_end else {
          raise InvalidField("region hyperslab coordinate overflow")
        }
        guard stride_end + block[axis] - 1UL < dimensions[axis] else {
          raise InvalidField("region hyperslab extends outside the dataset")
        }
      }
    }
    RegionBlocks(blocks) =>
      for block in blocks {
        guard block.start.length() == rank && block.end.length() == rank else {
          raise InvalidField("region block rank does not match the dataset")
        }
        for axis in 0..<rank {
          guard block.start[axis] <= block.end[axis] &&
            block.end[axis] < dimensions[axis] else {
            raise InvalidField("region block is outside the dataset")
          }
        }
      }
  }
}

///|
fn parse_region_selection(
  reader : Reader,
  dataspace : Dataspace,
) -> Hdf5RegionSelection raise Hdf5Error {
  let selection_type = reader.read_u32_le().reinterpret_as_int()
  let version = reader.read_u32_le().reinterpret_as_int()
  let selection = match selection_type {
    0 => parse_region_empty_selection(reader, version, RegionEmpty)
    1 => parse_region_points(reader, version)
    2 => parse_region_hyperslab(reader, version)
    3 => parse_region_empty_selection(reader, version, RegionAll)
    _ => raise UnsupportedFeature("unsupported region selection type")
  }
  guard reader.remaining() == 0 else {
    raise InvalidField("region selection has trailing bytes")
  }
  validate_region_selection(dataspace, selection)
  selection
}

///|
fn Hdf5File::read_reference_heap_object(
  self : Hdf5File,
  heap_address : UInt64,
  object_index : Int32,
) -> Bytes raise Hdf5Error {
  match self.source {
    Some(source) =>
      read_source_global_heap_object(
        source,
        heap_address,
        object_index,
        self.superblock.length_size,
        self.superblock.base_address,
      )
    None =>
      read_global_heap_object(
        self.bytes,
        heap_address,
        object_index,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
  }
}

///|
/// Resolves a legacy HDF5 dataset-region reference.
///
/// The all-zero null reference returns `None`. The target must be a dataset in
/// the same file. Region selection versions 1-3 for hyperslabs, versions 1-2
/// for points, and all/none selections are supported. Version-4 encoded
/// references are not supported.
pub fn Hdf5File::resolve_region_reference(
  self : Hdf5File,
  reference : Bytes,
) -> Hdf5RegionReferenceTarget? raise Hdf5Error {
  let offset_size = self.superblock.offset_size
  let reference_size = offset_size + 4
  guard reference.length() == reference_size else {
    raise InvalidField(
      "region reference must be " +
      reference_size.to_string() +
      " bytes, found " +
      reference.length().to_string(),
    )
  }
  if reference_is_null(reference) {
    return None
  }
  let reader = Reader::new(reference)
  let heap_address = reader.read_uint_le(offset_size)
  let object_index_value = reader.read_u32_le()
  guard heap_address != 0UL &&
    heap_address != undefined_address &&
    object_index_value > 0U &&
    object_index_value <= 0xFFFFU else {
    raise InvalidField("invalid region-reference heap identifier")
  }
  let payload = self.read_reference_heap_object(
    heap_address,
    object_index_value.reinterpret_as_int(),
  )
  guard payload.length() > offset_size else {
    raise InvalidField("region-reference heap object is too short")
  }
  let payload_reader = Reader::new(payload)
  let address = payload_reader.read_uint_le(offset_size)
  let header = self.object_header_at(address) catch {
    error =>
      raise InvalidField(
        "region-reference target is not a valid object header: " +
        error.message(),
      )
  }
  let kind = object_kind(header)
  guard kind == Dataset else {
    raise InvalidField("region-reference target is not a dataset")
  }
  let metadata = read_dataset_metadata(
    header,
    offset_size,
    self.superblock.length_size,
  )
  let selection = parse_region_selection(payload_reader, metadata.dataspace)
  let path = self.find_reference_path(address)
  Some({ address, kind, path, dataspace: metadata.dataspace, selection, })
}

///|
fn all_region_indices(dataspace : Dataspace) -> Array[UInt64] raise Hdf5Error {
  let count = dataspace.element_count()
  guard count >= 0 && count <= 0x7FFFFFFFL else {
    raise InvalidField("region selection contains too many elements")
  }
  let indices : Array[UInt64] = []
  for index in 0..<count.to_int() {
    indices.push(index.to_uint64())
  }
  indices
}

///|
fn sort_unique_region_indices(indices : Array[UInt64]) -> Array[UInt64] {
  indices.sort_by((left, right) => {
    if left < right {
      -1
    } else if left > right {
      1
    } else {
      0
    }
  })
  let unique : Array[UInt64] = []
  for index in indices {
    if unique.length() == 0 || unique[unique.length() - 1] != index {
      unique.push(index)
    }
  }
  unique
}

///|
fn regular_region_selection_indices(
  dataspace : Dataspace,
  start : Array[UInt64],
  stride : Array[UInt64],
  count : Array[UInt64],
  block : Array[UInt64],
) -> (Array[UInt64], Array[UInt64]) raise Hdf5Error {
  let rank = dataspace.dimensions.length()
  let dimensions : Array[UInt64] = []
  for axis in 0..<rank {
    guard block[axis] == 0UL ||
      count[axis] <= 0xFFFFFFFFFFFFFFFFUL / block[axis] else {
      raise InvalidField("region hyperslab shape overflow")
    }
    dimensions.push(count[axis] * block[axis])
  }
  let total = selection_element_count(dimensions)
  guard total >= 0 && total <= 0x7FFFFFFFL else {
    raise InvalidField("region selection contains too many elements")
  }
  let indices : Array[UInt64] = []
  if total == 0L {
    return (dimensions, indices)
  }
  let positions = Array::make(rank, 0UL)
  for _ in 0..<total.to_int() {
    let coordinate = Array::make(rank, 0UL)
    for axis in 0..<rank {
      coordinate[axis] = start[axis] +
        positions[axis] / block[axis] * stride[axis] +
        positions[axis] % block[axis]
    }
    indices.push(coordinate_linear_index(coordinate, dataspace.dimensions))
    let mut axis = rank - 1
    while axis >= 0 {
      positions[axis] = positions[axis] + 1UL
      if positions[axis] < dimensions[axis] {
        break
      }
      positions[axis] = 0UL
      axis = axis - 1
    }
  }
  (dimensions, sort_unique_region_indices(indices))
}

///|
fn block_region_selection_indices(
  dataspace : Dataspace,
  blocks : Array[Hdf5RegionBlock],
) -> (Array[UInt64], Array[UInt64]) raise Hdf5Error {
  let rank = dataspace.dimensions.length()
  let indices : Array[UInt64] = []
  let block_dimensions : Array[Array[UInt64]] = []
  let mut total = 0L
  for block in blocks {
    let dimensions : Array[UInt64] = []
    for axis in 0..<rank {
      dimensions.push(block.end[axis] - block.start[axis] + 1UL)
    }
    let count = selection_element_count(dimensions)
    guard count >= 0 && total <= 0x7FFFFFFFL - count else {
      raise InvalidField("region selection contains too many elements")
    }
    total = total + count
    block_dimensions.push(dimensions)
  }
  for block_index in 0..<blocks.length() {
    let block = blocks[block_index]
    let dimensions = block_dimensions[block_index]
    let positions = Array::make(rank, 0UL)
    let count = selection_element_count(dimensions).to_int()
    for _ in 0..<count {
      let coordinate = Array::make(rank, 0UL)
      for axis in 0..<rank {
        coordinate[axis] = block.start[axis] + positions[axis]
      }
      indices.push(coordinate_linear_index(coordinate, dataspace.dimensions))
      let mut axis = rank - 1
      while axis >= 0 {
        positions[axis] = positions[axis] + 1UL
        if positions[axis] < dimensions[axis] {
          break
        }
        positions[axis] = 0UL
        axis = axis - 1
      }
    }
  }
  let indices = sort_unique_region_indices(indices)
  ([indices.length().to_uint64()], indices)
}

///|
fn region_selection_indices(
  dataspace : Dataspace,
  selection : Hdf5RegionSelection,
) -> (Array[UInt64], Array[UInt64]) raise Hdf5Error {
  match selection {
    RegionEmpty => ([0UL], [])
    RegionAll => (dataspace.dimensions.copy(), all_region_indices(dataspace))
    RegionPoints(points) => {
      let indices : Array[UInt64] = []
      for point in points {
        indices.push(coordinate_linear_index(point, dataspace.dimensions))
      }
      ([points.length().to_uint64()], indices)
    }
    RegionHyperslab(start, stride, count, block) =>
      regular_region_selection_indices(dataspace, start, stride, count, block)
    RegionBlocks(blocks) => block_region_selection_indices(dataspace, blocks)
  }
}

///|
/// Reads the values selected by a legacy HDF5 dataset-region reference.
///
/// A null reference returns `None`. The result uses the selection shape for a
/// regular hyperslab and a flat element count for points or irregular blocks.
pub fn Hdf5File::read_region_reference(
  self : Hdf5File,
  reference : Bytes,
) -> DatasetSelectionResult? raise Hdf5Error {
  let target = match self.resolve_region_reference(reference) {
    Some(target) => target
    None => return None
  }
  let header = self.object_header_at(target.address)
  let metadata = read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  let (dimensions, indices) = region_selection_indices(
    metadata.dataspace,
    target.selection,
  )
  let result = match self.source {
    Some(source) =>
      read_source_dataset_indices(
        source,
        header,
        metadata,
        dimensions,
        indices,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
    None =>
      read_dataset_indices(
        self.bytes,
        header,
        metadata,
        dimensions,
        indices,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
  }
  Some(result)
}
