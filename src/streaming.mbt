///|
/// One decoded tile from a dataset. `offset` is the tile origin in dataset
/// coordinates and `dimensions` is the edge-clipped tile shape.
pub(all) struct DatasetChunk {
  offset : Array[UInt64]
  dimensions : Array[UInt64]
  values : DatasetValues
} derive(@debug.Debug)

///|
pub extend DatasetChunk with @debug.Debug::{to_repr}

///|
/// Iterates over a dataset in caller-selected tiles without materializing the
/// complete dataset. Storage chunks that intersect a tile are decoded at most
/// once per `next` call.
pub struct DatasetChunkIterator {
  priv file : Hdf5File
  priv header : ObjectHeader
  priv metadata : DatasetMetadata
  priv tile_dimensions : Array[UInt64]
  priv mut next_index : Int64
  priv total_tiles : Int64
  priv resolver : Hdf5Resolver?
  priv referring_file : String
}

///|
fn dataset_tile_count(
  dataspace : Dataspace,
  tile_dimensions : Array[UInt64],
) -> Int64 raise Hdf5Error {
  guard dataspace.kind != Null || tile_dimensions.length() == 0 else {
    raise InvalidArgument("null datasets do not accept tile dimensions")
  }
  if dataspace.kind == Null {
    return 0L
  }
  let rank = dataspace.dimensions.length()
  guard tile_dimensions.length() == rank else {
    raise InvalidArgument("tile rank does not match dataset rank")
  }
  let mut tile_elements = 1L
  for axis in 0..<rank {
    let tile = tile_dimensions[axis]
    guard tile > 0UL else {
      raise InvalidArgument("tile dimensions must be positive")
    }
    guard tile <= 0x7FFFFFFFUL else {
      raise InvalidArgument("tile dimension is too large")
    }
    guard tile_elements <= 0x7FFFFFFFFFFFFFFFL / tile.reinterpret_as_int64() else {
      raise InvalidArgument("tile element count overflow")
    }
    tile_elements = tile_elements * tile.reinterpret_as_int64()
  }
  guard tile_elements <= 0x7FFFFFFFL else {
    raise InvalidArgument("tile contains too many elements")
  }
  if rank == 0 || dataspace.dimensions.contains(0UL) {
    return if rank == 0 { 1L } else { 0L }
  }
  let mut tiles = 1L
  for axis in 0..<rank {
    let dimension = dataspace.dimensions[axis]
    let tile = tile_dimensions[axis]
    let mut grid = dimension / tile
    if dimension % tile != 0UL {
      grid = grid + 1UL
    }
    guard grid <= 0x7FFFFFFFFFFFFFFFUL else {
      raise InvalidArgument("tile count overflow")
    }
    let grid_count = grid.reinterpret_as_int64()
    guard tiles <= 0x7FFFFFFFFFFFFFFFL / grid_count else {
      raise InvalidArgument("tile count overflow")
    }
    tiles = tiles * grid_count
  }
  tiles
}

///|
fn make_dataset_chunk_iterator(
  file : Hdf5File,
  header : ObjectHeader,
  metadata : DatasetMetadata,
  tile_dimensions : Array[UInt64],
  resolver : Hdf5Resolver?,
  referring_file : String,
) -> DatasetChunkIterator raise Hdf5Error {
  let total_tiles = dataset_tile_count(metadata.dataspace, tile_dimensions)
  {
    file,
    header,
    metadata,
    tile_dimensions: tile_dimensions.copy(),
    next_index: 0L,
    total_tiles,
    resolver,
    referring_file,
  }
}

///|
/// Creates a bounded-memory iterator over a dataset.
pub fn Hdf5File::chunk_iterator(
  self : Hdf5File,
  path : String,
  tile_dimensions : Array[UInt64],
) -> DatasetChunkIterator raise Hdf5Error {
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  let header = self.object_header_at(node.address)
  let metadata = read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
  make_dataset_chunk_iterator(self, header, metadata, tile_dimensions, None, "")
}

///|
fn DatasetChunkIterator::read_selection(
  self : DatasetChunkIterator,
  selection : DatasetSelection,
) -> DatasetSelectionResult raise Hdf5Error {
  if self.header.first(msg_external_data_files) is Some(_) {
    let resolver = match self.resolver {
      Some(resolver) => resolver
      None =>
        raise UnsupportedFeature(
          "external dataset storage requires Hdf5Resolver",
        )
    }
    return resolver.read_external_dataset_selection(
      self.referring_file,
      self.file,
      self.header,
      self.metadata,
      selection,
    )
  }
  let file = self.file
  match file.source {
    Some(source) =>
      read_source_dataset_selection(
        source,
        self.header,
        self.metadata,
        selection,
        file.superblock.offset_size,
        file.superblock.length_size,
        file.superblock.base_address,
      )
    None =>
      read_dataset_selection(
        file.bytes,
        self.header,
        self.metadata,
        selection,
        file.superblock.offset_size,
        file.superblock.length_size,
        file.superblock.base_address,
      )
  }
}

///|
/// Returns the next tile, or `None` after the dataset has been consumed.
/// Callers can stop early by no longer calling `next`.
pub fn DatasetChunkIterator::next(
  self : DatasetChunkIterator,
) -> DatasetChunk? raise Hdf5Error {
  if self.next_index >= self.total_tiles {
    return None
  }
  let tile_index = self.next_index
  self.next_index = self.next_index + 1L
  let rank = self.metadata.dataspace.dimensions.length()
  if rank == 0 {
    let selected = self.read_selection(Hyperslab([], [], []))
    return Some({ offset: [], dimensions: [], values: selected.values, })
  }
  let offset : Array[UInt64] = Array::make(rank, 0UL)
  let dimensions : Array[UInt64] = Array::make(rank, 0UL)
  let mut remaining = tile_index
  let mut axis = rank - 1
  while axis >= 0 {
    let dimension = self.metadata.dataspace.dimensions[axis]
    let tile = self.tile_dimensions[axis]
    let mut grid = dimension / tile
    if dimension % tile != 0UL {
      grid = grid + 1UL
    }
    let grid_count = grid.reinterpret_as_int64()
    let coordinate = remaining % grid_count
    remaining = remaining / grid_count
    let start = coordinate.reinterpret_as_uint64() * tile
    let available = dimension - start
    offset[axis] = start
    dimensions[axis] = if available < tile { available } else { tile }
    axis = axis - 1
  }
  let stride : Array[UInt64] = Array::make(rank, 1UL)
  let selected = self.read_selection(
    Hyperslab(offset.copy(), dimensions.copy(), stride),
  )
  Some({ offset, dimensions, values: selected.values, })
}
