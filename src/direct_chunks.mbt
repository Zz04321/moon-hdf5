///|
fn read_direct_chunks(
  bytes : Bytes,
  layout : DataLayout,
  dataspace : Dataspace,
  datatype : Datatype,
  base_address : UInt64,
  filters : Array[Filter],
  output : FixedArray[Byte],
) -> Unit raise Hdf5Error {
  guard layout.element_size == datatype.size else {
    raise InvalidField("chunk element size does not match datatype")
  }
  let rank = dataspace.dimensions.length()
  let chunk_size = chunk_byte_size(layout.chunk_dimensions, datatype.size)
  let start = address_to_offset(layout.data_address, base_address)
  match layout.chunk_index {
    Single(stored_size, mask) => {
      guard dataspace.maximum_dimensions == dataspace.dimensions &&
        dataspace.dimensions ==
        layout.chunk_dimensions.map(size => size.to_uint64()) else {
        raise InvalidField("single chunk dimensions must match dataspace")
      }
      guard stored_size is Some(_) || filters.length() == 0 else {
        raise InvalidField("filtered single chunk is missing its stored size")
      }
      let decoded = read_filtered_chunk(
        bytes,
        layout.data_address,
        base_address,
        stored_size.unwrap_or(chunk_size.to_uint64()),
        filters,
        mask,
        datatype.size,
        chunk_size,
      )
      copy_chunk(
        decoded,
        decoded.length(),
        Array::make(rank, 0UL),
        layout.chunk_dimensions,
        dataspace.dimensions,
        datatype.size,
        output,
      )
    }
    Implicit => {
      guard filters.length() == 0 else {
        raise InvalidField("implicit chunks cannot have filters")
      }
      guard dataspace.maximum_dimensions.length() == rank else {
        raise InvalidField("implicit chunks require maximum dimensions")
      }
      let current_grid : Array[Int64] = []
      let maximum_grid : Array[Int64] = []
      for dimension in 0..<rank {
        let current = dataspace.dimensions[dimension]
        let maximum = dataspace.maximum_dimensions[dimension]
        guard current <= maximum && maximum <= 0x7FFFFFFFUL else {
          raise InvalidField(
            "implicit chunks require bounded maximum dimensions",
          )
        }
        let width = layout.chunk_dimensions[dimension].to_int64()
        current_grid.push((current.reinterpret_as_int64() + width - 1) / width)
        maximum_grid.push((maximum.reinterpret_as_int64() + width - 1) / width)
      }
      if output.length() == 0 {
        return
      }
      let total = product(current_grid)
      let maximum_total = product(maximum_grid)
      guard maximum_total <=
        (bytes.length() - start).to_int64() / chunk_size.to_int64() else {
        raise InvalidField("implicit chunk array extends beyond input")
      }
      for linear in 0..<total.to_int() {
        let offsets : Array[UInt64] = Array::make(rank, 0UL)
        let coordinates : Array[Int64] = Array::make(rank, 0L)
        let mut remaining = linear.to_int64()
        let mut dimension = rank
        while dimension > 0 {
          dimension = dimension - 1
          coordinates[dimension] = remaining % current_grid[dimension]
          remaining = remaining / current_grid[dimension]
          offsets[dimension] = (coordinates[dimension] *
          layout.chunk_dimensions[dimension].to_int64()).reinterpret_as_uint64()
        }
        let mut stored_index = 0L
        for dimension in 0..<rank {
          stored_index = stored_index * maximum_grid[dimension] +
            coordinates[dimension]
        }
        let position = start + stored_index.to_int() * chunk_size
        let data = Reader::at(bytes, position).read_bytes(chunk_size)
        copy_chunk(
          data,
          chunk_size,
          offsets,
          layout.chunk_dimensions,
          dataspace.dimensions,
          datatype.size,
          output,
        )
      }
    }
    BTreeV1 | FixedArray(_) | ExtensibleArray(_) | BTreeV2(_) =>
      raise InvalidField("expected direct chunk index")
  }
}
