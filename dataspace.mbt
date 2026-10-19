///|
pub(all) enum DataspaceKind {
  Scalar
  Simple
  Null
} derive(Eq, @debug.Debug)

///|
pub extend DataspaceKind with Eq::{not_equal, equal}

///|
pub extend DataspaceKind with @debug.Debug::{to_repr}

///|
pub(all) struct Dataspace {
  kind : DataspaceKind
  dimensions : Array[UInt64]
  maximum_dimensions : Array[UInt64]
} derive(@debug.Debug)

///|
pub extend Dataspace with @debug.Debug::{to_repr}

///|
fn parse_dataspace_data(
  data : Bytes,
  length_size : Int32,
) -> Dataspace raise Hdf5Error {
  let reader = Reader::new(data)
  let version = reader.read_u8().to_int()
  let kind = match version {
    1 => {
      let dimensionality = reader.read_u8().to_int()
      let flags = reader.read_u8().to_int()
      reader.skip(1 + 4)
      let kind : DataspaceKind = if dimensionality == 0 {
        Scalar
      } else {
        Simple
      }
      (kind, flags, dimensionality)
    }
    2 => {
      let dimensionality = reader.read_u8().to_int()
      let flags = reader.read_u8().to_int()
      let kind = match reader.read_u8().to_int() {
        0 => Scalar
        1 => Simple
        2 => Null
        _ => raise InvalidField("invalid dataspace type")
      }
      (kind, flags, dimensionality)
    }
    _ => raise InvalidField("unsupported dataspace message version")
  }
  let (kind, flags, dimensionality) = kind
  if kind == Null {
    return { kind: Null, dimensions: [], maximum_dimensions: [], }
  }
  let dimensions : Array[UInt64] = []
  for _ in 0..<dimensionality {
    dimensions.push(reader.read_uint_le(length_size))
  }
  let maximum_dimensions : Array[UInt64] = []
  if (flags & 0x01) != 0 {
    for _ in 0..<dimensionality {
      maximum_dimensions.push(reader.read_uint_le(length_size))
    }
  } else {
    for dimension in dimensions {
      maximum_dimensions.push(dimension)
    }
  }
  { kind, dimensions, maximum_dimensions, }
}

///|
pub fn parse_dataspace(
  message : HeaderMessage,
  length_size : Int32,
) -> Dataspace raise Hdf5Error {
  guard message.msg_type == msg_dataspace else {
    raise InvalidField("message is not a dataspace message")
  }
  parse_dataspace_data(message.data, length_size)
}

///|
pub fn Dataspace::element_count(self : Dataspace) -> Int64 raise Hdf5Error {
  match self.kind {
    Null => 0L
    Scalar => 1L
    Simple => {
      let mut count = 1L
      for dimension in self.dimensions {
        guard dimension <= 0x7FFFFFFFUL else {
          raise InvalidField("dataspace dimension is too large")
        }
        count = count * dimension.reinterpret_as_int64()
      }
      count
    }
  }
}
