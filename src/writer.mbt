///|
/// Values accepted by the HDF5 new-file writer.
pub(all) enum Hdf5WriteValues {
  Int8Values(Array[Int32])
  Int16Values(Array[Int32])
  Int32Values(Array[Int32])
  Int64Values(Array[Int64])
  UInt8Values(Array[UInt32])
  UInt16Values(Array[UInt32])
  UInt32Values(Array[UInt32])
  UInt64Values(Array[UInt64])
  Float32Values(Array[Float])
  Float64Values(Array[Double])
  FixedStringValues(Array[String], Int32)
  VariableStringValues(Array[String])
  StructuredValues(Datatype, Array[DatasetValue])
} derive(@debug.Debug)

///|
pub extend Hdf5WriteValues with @debug.Debug::{to_repr}

///|
fn writer_datatype(
  class : Int32,
  size : Int32,
  signed~ : Bool = false,
  enum_members~ : Array[EnumMember] = [],
  compound_members~ : Array[CompoundMember] = [],
  base_types~ : Array[Datatype] = [],
  array_dimensions~ : Array[UInt64] = [],
) -> Datatype {
  {
    class,
    version: 3,
    size,
    signed,
    little_endian: true,
    vlen_is_string: false,
    enum_members,
    compound_members,
    base_types,
    array_dimensions,
    bit_offset: 0,
    bit_precision: if class == 0 || class == 1 {
      size * 8
    } else {
      0
    },
    string_padding: if class == 3 {
      1
    } else {
      0
    },
    string_charset: if class == 3 {
      1
    } else {
      0
    },
    opaque_tag: "",
    reference_kind: 0,
  }
}

///|
/// Creates a standard little-endian integer datatype for structured writes.
pub fn Datatype::write_integer(
  signed : Bool,
  size : Int32,
) -> Datatype raise Hdf5Error {
  guard size == 1 || size == 2 || size == 4 || size == 8 else {
    raise InvalidField("writer integer datatype width is invalid")
  }
  writer_datatype(0, size, signed~)
}

///|
/// Creates a standard little-endian IEEE floating-point datatype.
pub fn Datatype::write_float(size : Int32) -> Datatype raise Hdf5Error {
  guard size == 4 || size == 8 else {
    raise InvalidField("writer floating-point datatype width is invalid")
  }
  writer_datatype(1, size)
}

///|
/// Creates a fixed-width UTF-8 string datatype.
pub fn Datatype::write_fixed_string(width : Int32) -> Datatype raise Hdf5Error {
  guard width > 0 && width <= 65535 else {
    raise InvalidField("fixed string width is invalid")
  }
  writer_datatype(3, width)
}

///|
/// Creates a fixed-size array datatype.
pub fn Datatype::write_array(
  base : Datatype,
  dimensions : Array[UInt64],
) -> Datatype raise Hdf5Error {
  guard dimensions.length() > 0 && dimensions.length() <= 32 else {
    raise InvalidField("writer array datatype rank is invalid")
  }
  let size = datatype_array_size(dimensions, base.size)
  writer_datatype(
    10,
    size,
    base_types=[base],
    array_dimensions=dimensions.copy(),
  )
}

///|
/// Creates an enumeration datatype with an integer base type.
pub fn Datatype::write_enum(
  base : Datatype,
  members : Array[EnumMember],
) -> Datatype raise Hdf5Error {
  guard base.class == 0 && members.length() > 0 && members.length() <= 65535 else {
    raise InvalidField("writer enum requires an integer base and members")
  }
  writer_datatype(
    8,
    base.size,
    signed=base.signed,
    enum_members=members.copy(),
    base_types=[base],
  )
}

///|
/// Creates a packed compound datatype in member order.
pub fn Datatype::write_compound(
  members : Array[(String, Datatype)],
) -> Datatype raise Hdf5Error {
  guard members.length() > 0 && members.length() <= 65535 else {
    raise InvalidField("writer compound datatype requires members")
  }
  let fields : Array[CompoundMember] = []
  let mut size = 0
  for (name, datatype) in members {
    guard size <= 0x7FFFFFFF - datatype.size else {
      raise InvalidField("writer compound datatype is too large")
    }
    fields.push({ name, offset: size, datatype, })
    size = size + datatype.size
  }
  writer_datatype(6, size, compound_members=fields)
}

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
  WritableExtensible(Array[UInt32], Array[Hdf5WriteFilter], Int32)
}

///|
priv struct WritableDataset {
  path : String
  dimensions : Array[UInt64]
  maximum_dimensions : Array[UInt64]
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
priv struct WritableGlobalHeap {
  address : UInt64
  strings : Array[String]
}

///|
priv struct WritableLayout {
  objects : Array[WritableObject]
  heap : WritableGlobalHeap
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
    Int8Values(items) => items.length()
    Int16Values(items) => items.length()
    Int32Values(items) => items.length()
    Int64Values(items) => items.length()
    UInt8Values(items) => items.length()
    UInt16Values(items) => items.length()
    UInt32Values(items) => items.length()
    UInt64Values(items) => items.length()
    Float32Values(items) => items.length()
    Float64Values(items) => items.length()
    FixedStringValues(items, _) => items.length()
    VariableStringValues(items) => items.length()
    StructuredValues(_, items) => items.length()
  }
}

///|
fn copy_write_values(values : Hdf5WriteValues) -> Hdf5WriteValues {
  match values {
    Int8Values(items) => Int8Values(items.copy())
    Int16Values(items) => Int16Values(items.copy())
    Int32Values(items) => Int32Values(items.copy())
    Int64Values(items) => Int64Values(items.copy())
    UInt8Values(items) => UInt8Values(items.copy())
    UInt16Values(items) => UInt16Values(items.copy())
    UInt32Values(items) => UInt32Values(items.copy())
    UInt64Values(items) => UInt64Values(items.copy())
    Float32Values(items) => Float32Values(items.copy())
    Float64Values(items) => Float64Values(items.copy())
    FixedStringValues(items, width) => FixedStringValues(items.copy(), width)
    VariableStringValues(items) => VariableStringValues(items.copy())
    StructuredValues(datatype, items) =>
      StructuredValues(
        copy_writer_datatype(datatype),
        items.map(copy_dataset_value),
      )
  }
}

///|
fn copy_writer_datatype(datatype : Datatype) -> Datatype {
  {
    class: datatype.class,
    version: datatype.version,
    size: datatype.size,
    signed: datatype.signed,
    little_endian: datatype.little_endian,
    vlen_is_string: datatype.vlen_is_string,
    enum_members: datatype.enum_members.map(enum_item => {
      name: enum_item.name,
      value: enum_item.value,
    }),
    compound_members: datatype.compound_members.map(field => {
      name: field.name,
      offset: field.offset,
      datatype: copy_writer_datatype(field.datatype),
    }),
    base_types: datatype.base_types.map(copy_writer_datatype),
    array_dimensions: datatype.array_dimensions.copy(),
    bit_offset: datatype.bit_offset,
    bit_precision: datatype.bit_precision,
    string_padding: datatype.string_padding,
    string_charset: datatype.string_charset,
    opaque_tag: datatype.opaque_tag,
    reference_kind: datatype.reference_kind,
  }
}

///|
fn copy_dataset_value(value : DatasetValue) -> DatasetValue {
  match value {
    Compound(fields) =>
      Compound(
        fields.map(field => {
          name: field.name,
          value: copy_dataset_value(field.value),
        }),
      )
    Elements(items) => Elements(items.map(copy_dataset_value))
    Sequence(items) => Sequence(items.map(copy_dataset_value))
    Opaque(bytes) => Opaque(bytes)
    Reference(bytes) => Reference(bytes)
    _ => value
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
fn validate_writer_byte_size(
  count : Int32,
  size : Int32,
) -> Unit raise Hdf5Error {
  guard size > 0 && count <= 0x7FFFFFFF / size else {
    raise InvalidField("writer dataset byte size is too large")
  }
}

///|
fn validate_writer_datatype_name(
  name : String,
  description : String,
) -> Unit raise Hdf5Error {
  let encoded = @utf8.encode(name)
  guard encoded.length() > 0 && encoded.length() <= 65535 else {
    raise InvalidField(description + " is empty or too long")
  }
  for byte in encoded {
    guard byte != b'\x00' else {
      raise InvalidField(description + " contains NUL")
    }
  }
}

///|
fn validate_writer_datatype(
  datatype : Datatype,
  depth : Int32,
) -> Unit raise Hdf5Error {
  guard depth < 16 && datatype.size > 0 else {
    raise InvalidField("writer datatype is invalid or too deeply nested")
  }
  match datatype.class {
    0 => {
      guard (
          datatype.size == 1 ||
          datatype.size == 2 ||
          datatype.size == 4 ||
          datatype.size == 8
        ) &&
        datatype.little_endian else {
        raise InvalidField("writer integer datatype is unsupported")
      }
    }
    1 => {
      guard (datatype.size == 4 || datatype.size == 8) && datatype.little_endian else {
        raise InvalidField("writer floating-point datatype is unsupported")
      }
    }
    3 => {
      guard datatype.size <= 65535 &&
        datatype.string_padding == 1 &&
        datatype.string_charset == 1 else {
        raise InvalidField("writer string datatype is unsupported")
      }
    }
    6 => {
      guard datatype.compound_members.length() > 0 &&
        datatype.compound_members.length() <= 65535 else {
        raise InvalidField("writer compound datatype requires members")
      }
      for index in 0..<datatype.compound_members.length() {
        let field = datatype.compound_members[index]
        validate_writer_datatype_name(field.name, "compound member name")
        validate_writer_datatype(field.datatype, depth + 1)
        guard field.offset >= 0 &&
          field.offset <= datatype.size - field.datatype.size else {
          raise InvalidField("compound member extends beyond datatype size")
        }
        for previous in 0..<index {
          guard datatype.compound_members[previous].name != field.name else {
            raise InvalidField("duplicate compound member name")
          }
          let earlier = datatype.compound_members[previous]
          guard field.offset + field.datatype.size <= earlier.offset ||
            earlier.offset + earlier.datatype.size <= field.offset else {
            raise InvalidField("compound writer members overlap")
          }
        }
      }
    }
    8 => {
      guard datatype.base_types.length() == 1 &&
        datatype.enum_members.length() > 0 &&
        datatype.enum_members.length() <= 65535 else {
        raise InvalidField("writer enum datatype is invalid")
      }
      let base = datatype.base_types[0]
      validate_writer_datatype(base, depth + 1)
      guard base.class == 0 && base.size == datatype.size else {
        raise InvalidField("writer enum base datatype is invalid")
      }
      for index in 0..<datatype.enum_members.length() {
        let enum_item = datatype.enum_members[index]
        validate_writer_datatype_name(enum_item.name, "enum member name")
        guard (base.signed && enum_item.value is EnumValue::Signed(_)) ||
          (!base.signed && enum_item.value is EnumValue::Unsigned(_)) else {
          raise InvalidField("enum member signedness does not match its base")
        }
        validate_writer_integer(
          match enum_item.value {
            EnumValue::Signed(number) => SignedInteger(number)
            EnumValue::Unsigned(number) => UnsignedInteger(number)
          },
          base,
        )
        for previous in 0..<index {
          guard datatype.enum_members[previous].name != enum_item.name &&
            datatype.enum_members[previous].value != enum_item.value else {
            raise InvalidField("duplicate enum member name or value")
          }
        }
      }
    }
    10 => {
      guard datatype.base_types.length() == 1 &&
        datatype.array_dimensions.length() > 0 &&
        datatype.array_dimensions.length() <= 32 else {
        raise InvalidField("writer array datatype is invalid")
      }
      let base = datatype.base_types[0]
      validate_writer_datatype(base, depth + 1)
      guard datatype_array_size(datatype.array_dimensions, base.size) ==
        datatype.size else {
        raise InvalidField("array datatype size does not match its base type")
      }
    }
    _ => raise InvalidField("writer structured datatype class is unsupported")
  }
}

///|
fn validate_writer_integer(
  value : DatasetValue,
  datatype : Datatype,
) -> Unit raise Hdf5Error {
  if datatype.signed {
    guard value is SignedInteger(number) else {
      raise InvalidField("structured integer value has the wrong kind")
    }
    if datatype.size < 8 {
      let bits = datatype.size * 8
      let minimum = -(1L << (bits - 1))
      let maximum = (1L << (bits - 1)) - 1L
      guard number >= minimum && number <= maximum else {
        raise InvalidField("structured signed integer is out of range")
      }
    }
  } else {
    guard value is UnsignedInteger(number) else {
      raise InvalidField("structured integer value has the wrong kind")
    }
    if datatype.size < 8 {
      guard number < 1UL << (datatype.size * 8) else {
        raise InvalidField("structured unsigned integer is out of range")
      }
    }
  }
}

///|
fn validate_structured_value(
  value : DatasetValue,
  datatype : Datatype,
  depth : Int32,
) -> Unit raise Hdf5Error {
  guard depth < 16 else {
    raise InvalidField("writer value is too deeply nested")
  }
  match datatype.class {
    0 => validate_writer_integer(value, datatype)
    1 => {
      guard value is FloatingPoint(_) else {
        raise InvalidField("structured floating-point value has the wrong kind")
      }
    }
    3 => {
      guard value is Text(text) else {
        raise InvalidField("structured string value has the wrong kind")
      }
      guard @utf8.encode(text).length() <= datatype.size else {
        raise InvalidField("structured string exceeds its width")
      }
    }
    6 => {
      guard value is Compound(fields) &&
        fields.length() == datatype.compound_members.length() else {
        raise InvalidField("compound value does not match its datatype")
      }
      for index in 0..<fields.length() {
        guard fields[index].name == datatype.compound_members[index].name else {
          raise InvalidField("compound value fields are out of order")
        }
        validate_structured_value(
          fields[index].value,
          datatype.compound_members[index].datatype,
          depth + 1,
        )
      }
    }
    8 => {
      guard value is Enumeration(name) else {
        raise InvalidField("enum value has the wrong kind")
      }
      guard datatype.enum_members.search_by(enum_item => enum_item.name == name)
        is Some(_) else {
        raise InvalidField("enum value does not match any member")
      }
    }
    10 => {
      guard value is Elements(items) else {
        raise InvalidField("array value has the wrong kind")
      }
      let base = datatype.base_types[0]
      guard items.length() == datatype.size / base.size else {
        raise InvalidField(
          "array value element count does not match its datatype",
        )
      }
      for item in items {
        validate_structured_value(item, base, depth + 1)
      }
    }
    _ => raise InvalidField("writer structured datatype class is unsupported")
  }
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
      validate_writer_byte_size(items.length(), width)
    }
    VariableStringValues(items) => {
      validate_writer_byte_size(items.length(), 16)
      for item in items {
        guard @utf8.encode(item).length() <= 0x7FFFFFFF else {
          raise InvalidField("variable string value is too large")
        }
      }
    }
    Int8Values(items) => {
      validate_writer_byte_size(items.length(), 1)
      for item in items {
        guard item >= -128 && item <= 127 else {
          raise InvalidField("Int8 writer value is out of range")
        }
      }
    }
    Int16Values(items) => {
      validate_writer_byte_size(items.length(), 2)
      for item in items {
        guard item >= -32768 && item <= 32767 else {
          raise InvalidField("Int16 writer value is out of range")
        }
      }
    }
    UInt8Values(items) => {
      validate_writer_byte_size(items.length(), 1)
      for item in items {
        guard item <= 255U else {
          raise InvalidField("UInt8 writer value is out of range")
        }
      }
    }
    UInt16Values(items) => {
      validate_writer_byte_size(items.length(), 2)
      for item in items {
        guard item <= 65535U else {
          raise InvalidField("UInt16 writer value is out of range")
        }
      }
    }
    Int64Values(items) => validate_writer_byte_size(items.length(), 8)
    UInt64Values(items) => validate_writer_byte_size(items.length(), 8)
    Float64Values(items) => validate_writer_byte_size(items.length(), 8)
    Int32Values(items) => validate_writer_byte_size(items.length(), 4)
    UInt32Values(items) => validate_writer_byte_size(items.length(), 4)
    Float32Values(items) => validate_writer_byte_size(items.length(), 4)
    StructuredValues(datatype, items) => {
      validate_writer_datatype(datatype, 0)
      validate_writer_byte_size(items.length(), datatype.size)
      for item in items {
        validate_structured_value(item, datatype, 0)
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
    maximum_dimensions: dimensions.copy(),
    values: copy_write_values(values),
    storage: WritableContiguous,
    fill_value: None,
  })
}

///|
fn same_write_datatype(
  left : Hdf5WriteValues,
  right : Hdf5WriteValues,
) -> Bool raise Hdf5Error {
  match (left, right) {
    (Int8Values(_), Int8Values(_)) => true
    (Int16Values(_), Int16Values(_)) => true
    (Int32Values(_), Int32Values(_)) => true
    (Int64Values(_), Int64Values(_)) => true
    (UInt8Values(_), UInt8Values(_)) => true
    (UInt16Values(_), UInt16Values(_)) => true
    (UInt32Values(_), UInt32Values(_)) => true
    (UInt64Values(_), UInt64Values(_)) => true
    (Float32Values(_), Float32Values(_)) => true
    (Float64Values(_), Float64Values(_)) => true
    (FixedStringValues(_, left_width), FixedStringValues(_, right_width)) =>
      left_width == right_width
    (VariableStringValues(_), VariableStringValues(_)) => true
    (StructuredValues(left_type, _), StructuredValues(right_type, _)) =>
      encode_writer_datatype(left_type) == encode_writer_datatype(right_type)
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
fn validate_writer_chunk_size(
  chunk_dimensions : Array[UInt32],
  element_size : Int32,
) -> Unit raise Hdf5Error {
  let mut chunk_bytes = element_size.to_int64()
  for dimension in chunk_dimensions {
    guard chunk_bytes <= 0x7FFFFFFFL / dimension.to_int64() else {
      raise InvalidField("writer chunk is too large")
    }
    chunk_bytes = chunk_bytes * dimension.to_int64()
  }
}

///|
fn validated_write_fill(
  values : Hdf5WriteValues,
  fill_value : Hdf5WriteValues?,
) -> Hdf5WriteValues? raise Hdf5Error {
  match fill_value {
    Some(fill) => {
      guard !(values is VariableStringValues(_)) else {
        raise UnsupportedFeature(
          "custom variable-length fill values are not supported yet",
        )
      }
      guard writer_value_count(fill) == 1 && same_write_datatype(values, fill) else {
        raise InvalidField("fill value must be one value of the dataset type")
      }
      validate_write_values([], fill)
      Some(copy_write_values(fill))
    }
    None => None
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
  validate_writer_chunk_size(
    chunk_dimensions,
    encode_write_values(values).element_size,
  )
  validate_write_filters(filters)
  let copied_fill = validated_write_fill(values, fill_value)
  self.ensure_groups(writer_parent_path(normalized))
  self.datasets.push({
    path: normalized,
    dimensions: dimensions.copy(),
    maximum_dimensions: dimensions.copy(),
    values: copy_write_values(values),
    storage: WritableChunked(chunk_dimensions.copy(), filters.copy()),
    fill_value: copied_fill,
  })
}

///|
fn Hdf5Writer::writer_dataset_index(
  self : Hdf5Writer,
  path : String,
) -> Int32 raise Hdf5Error {
  let normalized = canonical_writer_path(path)
  match self.datasets.search_by(dataset => dataset.path == normalized) {
    Some(index) => index
    None => raise InvalidField("writer dataset does not exist")
  }
}

///|
fn validate_extensible_dimensions(
  dimensions : Array[UInt64],
  maximum_dimensions : Array[UInt64?],
  chunk_dimensions : Array[UInt32],
) -> (Array[UInt64], Int32) raise Hdf5Error {
  guard dimensions.length() > 0 &&
    maximum_dimensions.length() == dimensions.length() &&
    chunk_dimensions.length() == dimensions.length() else {
    raise InvalidField("extensible dimensions must have matching nonzero ranks")
  }
  let encoded_maximum : Array[UInt64] = []
  let mut unlimited_axis = -1
  for axis in 0..<dimensions.length() {
    guard dimensions[axis] <= 0x7FFFFFFFUL &&
      chunk_dimensions[axis] > 0U &&
      chunk_dimensions[axis] <= 0x7FFFFFFFU else {
      raise InvalidField("extensible dataset dimension is invalid")
    }
    match maximum_dimensions[axis] {
      Some(maximum) => {
        guard maximum > 0UL &&
          maximum <= 0x7FFFFFFFUL &&
          dimensions[axis] <= maximum else {
          raise InvalidField("extensible dataset maximum dimension is invalid")
        }
        encoded_maximum.push(maximum)
      }
      None => {
        guard unlimited_axis == -1 else {
          raise InvalidField(
            "extensible dataset requires one unlimited dimension",
          )
        }
        unlimited_axis = axis
        encoded_maximum.push(0xFFFFFFFFFFFFFFFFUL)
      }
    }
  }
  guard unlimited_axis >= 0 else {
    raise InvalidField("extensible dataset requires one unlimited dimension")
  }
  (encoded_maximum, unlimited_axis)
}

///|
/// Adds a chunked dataset with exactly one unlimited dimension.
///
/// A `None` maximum marks the unlimited axis. Other maximum dimensions are
/// finite and reserve a stable chunk-grid width for later resizes.
pub fn Hdf5Writer::add_extensible_dataset(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  maximum_dimensions : Array[UInt64?],
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
  let (encoded_maximum, unlimited_axis) = validate_extensible_dimensions(
    dimensions, maximum_dimensions, chunk_dimensions,
  )
  validate_writer_chunk_size(
    chunk_dimensions,
    encode_write_values(values).element_size,
  )
  validate_write_filters(filters)
  let copied_fill = validated_write_fill(values, fill_value)
  let grid_dimensions = dimensions.copy()
  for axis in 0..<grid_dimensions.length() {
    if axis != unlimited_axis {
      grid_dimensions[axis] = encoded_maximum[axis]
    }
  }
  let (_, total_chunks) = chunk_grid(grid_dimensions, chunk_dimensions)
  guard total_chunks <= 128 else {
    raise InvalidField("writer extensible-array index exceeds 128 chunks")
  }
  self.ensure_groups(writer_parent_path(normalized))
  self.datasets.push({
    path: normalized,
    dimensions: dimensions.copy(),
    maximum_dimensions: encoded_maximum,
    values: copy_write_values(values),
    storage: WritableExtensible(
      chunk_dimensions.copy(),
      filters.copy(),
      unlimited_axis,
    ),
    fill_value: copied_fill,
  })
}

///|
fn default_structured_value(
  datatype : Datatype,
) -> DatasetValue raise Hdf5Error {
  match datatype.class {
    0 => if datatype.signed { SignedInteger(0L) } else { UnsignedInteger(0UL) }
    1 => FloatingPoint(0.0)
    3 => Text("")
    6 =>
      Compound(
        datatype.compound_members.map(field => {
          name: field.name,
          value: default_structured_value(field.datatype),
        }),
      )
    8 => {
      let zero = datatype.enum_members.search_by(enum_item => {
        enum_item.value ==
        (if datatype.signed {
          EnumValue::Signed(0L)
        } else {
          EnumValue::Unsigned(0UL)
        })
      })
      guard zero is Some(index) else {
        raise InvalidField("enum datatype has no zero-valued default fill")
      }
      Enumeration(datatype.enum_members[index].name)
    }
    10 => {
      let base = datatype.base_types[0]
      Elements(
        Array::make(datatype.size / base.size, default_structured_value(base)),
      )
    }
    _ => raise InvalidField("writer structured datatype class is unsupported")
  }
}

///|
fn default_write_fill(
  values : Hdf5WriteValues,
) -> Hdf5WriteValues raise Hdf5Error {
  match values {
    Int8Values(_) => Int8Values([0])
    Int16Values(_) => Int16Values([0])
    Int32Values(_) => Int32Values([0])
    Int64Values(_) => Int64Values([0L])
    UInt8Values(_) => UInt8Values([0U])
    UInt16Values(_) => UInt16Values([0U])
    UInt32Values(_) => UInt32Values([0U])
    UInt64Values(_) => UInt64Values([0UL])
    Float32Values(_) => Float32Values([0.0])
    Float64Values(_) => Float64Values([0.0])
    FixedStringValues(_, width) => FixedStringValues([""], width)
    VariableStringValues(_) => VariableStringValues([""])
    StructuredValues(datatype, _) =>
      StructuredValues(datatype, [default_structured_value(datatype)])
  }
}

///|
fn writer_linear_destination(
  source_index : Int32,
  source_dimensions : Array[UInt64],
  destination_dimensions : Array[UInt64],
  axis_offset : Int32,
  offset : UInt64,
) -> Int32? {
  let rank = source_dimensions.length()
  let coordinates = Array::make(rank, 0UL)
  let mut remaining = source_index.to_uint64()
  let mut axis = rank
  while axis > 0 {
    axis = axis - 1
    let width = source_dimensions[axis]
    if width == 0UL {
      return None
    }
    coordinates[axis] = remaining % width
    remaining = remaining / width
  }
  let mut destination = 0UL
  for dimension in 0..<rank {
    let coordinate = coordinates[dimension] +
      (if dimension == axis_offset { offset } else { 0UL })
    if coordinate >= destination_dimensions[dimension] {
      return None
    }
    destination = destination * destination_dimensions[dimension] + coordinate
  }
  Some(destination.to_int())
}

///|
fn repeated_write_values(
  template : Hdf5WriteValues,
  fill : Hdf5WriteValues,
  count : Int32,
) -> Hdf5WriteValues raise Hdf5Error {
  match (template, fill) {
    (Int8Values(_), Int8Values(items)) =>
      Int8Values(Array::make(count, items[0]))
    (Int16Values(_), Int16Values(items)) =>
      Int16Values(Array::make(count, items[0]))
    (Int32Values(_), Int32Values(items)) =>
      Int32Values(Array::make(count, items[0]))
    (Int64Values(_), Int64Values(items)) =>
      Int64Values(Array::make(count, items[0]))
    (UInt8Values(_), UInt8Values(items)) =>
      UInt8Values(Array::make(count, items[0]))
    (UInt16Values(_), UInt16Values(items)) =>
      UInt16Values(Array::make(count, items[0]))
    (UInt32Values(_), UInt32Values(items)) =>
      UInt32Values(Array::make(count, items[0]))
    (UInt64Values(_), UInt64Values(items)) =>
      UInt64Values(Array::make(count, items[0]))
    (Float32Values(_), Float32Values(items)) =>
      Float32Values(Array::make(count, items[0]))
    (Float64Values(_), Float64Values(items)) =>
      Float64Values(Array::make(count, items[0]))
    (FixedStringValues(_, width), FixedStringValues(items, _)) =>
      FixedStringValues(Array::make(count, items[0]), width)
    (VariableStringValues(_), VariableStringValues(items)) =>
      VariableStringValues(Array::make(count, items[0]))
    (StructuredValues(datatype, _), StructuredValues(_, items)) =>
      StructuredValues(datatype, Array::make(count, items[0]))
    _ => raise InvalidField("fill value must be one value of the dataset type")
  }
}

///|
/// Changes an extensible dataset shape while preserving its overlapping data.
pub fn Hdf5Writer::resize_dataset(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
) -> Unit raise Hdf5Error {
  let index = self.writer_dataset_index(path)
  let dataset = self.datasets[index]
  let (chunk_dimensions, filters, unlimited_axis) = match dataset.storage {
    WritableExtensible(chunks, pipeline, axis) => (chunks, pipeline, axis)
    _ => raise InvalidField("only extensible datasets can be resized")
  }
  guard dimensions.length() == dataset.dimensions.length() else {
    raise InvalidField("resized dataset rank does not match")
  }
  for axis in 0..<dimensions.length() {
    guard dimensions[axis] <= 0x7FFFFFFFUL &&
      (
        dataset.maximum_dimensions[axis] == 0xFFFFFFFFFFFFFFFFUL ||
        dimensions[axis] <= dataset.maximum_dimensions[axis]
      ) else {
      raise InvalidField("resized dataset exceeds its maximum dimensions")
    }
  }
  let grid_dimensions = dimensions.copy()
  for axis in 0..<grid_dimensions.length() {
    if axis != unlimited_axis {
      grid_dimensions[axis] = dataset.maximum_dimensions[axis]
    }
  }
  let (_, total_chunks) = chunk_grid(grid_dimensions, chunk_dimensions)
  guard total_chunks <= 128 else {
    raise InvalidField("writer extensible-array index exceeds 128 chunks")
  }
  let fill = match dataset.fill_value {
    Some(value) => value
    None => default_write_fill(dataset.values)
  }
  let values = overlay_write_values(
    repeated_write_values(dataset.values, fill, writer_shape_count(dimensions)),
    dataset.values,
    dataset.dimensions,
    dimensions,
    -1,
    0UL,
  )
  self.datasets[index] = {
    path: dataset.path,
    dimensions: dimensions.copy(),
    maximum_dimensions: dataset.maximum_dimensions,
    values,
    storage: WritableExtensible(chunk_dimensions, filters, unlimited_axis),
    fill_value: dataset.fill_value,
  }
}

///|
fn[T] overlay_array(
  output : Array[T],
  items : Array[T],
  source_dimensions : Array[UInt64],
  destination_dimensions : Array[UInt64],
  axis : Int32,
  offset : UInt64,
) -> Array[T] raise Hdf5Error {
  for index in 0..<items.length() {
    match
      writer_linear_destination(
        index, source_dimensions, destination_dimensions, axis, offset,
      ) {
      Some(destination) => output[destination] = items[index]
      None =>
        if axis >= 0 {
          raise InvalidField("appended value is outside the destination shape")
        }
    }
  }
  output
}

///|
fn overlay_write_values(
  target : Hdf5WriteValues,
  source : Hdf5WriteValues,
  source_dimensions : Array[UInt64],
  destination_dimensions : Array[UInt64],
  axis : Int32,
  offset : UInt64,
) -> Hdf5WriteValues raise Hdf5Error {
  match (target, source) {
    (Int8Values(output), Int8Values(items)) =>
      Int8Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (Int16Values(output), Int16Values(items)) =>
      Int16Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (Int32Values(output), Int32Values(items)) =>
      Int32Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (Int64Values(output), Int64Values(items)) =>
      Int64Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (UInt8Values(output), UInt8Values(items)) =>
      UInt8Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (UInt16Values(output), UInt16Values(items)) =>
      UInt16Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (UInt32Values(output), UInt32Values(items)) =>
      UInt32Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (UInt64Values(output), UInt64Values(items)) =>
      UInt64Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (Float32Values(output), Float32Values(items)) =>
      Float32Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (Float64Values(output), Float64Values(items)) =>
      Float64Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (StructuredValues(datatype, output), StructuredValues(_, items)) =>
      StructuredValues(
        datatype,
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (FixedStringValues(output, width), FixedStringValues(items, source_width)) => {
      guard width == source_width else {
        raise InvalidField("appended values must match the dataset type")
      }
      FixedStringValues(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
        width,
      )
    }
    (VariableStringValues(output), VariableStringValues(items)) =>
      VariableStringValues(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    _ => raise InvalidField("appended values must match the dataset type")
  }
}

///|
/// Appends a block along the dataset's unlimited dimension.
pub fn Hdf5Writer::append_dataset(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
) -> Unit raise Hdf5Error {
  let index = self.writer_dataset_index(path)
  let dataset = self.datasets[index]
  let unlimited_axis = match dataset.storage {
    WritableExtensible(_, _, axis) => axis
    _ => raise InvalidField("only extensible datasets can be appended")
  }
  guard dimensions.length() == dataset.dimensions.length() else {
    raise InvalidField("appended dataset rank does not match")
  }
  validate_write_values(dimensions, values)
  guard same_write_datatype(dataset.values, values) else {
    raise InvalidField("appended values must match the dataset type")
  }
  for axis in 0..<dimensions.length() {
    if axis != unlimited_axis && dimensions[axis] != dataset.dimensions[axis] {
      raise InvalidField("appended block must match fixed dataset dimensions")
    }
  }
  guard dataset.dimensions[unlimited_axis] <=
    0x7FFFFFFFUL - dimensions[unlimited_axis] else {
    raise InvalidField("appended dataset dimension is too large")
  }
  let old_dimensions = dataset.dimensions.copy()
  let new_dimensions = old_dimensions.copy()
  new_dimensions[unlimited_axis] = new_dimensions[unlimited_axis] +
    dimensions[unlimited_axis]
  self.resize_dataset(path, new_dimensions)
  let resized = self.datasets[index]
  self.datasets[index] = {
    path: resized.path,
    dimensions: resized.dimensions,
    maximum_dimensions: resized.maximum_dimensions,
    values: overlay_write_values(
      resized.values,
      values,
      dimensions,
      new_dimensions,
      unlimited_axis,
      old_dimensions[unlimited_axis],
    ),
    storage: resized.storage,
    fill_value: resized.fill_value,
  }
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
fn write_uint_width(
  buffer : @buffer.Buffer,
  value : UInt64,
  size : Int32,
) -> Unit {
  for shift in 0..<size {
    buffer.write_byte(((value >> (shift * 8)) & 0xFFUL).to_byte())
  }
}

///|
fn encode_integer_datatype(signed : Bool, size : Int32) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=12)
  buffer.write_byte(b'\x10')
  buffer.write_byte(if signed { b'\x08' } else { b'\x00' })
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(size.reinterpret_as_uint())
  buffer.write_uint16_le(0)
  buffer.write_uint16_le((size * 8).to_uint16())
  buffer.to_bytes()
}

///|
fn encode_float_datatype(size : Int32) -> Bytes {
  let single = size == 4
  let buffer = @buffer.Buffer::Buffer(size_hint=20)
  buffer.write_byte(b'\x11')
  buffer.write_byte(b'\x20')
  buffer.write_byte(if single { b'\x1f' } else { b'\x3f' })
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(size.reinterpret_as_uint())
  buffer.write_uint16_le(0)
  buffer.write_uint16_le((size * 8).to_uint16())
  buffer.write_byte(if single { b'\x17' } else { b'\x34' })
  buffer.write_byte(if single { b'\x08' } else { b'\x0b' })
  buffer.write_byte(b'\x00')
  buffer.write_byte(if single { b'\x17' } else { b'\x34' })
  buffer.write_uint_le(if single { 127U } else { 1023U })
  buffer.to_bytes()
}

///|
fn encode_writer_datatype(datatype : Datatype) -> Bytes raise Hdf5Error {
  validate_writer_datatype(datatype, 0)
  match datatype.class {
    0 => encode_integer_datatype(datatype.signed, datatype.size)
    1 => encode_float_datatype(datatype.size)
    3 => encode_string_datatype(datatype.size)
    6 => {
      let buffer = @buffer.Buffer::Buffer()
      buffer.write_byte(b'\x36')
      buffer.write_byte((datatype.compound_members.length() & 0xFF).to_byte())
      buffer.write_byte((datatype.compound_members.length() >> 8).to_byte())
      buffer.write_byte(b'\x00')
      buffer.write_uint_le(datatype.size.reinterpret_as_uint())
      let offset_width = datatype_offset_width(datatype.size)
      for field in datatype.compound_members {
        writer_write_bytes(buffer, @utf8.encode(field.name))
        buffer.write_byte(b'\x00')
        write_uint_width(buffer, field.offset.to_uint64(), offset_width)
        writer_write_bytes(buffer, encode_writer_datatype(field.datatype))
      }
      buffer.to_bytes()
    }
    8 => {
      let buffer = @buffer.Buffer::Buffer()
      buffer.write_byte(b'\x38')
      buffer.write_byte((datatype.enum_members.length() & 0xFF).to_byte())
      buffer.write_byte((datatype.enum_members.length() >> 8).to_byte())
      buffer.write_byte(b'\x00')
      buffer.write_uint_le(datatype.size.reinterpret_as_uint())
      writer_write_bytes(buffer, encode_writer_datatype(datatype.base_types[0]))
      for enum_item in datatype.enum_members {
        writer_write_bytes(buffer, @utf8.encode(enum_item.name))
        buffer.write_byte(b'\x00')
      }
      for enum_item in datatype.enum_members {
        let raw = match enum_item.value {
          EnumValue::Signed(value) => value.reinterpret_as_uint64()
          EnumValue::Unsigned(value) => value
        }
        write_uint_width(buffer, raw, datatype.size)
      }
      buffer.to_bytes()
    }
    10 => {
      let buffer = @buffer.Buffer::Buffer()
      buffer.write_byte(b'\x3a')
      buffer.write_byte(b'\x00')
      buffer.write_byte(b'\x00')
      buffer.write_byte(b'\x00')
      buffer.write_uint_le(datatype.size.reinterpret_as_uint())
      buffer.write_byte(datatype.array_dimensions.length().to_byte())
      for dimension in datatype.array_dimensions {
        buffer.write_uint_le(dimension.to_uint())
      }
      writer_write_bytes(buffer, encode_writer_datatype(datatype.base_types[0]))
      buffer.to_bytes()
    }
    _ => raise InvalidField("writer structured datatype class is unsupported")
  }
}

///|
fn encode_structured_value(
  datatype : Datatype,
  value : DatasetValue,
) -> Bytes raise Hdf5Error {
  validate_structured_value(value, datatype, 0)
  let buffer = @buffer.Buffer::Buffer(size_hint=datatype.size)
  match datatype.class {
    0 => {
      let raw = match value {
        SignedInteger(number) => number.reinterpret_as_uint64()
        UnsignedInteger(number) => number
        _ => 0UL
      }
      write_uint_width(buffer, raw, datatype.size)
    }
    1 =>
      match value {
        FloatingPoint(number) =>
          if datatype.size == 4 {
            buffer.write_float_le(Float::from_double(number))
          } else {
            buffer.write_double_le(number)
          }
        _ => ()
      }
    3 =>
      match value {
        Text(text) => {
          let encoded = @utf8.encode(text)
          writer_write_bytes(buffer, encoded)
          writer_write_zeros(buffer, datatype.size - encoded.length())
        }
        _ => ()
      }
    6 =>
      match value {
        Compound(fields) => {
          let output = FixedArray::make(datatype.size, b'\x00')
          for index in 0..<fields.length() {
            let field = datatype.compound_members[index]
            let encoded = encode_structured_value(
              field.datatype,
              fields[index].value,
            )
            output.blit_from_bytes(field.offset, encoded, 0, encoded.length())
          }
          writer_write_bytes(buffer, output.unsafe_reinterpret_as_bytes())
        }
        _ => ()
      }
    8 =>
      match value {
        Enumeration(name) => {
          let index = datatype.enum_members
            .search_by(enum_item => enum_item.name == name)
            .unwrap()
          let raw = match datatype.enum_members[index].value {
            EnumValue::Signed(number) => number.reinterpret_as_uint64()
            EnumValue::Unsigned(number) => number
          }
          write_uint_width(buffer, raw, datatype.size)
        }
        _ => ()
      }
    10 =>
      match value {
        Elements(items) =>
          for item in items {
            writer_write_bytes(
              buffer,
              encode_structured_value(datatype.base_types[0], item),
            )
          }
        _ => ()
      }
    _ => raise InvalidField("writer structured datatype class is unsupported")
  }
  guard buffer.length() == datatype.size else {
    raise InvalidField("structured value size does not match its datatype")
  }
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
fn encode_variable_string_datatype() -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=16)
  buffer.write_byte(b'\x39')
  buffer.write_byte(b'\x01')
  buffer.write_byte(b'\x01')
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(16U)
  writer_write_bytes(buffer, encode_string_datatype(1))
  buffer.to_bytes()
}

///|
fn empty_writable_heap() -> WritableGlobalHeap {
  { address: 0UL, strings: [], }
}

///|
fn writable_heap_index(
  heap : WritableGlobalHeap,
  value : String,
) -> Int32 raise Hdf5Error {
  match heap.strings.search_by(item => item == value) {
    Some(index) => index + 1
    None => {
      guard heap.address == 0UL else {
        raise InvalidField("variable string is missing from the global heap")
      }
      0
    }
  }
}

///|
fn encode_write_values(
  values : Hdf5WriteValues,
  heap~ : WritableGlobalHeap = empty_writable_heap(),
) -> EncodedWriteValues raise Hdf5Error {
  let payload = @buffer.Buffer::Buffer()
  let (datatype, element_size) = match values {
    Int8Values(items) => {
      for item in items {
        payload.write_byte(item.to_byte())
      }
      (encode_integer_datatype(true, 1), 1)
    }
    Int16Values(items) => {
      for item in items {
        payload.write_int16_le(Int16::from_int(item))
      }
      (encode_integer_datatype(true, 2), 2)
    }
    Int32Values(items) => {
      for item in items {
        payload.write_int_le(item)
      }
      (encode_integer_datatype(true, 4), 4)
    }
    Int64Values(items) => {
      for item in items {
        payload.write_int64_le(item)
      }
      (encode_integer_datatype(true, 8), 8)
    }
    UInt8Values(items) => {
      for item in items {
        payload.write_byte(item.to_byte())
      }
      (encode_integer_datatype(false, 1), 1)
    }
    UInt16Values(items) => {
      for item in items {
        payload.write_uint16_le(item.to_uint16())
      }
      (encode_integer_datatype(false, 2), 2)
    }
    UInt32Values(items) => {
      for item in items {
        payload.write_uint_le(item)
      }
      (encode_integer_datatype(false, 4), 4)
    }
    UInt64Values(items) => {
      for item in items {
        payload.write_uint64_le(item)
      }
      (encode_integer_datatype(false, 8), 8)
    }
    Float32Values(items) => {
      for item in items {
        payload.write_float_le(item)
      }
      (encode_float_datatype(4), 4)
    }
    Float64Values(items) => {
      for item in items {
        payload.write_double_le(item)
      }
      (encode_float_datatype(8), 8)
    }
    FixedStringValues(items, width) => {
      for item in items {
        let encoded = @utf8.encode(item)
        writer_write_bytes(payload, encoded)
        writer_write_zeros(payload, width - encoded.length())
      }
      (encode_string_datatype(width), width)
    }
    VariableStringValues(items) => {
      for item in items {
        let encoded = @utf8.encode(item)
        payload.write_uint_le(encoded.length().reinterpret_as_uint())
        if encoded.length() == 0 {
          payload.write_uint64_le(0UL)
          payload.write_uint_le(0U)
        } else {
          payload.write_uint64_le(heap.address)
          payload.write_uint_le(
            writable_heap_index(heap, item).reinterpret_as_uint(),
          )
        }
      }
      (encode_variable_string_datatype(), 16)
    }
    StructuredValues(datatype, items) => {
      for item in items {
        writer_write_bytes(payload, encode_structured_value(datatype, item))
      }
      (encode_writer_datatype(datatype), datatype.size)
    }
  }
  { datatype, payload: payload.to_bytes(), element_size, }
}

///|
fn encode_dataspace(
  dimensions : Array[UInt64],
  maximum_dimensions? : Array[UInt64],
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(
    size_hint=4 +
      dimensions.length() * (if maximum_dimensions is Some(_) { 16 } else { 8 }),
  )
  buffer.write_byte(b'\x02')
  buffer.write_byte(dimensions.length().to_byte())
  buffer.write_byte(
    if maximum_dimensions is Some(_) {
      b'\x01'
    } else {
      b'\x00'
    },
  )
  buffer.write_byte(if dimensions.length() == 0 { b'\x00' } else { b'\x01' })
  for dimension in dimensions {
    buffer.write_uint64_le(dimension)
  }
  if maximum_dimensions is Some(maximum) {
    for dimension in maximum {
      buffer.write_uint64_le(dimension)
    }
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
  extensible~ : Bool = false,
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
  buffer.write_byte(if extensible { b'\x04' } else { b'\x03' })
  if extensible {
    buffer.write_byte(b'\x20')
    buffer.write_byte(b'\x80')
    buffer.write_byte(b'\x04')
    buffer.write_byte(b'\x10')
    buffer.write_byte(b'\x0a')
  } else {
    buffer.write_byte(b'\x0a')
  }
  buffer.write_uint64_le(address)
  buffer.to_bytes()
}

///|
fn encode_fill_value(fill : Hdf5WriteValues) -> Bytes raise Hdf5Error {
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
    guard count <= 0x7FFFFFFFUL else {
      raise InvalidField("writer chunk grid is too large")
    }
    grid.push(count.to_int())
    if count == 0UL {
      total = 0L
    } else {
      guard total <= 0x7FFFFFFFL / count.reinterpret_as_int64() else {
        raise InvalidField("writer chunk grid is too large")
      }
      total = total * count.reinterpret_as_int64()
    }
  }
  (grid, total.to_int())
}

///|
fn encode_dataset_chunks(
  dataset : WritableDataset,
  heap : WritableGlobalHeap,
) -> Array[EncodedChunkEntry] raise Hdf5Error {
  let (chunk_dimensions, filters, order, grid_dimensions) = match
    dataset.storage {
    WritableChunked(dimensions, filters) => {
      let order = Array::make(dimensions.length(), 0)
      for axis in 0..<order.length() {
        order[axis] = axis
      }
      (dimensions, filters, order, dataset.dimensions)
    }
    WritableExtensible(dimensions, filters, unlimited_axis) => {
      let order = [unlimited_axis]
      let grid_dimensions = dataset.dimensions.copy()
      for axis in 0..<dimensions.length() {
        if axis != unlimited_axis {
          order.push(axis)
          grid_dimensions[axis] = dataset.maximum_dimensions[axis]
        }
      }
      (dimensions, filters, order, grid_dimensions)
    }
    WritableContiguous => raise InvalidField("expected chunked writer dataset")
  }
  let encoded = encode_write_values(dataset.values, heap~)
  let fill = match dataset.fill_value {
    Some(value) => encode_write_values(value).payload
    None => Bytes::make(encoded.element_size, b'\x00')
  }
  let (grid, chunk_count) = chunk_grid(grid_dimensions, chunk_dimensions)
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
    for order_index in 0..<rank {
      let dimension = order[order_index]
      let width = chunk_dimensions[dimension].reinterpret_as_int()
      chunk_index = chunk_index * grid[dimension] +
        coordinates[dimension] / width
    }
    for dimension in 0..<rank {
      let width = chunk_dimensions[dimension].reinterpret_as_int()
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
  heap : WritableGlobalHeap,
) -> Bytes raise Hdf5Error {
  let (_, filters) = match dataset.storage {
    WritableChunked(dimensions, filters) => (dimensions, filters)
    WritableExtensible(dimensions, filters, _) => (dimensions, filters)
    WritableContiguous => raise InvalidField("expected chunked writer dataset")
  }
  let entries = encode_dataset_chunks(dataset, heap)
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
fn encode_extensible_array_storage(
  start : UInt64,
  dataset : WritableDataset,
  heap : WritableGlobalHeap,
) -> Bytes raise Hdf5Error {
  let (_, filters, _) = match dataset.storage {
    WritableExtensible(dimensions, filters, axis) => (dimensions, filters, axis)
    _ => raise InvalidField("expected extensible writer dataset")
  }
  let entries = encode_dataset_chunks(dataset, heap)
  guard entries.length() <= 128 else {
    raise InvalidField("writer extensible-array index exceeds 128 chunks")
  }
  let filtered = filters.length() > 0
  let entry_size = if filtered { 14 } else { 8 }
  let header_length = 72
  let index_address = start + header_length.to_uint64()
  let pointer_count = 31
  let index_length = 14 + 128 * entry_size + pointer_count * 8 + 4
  let mut chunk_address = writer_align(index_address.to_int() + index_length)
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
  writer_write_bytes(header, b"EAHD")
  header.write_byte(b'\x00')
  header.write_byte(if filtered { b'\x01' } else { b'\x00' })
  header.write_byte(entry_size.to_byte())
  header.write_byte(b'\x20')
  header.write_byte(b'\x80')
  header.write_byte(b'\x10')
  header.write_byte(b'\x04')
  header.write_byte(b'\x0a')
  for _ in 0..<4 {
    header.write_uint64_le(0UL)
  }
  header.write_uint64_le(entries.length().to_uint64())
  header.write_uint64_le(128UL)
  header.write_uint64_le(index_address)
  header.write_uint_le(metadata_checksum(header.to_bytes(), 0, header.length()))

  let index = @buffer.Buffer::Buffer(size_hint=index_length)
  writer_write_bytes(index, b"EAIB")
  index.write_byte(b'\x00')
  index.write_byte(if filtered { b'\x01' } else { b'\x00' })
  index.write_uint64_le(start)
  for entry_index in 0..<128 {
    let allocated = entry_index < entries.length() &&
      entries[entry_index].allocated
    index.write_uint64_le(
      if allocated {
        entries[entry_index].address
      } else {
        0xFFFFFFFFFFFFFFFFUL
      },
    )
    if filtered {
      let size = if allocated { entries[entry_index].data.length() } else { 0 }
      guard size <= 65535 else {
        raise InvalidField("filtered writer chunk is too large")
      }
      index.write_uint16_le(size.to_uint16())
      index.write_uint_le(0U)
    }
  }
  for _ in 0..<pointer_count {
    index.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  }
  index.write_uint_le(metadata_checksum(index.to_bytes(), 0, index.length()))

  let output = @buffer.Buffer::Buffer(size_hint=chunk_address - start.to_int())
  writer_write_bytes(output, header.to_bytes())
  writer_write_bytes(output, index.to_bytes())
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
fn encode_attribute(
  attribute : WritableAttribute,
  heap : WritableGlobalHeap,
) -> Bytes raise Hdf5Error {
  let encoded = encode_write_values(attribute.values, heap~)
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
  heap : WritableGlobalHeap,
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
      let encoded = encode_write_values(dataset.values, heap~)
      write_header_message(
        buffer,
        msg_dataspace,
        match dataset.storage {
          WritableExtensible(_, _, _) =>
            encode_dataspace(
              dataset.dimensions,
              maximum_dimensions=dataset.maximum_dimensions,
            )
          _ => encode_dataspace(dataset.dimensions)
        },
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
        WritableExtensible(chunk_dimensions, filters, _) => {
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
              extensible=true,
            ),
          )
        }
      }
    }
  }
  for attribute in self.object_attributes(object.path) {
    write_header_message(
      buffer,
      msg_attribute,
      encode_attribute(attribute, heap),
    )
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
fn collect_heap_strings(
  output : Array[String],
  values : Hdf5WriteValues,
) -> Unit raise Hdf5Error {
  match values {
    VariableStringValues(items) =>
      for item in items {
        if item != "" && !output.contains(item) {
          guard output.length() < 65535 else {
            raise InvalidField("writer global heap has too many objects")
          }
          output.push(item)
        }
      }
    _ => ()
  }
}

///|
fn Hdf5Writer::global_heap_strings(
  self : Hdf5Writer,
) -> Array[String] raise Hdf5Error {
  let strings : Array[String] = []
  for dataset in self.datasets {
    collect_heap_strings(strings, dataset.values)
  }
  for attribute in self.attributes {
    collect_heap_strings(strings, attribute.values)
  }
  strings
}

///|
fn encode_global_heap(heap : WritableGlobalHeap) -> Bytes raise Hdf5Error {
  if heap.strings.length() == 0 {
    return b""
  }
  let mut used_size = 16
  for item in heap.strings {
    let length = @utf8.encode(item).length()
    guard used_size <= 0x7FFFFFFF - 16 - round_up_to_eight(length) else {
      raise InvalidField("writer global heap is too large")
    }
    used_size = used_size + 16 + round_up_to_eight(length)
  }
  let collection_size_64 = (used_size.to_int64() + 16L + 4095L) / 4096L * 4096L
  guard collection_size_64 <= 0x7FFFFFFFL else {
    raise InvalidField("writer global heap is too large")
  }
  let collection_size = collection_size_64.to_int()
  let buffer = @buffer.Buffer::Buffer(size_hint=collection_size)
  writer_write_bytes(buffer, b"GCOL")
  buffer.write_byte(b'\x01')
  writer_write_zeros(buffer, 3)
  buffer.write_uint64_le(collection_size.to_uint64())
  for index in 0..<heap.strings.length() {
    let encoded = @utf8.encode(heap.strings[index])
    buffer.write_uint16_le((index + 1).to_uint16())
    buffer.write_uint16_le(0)
    buffer.write_uint_le(0U)
    buffer.write_uint64_le(encoded.length().to_uint64())
    writer_write_bytes(buffer, encoded)
    writer_write_zeros(
      buffer,
      round_up_to_eight(encoded.length()) - encoded.length(),
    )
  }
  buffer.write_uint16_le(0)
  buffer.write_uint16_le(0)
  buffer.write_uint_le(0U)
  buffer.write_uint64_le((collection_size - used_size).to_uint64())
  writer_write_zeros(buffer, collection_size - buffer.length())
  buffer.to_bytes()
}

///|
fn assign_writable_addresses(
  writer : Hdf5Writer,
  templates : Array[WritableObject],
) -> WritableLayout raise Hdf5Error {
  let empty_heap = empty_writable_heap()
  let lengths : Array[Int32] = []
  for object in templates {
    lengths.push(
      encode_object_header(writer.object_payload(object, templates, empty_heap)).length(),
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
  let strings = writer.global_heap_strings()
  let heap = {
    address: if strings.length() == 0 {
      0UL
    } else {
      offset.to_uint64()
    },
    strings,
  }
  if heap.strings.length() > 0 {
    offset = writer_align(offset + encode_global_heap(heap).length())
  }
  for index in 0..<objects.length() {
    match objects[index].kind {
      WritableDatasetObject(dataset_index) => {
        let dataset = writer.datasets[dataset_index]
        let has_storage = match dataset.storage {
          WritableContiguous =>
            encode_write_values(dataset.values).payload.length() > 0
          WritableChunked(_, _) => true
          WritableExtensible(_, _, _) => true
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
                offset +
                encode_write_values(dataset.values, heap~).payload.length(),
              )
            WritableChunked(_, _) =>
              writer_align(
                offset +
                encode_fixed_array_storage(offset.to_uint64(), dataset, heap).length(),
              )
            WritableExtensible(_, _, _) =>
              writer_align(
                offset +
                encode_extensible_array_storage(
                  offset.to_uint64(),
                  dataset,
                  heap,
                ).length(),
              )
          }
        }
      }
      WritableGroup => ()
    }
  }
  { objects, heap, }
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
  let layout = assign_writable_addresses(self, initial_writable_objects(self))
  let objects = layout.objects
  let encoded_headers : Array[Bytes] = []
  let mut eof = 48
  for object in objects {
    let header = encode_object_header(
      self.object_payload(object, objects, layout.heap),
    )
    encoded_headers.push(header)
    eof = writer_align(object.address.to_int() + header.length())
  }
  let heap_payload = encode_global_heap(layout.heap)
  if heap_payload.length() > 0 {
    eof = writer_align(layout.heap.address.to_int() + heap_payload.length())
  }
  let raw_payloads : Array[(UInt64, Bytes)] = []
  for object in objects {
    match object.kind {
      WritableDatasetObject(index) => {
        let dataset = self.datasets[index]
        let payload = match dataset.storage {
          WritableContiguous =>
            encode_write_values(dataset.values, heap=layout.heap).payload
          WritableChunked(_, _) =>
            encode_fixed_array_storage(object.raw_address, dataset, layout.heap)
          WritableExtensible(_, _, _) =>
            encode_extensible_array_storage(
              object.raw_address,
              dataset,
              layout.heap,
            )
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
  if heap_payload.length() > 0 {
    writer_write_zeros(output, layout.heap.address.to_int() - output.length())
    writer_write_bytes(output, heap_payload)
  }
  for (address, payload) in raw_payloads {
    writer_write_zeros(output, address.to_int() - output.length())
    writer_write_bytes(output, payload)
  }
  writer_write_zeros(output, eof - output.length())
  output.to_bytes()
}
