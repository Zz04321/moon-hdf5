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
  ObjectReferenceValues(Array[String?])
  Complex64Values(Array[(Float, Float)])
  Complex128Values(Array[(Double, Double)])
  StructuredValues(Datatype, Array[DatasetValue])
} derive(@debug.Debug)

///|
pub extend Hdf5WriteValues with @debug.Debug::{to_repr}

///|
fn writer_datatype(
  class : Int32,
  size : Int32,
  signed~ : Bool = false,
  little_endian~ : Bool = true,
  reference_kind~ : Int32 = 0,
  bit_offset~ : Int32 = 0,
  bit_precision~ : Int32 = 0,
  opaque_tag~ : String = "",
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
    little_endian,
    vlen_is_string: false,
    enum_members,
    compound_members,
    base_types,
    array_dimensions,
    bit_offset,
    bit_precision: if bit_precision > 0 {
      bit_precision
    } else if class == 0 || class == 1 {
      size * 8
    } else {
      0
    },
    float_sign_position: 0,
    float_exponent_position: 0,
    float_exponent_size: 0,
    float_mantissa_position: 0,
    float_mantissa_size: 0,
    float_exponent_bias: 0,
    float_normalization: 0,
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
    opaque_tag,
    reference_kind,
  }
}

///|
/// Creates a standard little-endian integer datatype for structured writes.
pub fn Datatype::write_integer(
  signed : Bool,
  size : Int32,
) -> Datatype raise Hdf5Error {
  Datatype::write_integer_with_order(signed, size, true)
}

///|
/// Creates an integer datatype with an explicit byte order for structured writes.
pub fn Datatype::write_integer_with_order(
  signed : Bool,
  size : Int32,
  little_endian : Bool,
) -> Datatype raise Hdf5Error {
  guard size >= 1 && size <= 8 else {
    raise InvalidArgument("writer integer datatype width is invalid")
  }
  writer_datatype(0, size, signed~, little_endian~)
}

///|
/// Creates a standard little-endian IEEE floating-point datatype.
pub fn Datatype::write_float(size : Int32) -> Datatype raise Hdf5Error {
  Datatype::write_float_with_order(size, true)
}

///|
/// Creates an IEEE floating-point datatype with an explicit byte order for
/// structured writes.
pub fn Datatype::write_float_with_order(
  size : Int32,
  little_endian : Bool,
) -> Datatype raise Hdf5Error {
  guard size == 4 || size == 8 else {
    raise InvalidArgument("writer floating-point datatype width is invalid")
  }
  writer_datatype(1, size, little_endian~)
}

///|
/// Creates an HDF5 time datatype backed by an unsigned integer.
pub fn Datatype::write_time(
  size : Int32,
  little_endian~ : Bool = true,
) -> Datatype raise Hdf5Error {
  guard size == 1 || size == 2 || size == 4 || size == 8 else {
    raise InvalidArgument("writer time datatype width is invalid")
  }
  writer_datatype(2, size, little_endian~, bit_precision=size * 8)
}

///|
/// Creates a bitfield datatype with a declared offset and precision.
pub fn Datatype::write_bitfield(
  size : Int32,
  precision : Int32,
  offset~ : Int32 = 0,
  little_endian~ : Bool = true,
) -> Datatype raise Hdf5Error {
  guard (size == 1 || size == 2 || size == 4 || size == 8) &&
    precision > 0 &&
    precision <= 64 &&
    offset >= 0 &&
    offset <= size * 8 - precision else {
    raise InvalidArgument("writer bitfield layout is invalid")
  }
  writer_datatype(
    4,
    size,
    little_endian~,
    bit_offset=offset,
    bit_precision=precision,
  )
}

///|
/// Creates an opaque datatype with an optional ASCII tag.
pub fn Datatype::write_opaque(
  size : Int32,
  tag : String,
) -> Datatype raise Hdf5Error {
  guard size > 0 else {
    raise InvalidArgument("writer opaque datatype width is invalid")
  }
  let encoded = @utf8.encode(tag)
  for byte in encoded {
    guard byte > b'\x00' && byte < b'\x80' else {
      raise InvalidArgument("opaque datatype tag must be non-NUL ASCII")
    }
  }
  let padded = if encoded.length() == 0 {
    0
  } else {
    round_up_to_eight(encoded.length() + 1)
  }
  guard padded <= 255 else {
    raise InvalidArgument("opaque datatype tag is too long")
  }
  writer_datatype(5, size, opaque_tag=tag)
}

///|
/// Creates a variable-length sequence datatype with a fixed-size base type.
pub fn Datatype::write_vlen_sequence(
  base : Datatype,
) -> Datatype raise Hdf5Error {
  guard !datatype_contains_vlen(base) else {
    raise InvalidArgument("writer variable-length sequence base is invalid")
  }
  guard !datatype_contains_region_reference(base) else {
    raise InvalidArgument(
      "writer variable-length sequence base cannot contain region references",
    )
  }
  writer_datatype(9, 16, base_types=[base])
}

///|
/// Creates a fixed-width UTF-8 string datatype.
pub fn Datatype::write_fixed_string(width : Int32) -> Datatype raise Hdf5Error {
  guard width > 0 && width <= 65535 else {
    raise InvalidArgument("fixed string width is invalid")
  }
  writer_datatype(3, width)
}

///|
/// Creates a legacy object-reference datatype for structured writes.
pub fn Datatype::write_object_reference() -> Datatype {
  writer_datatype(7, 8)
}

///|
/// Creates a legacy dataset-region reference datatype.
pub fn Datatype::write_region_reference() -> Datatype {
  writer_datatype(7, 12, reference_kind=1)
}

///|
/// Creates a fixed-size array datatype.
pub fn Datatype::write_array(
  base : Datatype,
  dimensions : Array[UInt64],
) -> Datatype raise Hdf5Error {
  guard dimensions.length() > 0 && dimensions.length() <= 32 else {
    raise InvalidArgument("writer array datatype rank is invalid")
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
    raise InvalidArgument("writer enum requires an integer base and members")
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
    raise InvalidArgument("writer compound datatype requires members")
  }
  let fields : Array[CompoundMember] = []
  let mut size = 0
  for (name, datatype) in members {
    guard size <= 0x7FFFFFFF - datatype.size else {
      raise InvalidArgument("writer compound datatype is too large")
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
  /// Integer bits or decimal digits for floats; zero selects automatic bits for integers.
  ScaleOffset(Int32)
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
  WritableExternal(Array[Hdf5ExternalSlot])
}

///|
/// One external raw-data slot stored outside the HDF5 file.
pub(all) struct Hdf5ExternalSlot {
  file_name : String
  file_offset : UInt64
  data_size : UInt64
}

///|
priv struct WritableChunkFilterMask {
  chunk_index : Int32
  filter_mask : UInt32
}

///|
priv struct WritableDataset {
  path : String
  dimensions : Array[UInt64]
  maximum_dimensions : Array[UInt64]
  values : Hdf5WriteValues
  storage : WritableStorage
  fill_value : Hdf5WriteValues?
  chunk_filter_masks : Array[WritableChunkFilterMask]
}

///|
priv struct WritableAttribute {
  object_path : String
  name : String
  dimensions : Array[UInt64]
  values : Hdf5WriteValues
  creation_order : UInt64
}

///|
priv struct WritableSoftLink {
  parent_path : String
  name : String
  target : String
}

///|
priv struct WritableExternalLink {
  parent_path : String
  name : String
  target_file : String
  target_path : String
}

///|
priv struct WritableLinkOrder {
  path : String
  order : UInt64
}

///|
priv struct WritableLinkOrderCounter {
  parent_path : String
  next_order : UInt64
}

///|
/// In-memory builder for a new HDF5 file.
pub struct Hdf5Writer {
  priv groups : Array[String]
  priv datasets : Array[WritableDataset]
  priv attributes : Array[WritableAttribute]
  priv soft_links : Array[WritableSoftLink]
  priv external_links : Array[WritableExternalLink]
  priv link_orders : Array[WritableLinkOrder]
  priv link_order_counters : Array[WritableLinkOrderCounter]
  priv mut share_attributes : Bool
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
  dataspace : Dataspace?
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
  filter_mask : UInt32
}

///|
priv struct WritableGlobalHeap {
  address : UInt64
  region_address : UInt64
  strings : Array[String]
  regions : Array[Bytes]
  sequences : Array[Bytes]
}

///|
priv enum WritableDenseKind {
  WritableDenseLinks
  WritableDenseAttributes
} derive(Eq)

///|
priv struct WritableDenseMetadata {
  path : String
  kind : WritableDenseKind
  heap_address : UInt64
  index_address : UInt64
  order_index_address : UInt64
}

///|
priv struct WritableExternalHeap {
  path : String
  address : UInt64
}

///|
priv struct WritableDenseHeapEntry {
  name : String
  hash : UInt32
  data : Bytes
  creation_order : UInt64
  flags : Byte
  shared_id : Bytes?
}

///|
priv struct WritableSharedAttribute {
  path : String
  name : String
  data : Bytes
  id : Bytes
}

///|
priv struct WritableSharedMetadata {
  extension_address : UInt64
  table_address : UInt64
  heap_address : UInt64
  attributes : Array[WritableSharedAttribute]
}

///|
priv struct WritableLayout {
  objects : Array[WritableObject]
  heap : WritableGlobalHeap
  dense : Array[WritableDenseMetadata]
  external : Array[WritableExternalHeap]
  shared : WritableSharedMetadata?
}

///|
/// Creates an empty writer containing the root group.
pub fn Hdf5Writer::new() -> Hdf5Writer {
  {
    groups: ["/"],
    datasets: [],
    attributes: [],
    soft_links: [],
    external_links: [],
    link_orders: [],
    link_order_counters: [],
    share_attributes: false,
  }
}

///|
/// Enables or disables writing whole attributes through the shared-object
/// header message (SOHM) table. Attributes whose serialized message is at most
/// 4 KiB are shared; larger attributes remain inline or dense.
pub fn Hdf5Writer::set_attribute_sharing(
  self : Hdf5Writer,
  enabled : Bool,
) -> Unit {
  self.share_attributes = enabled
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
    raise InvalidArgument("writer paths must be absolute")
  }
  let segments : Array[String] = []
  for raw in path.split("/") {
    let segment = raw.to_owned()
    if segment != "" {
      guard segment != "." && segment != ".." else {
        raise InvalidArgument("writer paths cannot contain dot segments")
      }
      let encoded = @utf8.encode(segment)
      guard encoded.length() <= 65535 else {
        raise InvalidArgument("writer path component is too long")
      }
      for byte in encoded {
        guard byte != b'\x00' else {
          raise InvalidArgument("writer paths cannot contain NUL")
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
  guard segments.length() > 0 else {
    raise InvalidArgument("root has no parent")
  }
  segments.pop() |> ignore
  writer_path(segments)
}

///|
fn writer_basename(path : String) -> String raise Hdf5Error {
  let segments = writer_path_segments(path)
  guard segments.length() > 0 else { raise InvalidArgument("root has no name") }
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
    ObjectReferenceValues(items) => items.length()
    Complex64Values(items) => items.length()
    Complex128Values(items) => items.length()
    StructuredValues(_, items) => items.length()
  }
}

///|
fn writer_value_element_size(values : Hdf5WriteValues) -> Int32 {
  match values {
    Int8Values(_) | UInt8Values(_) => 1
    Int16Values(_) | UInt16Values(_) => 2
    Int32Values(_) | UInt32Values(_) | Float32Values(_) => 4
    Int64Values(_) | UInt64Values(_) | Float64Values(_) => 8
    FixedStringValues(_, width) => width
    VariableStringValues(_) => 16
    ObjectReferenceValues(_) => 8
    Complex64Values(_) => 8
    Complex128Values(_) => 16
    StructuredValues(datatype, _) => datatype.size
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
    ObjectReferenceValues(items) => ObjectReferenceValues(items.copy())
    Complex64Values(items) => Complex64Values(items.copy())
    Complex128Values(items) => Complex128Values(items.copy())
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
    float_sign_position: datatype.float_sign_position,
    float_exponent_position: datatype.float_exponent_position,
    float_exponent_size: datatype.float_exponent_size,
    float_mantissa_position: datatype.float_mantissa_position,
    float_mantissa_size: datatype.float_mantissa_size,
    float_exponent_bias: datatype.float_exponent_bias,
    float_normalization: datatype.float_normalization,
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
    RegionReference(path, selection) =>
      RegionReference(path, copy_region_selection(selection))
    _ => value
  }
}

///|
fn copy_region_selection(
  selection : Hdf5RegionSelection,
) -> Hdf5RegionSelection {
  match selection {
    RegionEmpty | RegionAll => selection
    RegionPoints(points) => RegionPoints(points.map(point => point.copy()))
    RegionHyperslab(start, stride, count, block) =>
      RegionHyperslab(start.copy(), stride.copy(), count.copy(), block.copy())
    RegionBlocks(blocks) =>
      RegionBlocks(
        blocks.map(block => {
          start: block.start.copy(),
          end: block.end.copy(),
        }),
      )
  }
}

///|
fn writer_shape_count(dimensions : Array[UInt64]) -> Int32 raise Hdf5Error {
  let mut count = 1UL
  for dimension in dimensions {
    guard dimension <= 0x7FFFFFFFUL else {
      raise InvalidArgument("writer dimension is too large")
    }
  }
  if dimensions.contains(0UL) {
    return 0
  }
  for dimension in dimensions {
    guard count <= 0x7FFFFFFFUL / dimension else {
      raise InvalidArgument("writer dataspace is too large")
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
    raise InvalidArgument("writer dataset byte size is too large")
  }
}

///|
fn validate_writer_datatype_name(
  name : String,
  description : String,
) -> Unit raise Hdf5Error {
  let encoded = @utf8.encode(name)
  guard encoded.length() > 0 && encoded.length() <= 65535 else {
    raise InvalidArgument(description + " is empty or too long")
  }
  for byte in encoded {
    guard byte != b'\x00' else {
      raise InvalidArgument(description + " contains NUL")
    }
  }
}

///|
fn datatype_contains_region_reference(datatype : Datatype) -> Bool {
  if datatype.class == 7 && datatype.reference_kind == 1 {
    return true
  }
  for field in datatype.compound_members {
    if datatype_contains_region_reference(field.datatype) {
      return true
    }
  }
  for base in datatype.base_types {
    if datatype_contains_region_reference(base) {
      return true
    }
  }
  false
}

///|
fn validate_writer_datatype(
  datatype : Datatype,
  depth : Int32,
) -> Unit raise Hdf5Error {
  guard depth < 16 && datatype.size > 0 else {
    raise InvalidArgument("writer datatype is invalid or too deeply nested")
  }
  match datatype.class {
    0 => {
      guard datatype.size >= 1 && datatype.size <= 8 else {
        raise InvalidArgument("writer integer datatype is unsupported")
      }
    }
    1 => {
      guard datatype.size == 4 || datatype.size == 8 else {
        raise InvalidArgument("writer floating-point datatype is unsupported")
      }
    }
    2 => {
      guard (
          datatype.size == 1 ||
          datatype.size == 2 ||
          datatype.size == 4 ||
          datatype.size == 8
        ) &&
        datatype.bit_precision > 0 &&
        datatype.bit_precision <= datatype.size * 8 else {
        raise InvalidArgument("writer time datatype is unsupported")
      }
    }
    3 => {
      guard datatype.size <= 65535 &&
        datatype.string_padding == 1 &&
        datatype.string_charset == 1 else {
        raise InvalidArgument("writer string datatype is unsupported")
      }
    }
    4 => {
      guard (
          datatype.size == 1 ||
          datatype.size == 2 ||
          datatype.size == 4 ||
          datatype.size == 8
        ) &&
        datatype.bit_precision > 0 &&
        datatype.bit_precision <= 64 &&
        datatype.bit_offset >= 0 &&
        datatype.bit_offset <= datatype.size * 8 - datatype.bit_precision else {
        raise InvalidArgument("writer bitfield datatype is unsupported")
      }
    }
    5 => {
      let encoded = @utf8.encode(datatype.opaque_tag)
      for byte in encoded {
        guard byte > b'\x00' && byte < b'\x80' else {
          raise InvalidArgument("writer opaque datatype tag is invalid")
        }
      }
      guard (if encoded.length() == 0 {
          0
        } else {
          round_up_to_eight(encoded.length() + 1)
        }) <=
        255 else {
        raise InvalidArgument("writer opaque datatype tag is too long")
      }
    }
    6 => {
      guard datatype.compound_members.length() > 0 &&
        datatype.compound_members.length() <= 65535 else {
        raise InvalidArgument("writer compound datatype requires members")
      }
      for index in 0..<datatype.compound_members.length() {
        let field = datatype.compound_members[index]
        validate_writer_datatype_name(field.name, "compound member name")
        validate_writer_datatype(field.datatype, depth + 1)
        guard field.offset >= 0 &&
          field.offset <= datatype.size - field.datatype.size else {
          raise InvalidArgument("compound member extends beyond datatype size")
        }
        for previous in 0..<index {
          guard datatype.compound_members[previous].name != field.name else {
            raise InvalidArgument("duplicate compound member name")
          }
          let earlier = datatype.compound_members[previous]
          guard field.offset + field.datatype.size <= earlier.offset ||
            earlier.offset + earlier.datatype.size <= field.offset else {
            raise InvalidArgument("compound writer members overlap")
          }
        }
      }
    }
    7 => {
      guard (datatype.reference_kind == 0 && datatype.size == 8) ||
        (datatype.reference_kind == 1 && datatype.size == 12) else {
        raise InvalidArgument("writer reference datatype is unsupported")
      }
    }
    8 => {
      guard datatype.base_types.length() == 1 &&
        datatype.enum_members.length() > 0 &&
        datatype.enum_members.length() <= 65535 else {
        raise InvalidArgument("writer enum datatype is invalid")
      }
      let base = datatype.base_types[0]
      validate_writer_datatype(base, depth + 1)
      guard base.class == 0 && base.size == datatype.size else {
        raise InvalidArgument("writer enum base datatype is invalid")
      }
      for index in 0..<datatype.enum_members.length() {
        let enum_item = datatype.enum_members[index]
        validate_writer_datatype_name(enum_item.name, "enum member name")
        guard (base.signed && enum_item.value is EnumValue::Signed(_)) ||
          (!base.signed && enum_item.value is EnumValue::Unsigned(_)) else {
          raise InvalidArgument(
            "enum member signedness does not match its base",
          )
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
            raise InvalidArgument("duplicate enum member name or value")
          }
        }
      }
    }
    9 => {
      guard datatype.size == 16 &&
        !datatype.vlen_is_string &&
        datatype.base_types.length() == 1 else {
        raise InvalidArgument("writer variable-length sequence is invalid")
      }
      let base = datatype.base_types[0]
      guard !datatype_contains_vlen(base) else {
        raise InvalidArgument(
          "writer variable-length sequence base must be fixed-size",
        )
      }
      guard !datatype_contains_region_reference(base) else {
        raise InvalidArgument(
          "writer variable-length sequence base cannot contain region references",
        )
      }
      validate_writer_datatype(base, depth + 1)
    }
    10 => {
      guard datatype.base_types.length() == 1 &&
        datatype.array_dimensions.length() > 0 &&
        datatype.array_dimensions.length() <= 32 else {
        raise InvalidArgument("writer array datatype is invalid")
      }
      let base = datatype.base_types[0]
      validate_writer_datatype(base, depth + 1)
      guard datatype_array_size(datatype.array_dimensions, base.size) ==
        datatype.size else {
        raise InvalidArgument(
          "array datatype size does not match its base type",
        )
      }
    }
    _ =>
      raise InvalidArgument("writer structured datatype class is unsupported")
  }
}

///|
fn validate_writer_integer(
  value : DatasetValue,
  datatype : Datatype,
) -> Unit raise Hdf5Error {
  if datatype.signed {
    guard value is SignedInteger(number) else {
      raise InvalidArgument("structured integer value has the wrong kind")
    }
    if datatype.size < 8 {
      let bits = datatype.size * 8
      let minimum = -(1L << (bits - 1))
      let maximum = (1L << (bits - 1)) - 1L
      guard number >= minimum && number <= maximum else {
        raise InvalidArgument("structured signed integer is out of range")
      }
    }
  } else {
    guard value is UnsignedInteger(number) else {
      raise InvalidArgument("structured integer value has the wrong kind")
    }
    if datatype.size < 8 {
      guard number < 1UL << (datatype.size * 8) else {
        raise InvalidArgument("structured unsigned integer is out of range")
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
    raise InvalidArgument("writer value is too deeply nested")
  }
  match datatype.class {
    0 => validate_writer_integer(value, datatype)
    1 => {
      guard value is FloatingPoint(_) else {
        raise InvalidArgument(
          "structured floating-point value has the wrong kind",
        )
      }
    }
    2 => {
      guard value is Time(number) else {
        raise InvalidArgument("structured time value has the wrong kind")
      }
      if datatype.bit_precision < 64 {
        guard number < 1UL << datatype.bit_precision else {
          raise InvalidArgument("structured time is out of range")
        }
      }
    }
    3 => {
      guard value is Text(text) else {
        raise InvalidArgument("structured string value has the wrong kind")
      }
      guard @utf8.encode(text).length() <= datatype.size else {
        raise InvalidArgument("structured string exceeds its width")
      }
    }
    4 => {
      guard value is Bitfield(number) else {
        raise InvalidArgument("structured bitfield value has the wrong kind")
      }
      if datatype.bit_precision < 64 {
        guard number < 1UL << datatype.bit_precision else {
          raise InvalidArgument("structured bitfield is out of range")
        }
      }
    }
    5 => {
      guard value is Opaque(bytes) else {
        raise InvalidArgument("structured opaque value has the wrong kind")
      }
      guard bytes.length() == datatype.size else {
        raise InvalidArgument("structured opaque value has the wrong size")
      }
    }
    6 => {
      guard value is Compound(fields) &&
        fields.length() == datatype.compound_members.length() else {
        raise InvalidArgument("compound value does not match its datatype")
      }
      for index in 0..<fields.length() {
        guard fields[index].name == datatype.compound_members[index].name else {
          raise InvalidArgument("compound value fields are out of order")
        }
        validate_structured_value(
          fields[index].value,
          datatype.compound_members[index].datatype,
          depth + 1,
        )
      }
    }
    7 =>
      if datatype.reference_kind == 0 {
        guard value is ObjectReference(path) else {
          raise InvalidArgument("object reference value has the wrong kind")
        }
        match path {
          Some(target) => canonical_writer_path(target) |> ignore
          None => ()
        }
      } else {
        guard value is RegionReference(path, _) else {
          raise InvalidArgument("region reference value has the wrong kind")
        }
        match path {
          Some(target) => canonical_writer_path(target) |> ignore
          None => ()
        }
      }
    8 => {
      guard value is Enumeration(name) else {
        raise InvalidArgument("enum value has the wrong kind")
      }
      guard datatype.enum_members.search_by(enum_item => enum_item.name == name)
        is Some(_) else {
        raise InvalidArgument("enum value does not match any member")
      }
    }
    9 => {
      guard value is Sequence(items) else {
        raise InvalidArgument(
          "variable-length sequence value has the wrong kind",
        )
      }
      let base = datatype.base_types[0]
      for item in items {
        validate_structured_value(item, base, depth + 1)
      }
    }
    10 => {
      guard value is Elements(items) else {
        raise InvalidArgument("array value has the wrong kind")
      }
      let base = datatype.base_types[0]
      guard items.length() == datatype.size / base.size else {
        raise InvalidArgument(
          "array value element count does not match its datatype",
        )
      }
      for item in items {
        validate_structured_value(item, base, depth + 1)
      }
    }
    _ =>
      raise InvalidArgument("writer structured datatype class is unsupported")
  }
}

///|
fn validate_write_values(
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
) -> Unit raise Hdf5Error {
  guard dimensions.length() <= 32 else {
    raise InvalidArgument("writer dataspace rank exceeds 32")
  }
  guard writer_shape_count(dimensions) == writer_value_count(values) else {
    raise InvalidArgument("writer value count does not match dataspace")
  }
  match values {
    FixedStringValues(items, width) => {
      guard width > 0 && width <= 65535 else {
        raise InvalidArgument("fixed string width is invalid")
      }
      for item in items {
        guard @utf8.encode(item).length() <= width else {
          raise InvalidArgument("fixed string value exceeds its width")
        }
      }
      validate_writer_byte_size(items.length(), width)
    }
    VariableStringValues(items) => {
      validate_writer_byte_size(items.length(), 16)
      for item in items {
        guard @utf8.encode(item).length() <= 0x7FFFFFFF else {
          raise InvalidArgument("variable string value is too large")
        }
      }
    }
    ObjectReferenceValues(items) => {
      validate_writer_byte_size(items.length(), 8)
      for item in items {
        match item {
          Some(path) => canonical_writer_path(path) |> ignore
          None => ()
        }
      }
    }
    Int8Values(items) => {
      validate_writer_byte_size(items.length(), 1)
      for item in items {
        guard item >= -128 && item <= 127 else {
          raise InvalidArgument("Int8 writer value is out of range")
        }
      }
    }
    Int16Values(items) => {
      validate_writer_byte_size(items.length(), 2)
      for item in items {
        guard item >= -32768 && item <= 32767 else {
          raise InvalidArgument("Int16 writer value is out of range")
        }
      }
    }
    UInt8Values(items) => {
      validate_writer_byte_size(items.length(), 1)
      for item in items {
        guard item <= 255U else {
          raise InvalidArgument("UInt8 writer value is out of range")
        }
      }
    }
    UInt16Values(items) => {
      validate_writer_byte_size(items.length(), 2)
      for item in items {
        guard item <= 65535U else {
          raise InvalidArgument("UInt16 writer value is out of range")
        }
      }
    }
    Int64Values(items) => validate_writer_byte_size(items.length(), 8)
    UInt64Values(items) => validate_writer_byte_size(items.length(), 8)
    Float64Values(items) => validate_writer_byte_size(items.length(), 8)
    Complex64Values(items) => validate_writer_byte_size(items.length(), 8)
    Complex128Values(items) => validate_writer_byte_size(items.length(), 16)
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
fn writer_soft_link_path(parent_path : String, name : String) -> String {
  if parent_path == "/" {
    "/" + name
  } else {
    parent_path + "/" + name
  }
}

///|
fn Hdf5Writer::has_soft_link(self : Hdf5Writer, path : String) -> Bool {
  self.soft_links.search_by(link => {
    writer_soft_link_path(link.parent_path, link.name) == path
  })
  is Some(_)
}

///|
fn Hdf5Writer::has_external_link(self : Hdf5Writer, path : String) -> Bool {
  self.external_links.search_by(link => {
    writer_soft_link_path(link.parent_path, link.name) == path
  })
  is Some(_)
}

///|
fn Hdf5Writer::has_entry(self : Hdf5Writer, path : String) -> Bool {
  self.has_object(path) ||
  self.has_soft_link(path) ||
  self.has_external_link(path)
}

///|
fn validate_writer_link_name(
  name : String,
  kind : String,
) -> Unit raise Hdf5Error {
  let encoded = @utf8.encode(name)
  guard encoded.length() > 0 else {
    raise InvalidArgument(kind + " name is empty")
  }
  guard name.split("/").count() == 1 else {
    raise InvalidArgument(kind + " names cannot contain '/'")
  }
  guard name != "." && name != ".." else {
    raise InvalidArgument(kind + " names cannot be dot segments")
  }
  guard encoded.length() <= 65535 else {
    raise InvalidArgument(kind + " name is too long")
  }
  for byte in encoded {
    guard byte != b'\x00' else {
      raise InvalidArgument(kind + " names cannot contain NUL")
    }
  }
}

///|
fn validate_writer_link_string(
  value : String,
  description : String,
) -> Unit raise Hdf5Error {
  let encoded = @utf8.encode(value)
  guard encoded.length() > 0 else {
    raise InvalidArgument(description + " is empty")
  }
  guard encoded.length() <= 65535 else {
    raise InvalidArgument(description + " is too long")
  }
  for byte in encoded {
    guard byte != b'\x00' else {
      raise InvalidArgument(description + " cannot contain NUL")
    }
  }
}

///|
fn validate_soft_link_target(target : String) -> Unit raise Hdf5Error {
  let encoded = @utf8.encode(target)
  guard encoded.length() > 0 else {
    raise InvalidArgument("soft link target is empty")
  }
  guard encoded.length() <= 65535 else {
    raise InvalidArgument("soft link target is too long")
  }
  for byte in encoded {
    guard byte != b'\x00' else {
      raise InvalidArgument("soft link targets cannot contain NUL")
    }
  }
}

///|
fn validate_writer_link_message_size(
  name : String,
  info_size : Int32,
  kind : String,
) -> Unit raise Hdf5Error {
  let encoded_name = @utf8.encode(name)
  guard 16 + encoded_name.length() + info_size <= 65535 else {
    raise InvalidArgument(kind + " message is too long")
  }
}

///|
fn Hdf5Writer::record_link(
  self : Hdf5Writer,
  path : String,
) -> Unit raise Hdf5Error {
  guard path != "/" else { return }
  if self.link_orders.search_by(link => link.path == path) is Some(_) {
    return
  }
  let parent_path = writer_parent_path(path)
  let counter = match
    self.link_order_counters.search_by(counter => {
      counter.parent_path == parent_path
    }) {
    Some(index) => index
    None => {
      self.link_order_counters.push({ parent_path, next_order: 0UL, })
      self.link_order_counters.length() - 1
    }
  }
  let order = self.link_order_counters[counter].next_order
  guard order < 0xFFFFFFFFFFFFFFFFUL else {
    raise InvalidArgument("writer has too many links")
  }
  self.link_orders.push({ path, order, })
  self.link_order_counters[counter] = { parent_path, next_order: order + 1UL, }
}

///|
fn Hdf5Writer::link_creation_order(
  self : Hdf5Writer,
  path : String,
) -> UInt64 raise Hdf5Error {
  match self.link_orders.search_by(link => link.path == path) {
    Some(index) => self.link_orders[index].order
    None => raise InvalidField("writer link has no creation order")
  }
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
      raise InvalidArgument("writer path conflicts with a dataset")
    }
    guard !self.has_soft_link(current) else {
      raise InvalidArgument("writer path conflicts with a soft link")
    }
    guard !self.has_external_link(current) else {
      raise InvalidArgument("writer path conflicts with an external link")
    }
    if !self.groups.contains(current) {
      self.record_link(current)
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
/// Adds a soft link at `parent_path`/`name` pointing to `target`.
///
/// Absolute targets start at the root group. Relative targets are resolved
/// from the group containing the link.
pub fn Hdf5Writer::add_soft_link(
  self : Hdf5Writer,
  parent_path : String,
  name : String,
  target : String,
) -> Unit raise Hdf5Error {
  validate_writer_link_name(name, "soft link")
  validate_soft_link_target(target)
  validate_writer_link_message_size(
    name,
    @utf8.encode(target).length(),
    "soft link",
  )
  let normalized_parent = canonical_writer_path(parent_path)
  let link_path = writer_soft_link_path(normalized_parent, name)
  guard !self.has_entry(link_path) else {
    raise InvalidArgument("writer object path already exists")
  }
  self.ensure_groups(normalized_parent)
  self.record_link(link_path)
  self.soft_links.push({ parent_path: normalized_parent, name, target, })
}

///|
/// Adds an external link at `parent_path`/`name`.
///
/// `target_file` is resolved by the HDF5 reader relative to the file
/// containing the link. `target_path` names an object inside that file.
pub fn Hdf5Writer::add_external_link(
  self : Hdf5Writer,
  parent_path : String,
  name : String,
  target_file : String,
  target_path : String,
) -> Unit raise Hdf5Error {
  validate_writer_link_name(name, "external link")
  validate_writer_link_string(target_file, "external link target file")
  validate_writer_link_string(target_path, "external link target path")
  validate_writer_link_message_size(
    name,
    5 + @utf8.encode(target_file).length() + @utf8.encode(target_path).length(),
    "external link",
  )
  let normalized_parent = canonical_writer_path(parent_path)
  let link_path = writer_soft_link_path(normalized_parent, name)
  guard !self.has_entry(link_path) else {
    raise InvalidArgument("writer object path already exists")
  }
  self.ensure_groups(normalized_parent)
  self.record_link(link_path)
  self.external_links.push({
    parent_path: normalized_parent,
    name,
    target_file,
    target_path,
  })
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
    raise InvalidArgument("a dataset cannot replace the root group")
  }
  guard !self.has_entry(normalized) else {
    raise InvalidArgument("writer object path already exists")
  }
  validate_write_values(dimensions, values)
  self.ensure_groups(writer_parent_path(normalized))
  self.record_link(normalized)
  self.datasets.push({
    path: normalized,
    dimensions: dimensions.copy(),
    maximum_dimensions: dimensions.copy(),
    values: copy_write_values(values),
    storage: WritableContiguous,
    fill_value: None,
    chunk_filter_masks: [],
  })
}

///|
fn validate_external_slots(
  slots : Array[Hdf5ExternalSlot],
  expected_size : UInt64,
) -> Unit raise Hdf5Error {
  guard slots.length() > 0 && slots.length() <= 65535 else {
    raise InvalidArgument("external dataset requires 1 to 65535 slots")
  }
  let mut total = 0UL
  for slot in slots {
    let encoded = @utf8.encode(slot.file_name)
    guard encoded.length() > 0 && encoded.length() <= 65535 else {
      raise InvalidArgument("external data file name is empty or too long")
    }
    for byte in encoded {
      guard byte != b'\x00' else {
        raise InvalidArgument("external data file name cannot contain NUL")
      }
    }
    guard slot.data_size > 0UL else {
      raise InvalidArgument("external data slot is empty")
    }
    guard slot.file_offset <= 0xFFFFFFFFFFFFFFFFUL - slot.data_size else {
      raise InvalidArgument("external data file offset is too large")
    }
    guard total <= 0xFFFFFFFFFFFFFFFFUL - slot.data_size else {
      raise InvalidArgument("external data size is too large")
    }
    total = total + slot.data_size
  }
  guard total == expected_size else {
    raise InvalidArgument("external data slots do not match the dataset size")
  }
}

///|
/// Adds an unfiltered dataset whose raw bytes live in external files.
///
/// Slots are concatenated in caller order to form the row-major element
/// stream. Their sizes must add up exactly to the dataset byte size.
pub fn Hdf5Writer::add_external_dataset(
  self : Hdf5Writer,
  path : String,
  dimensions : Array[UInt64],
  values : Hdf5WriteValues,
  slots : Array[Hdf5ExternalSlot],
  fill_value? : Hdf5WriteValues,
) -> Unit raise Hdf5Error {
  let normalized = canonical_writer_path(path)
  guard normalized != "/" else {
    raise InvalidArgument("a dataset cannot replace the root group")
  }
  guard !self.has_entry(normalized) else {
    raise InvalidArgument("writer object path already exists")
  }
  validate_write_values(dimensions, values)
  guard !writer_values_contain_vlen(values) else {
    raise UnsupportedFeature(
      "external storage cannot hold variable-length values",
    )
  }
  let expected_size = writer_shape_count(dimensions).to_int64() *
    writer_value_element_size(values).to_int64()
  guard expected_size >= 0 else {
    raise InvalidArgument("external dataset size is too large")
  }
  validate_external_slots(slots, expected_size.reinterpret_as_uint64())
  let copied_fill = validated_write_fill(values, fill_value)
  self.ensure_groups(writer_parent_path(normalized))
  self.record_link(normalized)
  self.datasets.push({
    path: normalized,
    dimensions: dimensions.copy(),
    maximum_dimensions: dimensions.copy(),
    values: copy_write_values(values),
    storage: WritableExternal(slots.copy()),
    fill_value: copied_fill,
    chunk_filter_masks: [],
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
    (ObjectReferenceValues(_), ObjectReferenceValues(_)) => true
    (Complex64Values(_), Complex64Values(_)) => true
    (Complex128Values(_), Complex128Values(_)) => true
    (StructuredValues(left_type, _), StructuredValues(right_type, _)) =>
      encode_writer_datatype(left_type) == encode_writer_datatype(right_type)
    _ => false
  }
}

///|
fn validate_write_filters(
  filters : Array[Hdf5WriteFilter],
) -> Unit raise Hdf5Error {
  guard filters.length() <= 32 else {
    raise InvalidArgument("too many filters")
  }
  let mut saw_shuffle = false
  let mut saw_deflate = false
  let mut saw_fletcher = false
  let mut saw_scaleoffset = false
  for filter in filters {
    match filter {
      Shuffle => {
        guard !saw_shuffle && !saw_deflate && !saw_fletcher && !saw_scaleoffset else {
          raise InvalidArgument("shuffle must be the first writer filter")
        }
        saw_shuffle = true
      }
      Deflate(level) => {
        guard !saw_deflate && !saw_fletcher && level >= 0 && level <= 9 else {
          raise InvalidArgument("invalid deflate writer filter")
        }
        saw_deflate = true
      }
      Fletcher32 => {
        guard !saw_fletcher else {
          raise InvalidArgument("duplicate Fletcher32 writer filter")
        }
        saw_fletcher = true
      }
      ScaleOffset(minbits) => {
        guard !saw_scaleoffset && !saw_shuffle && !saw_deflate && !saw_fletcher else {
          raise InvalidArgument("scale-offset must be the first writer filter")
        }
        guard minbits >= 0 else {
          raise InvalidArgument("invalid scale-offset writer filter")
        }
        saw_scaleoffset = true
      }
    }
  }
}

///|
fn writer_values_contain_vlen(values : Hdf5WriteValues) -> Bool {
  match values {
    VariableStringValues(_) => true
    StructuredValues(datatype, _) => datatype_contains_vlen(datatype)
    _ => false
  }
}

///|
fn validate_writer_filters_for_values(
  values : Hdf5WriteValues,
  filters : Array[Hdf5WriteFilter],
) -> Unit raise Hdf5Error {
  guard filters.length() == 0 || !writer_values_contain_vlen(values) else {
    raise UnsupportedFeature(
      "filters are not supported for variable-length datasets",
    )
  }
  for filter in filters {
    match filter {
      ScaleOffset(scale) => {
        let (element_size, _, datatype_class) = writer_scaleoffset_layout(
          values,
        )
        guard scale >= 0 else {
          raise InvalidArgument("invalid scale-offset writer filter")
        }
        if datatype_class == 0 {
          guard scale <= element_size * 8 else {
            raise InvalidArgument(
              "scale-offset minimum bits exceed the datatype width",
            )
          }
        } else {
          let maximum_digits = if element_size == 4 { 38 } else { 308 }
          guard scale <= maximum_digits else {
            raise InvalidArgument(
              "scale-offset decimal digits exceed the datatype precision",
            )
          }
          for other in filters {
            match other {
              Fletcher32 =>
                raise InvalidArgument(
                  "Fletcher32 cannot be combined with floating-point scale-offset",
                )
              _ => ()
            }
          }
        }
      }
      _ => ()
    }
  }
}

///|
fn writer_chunk_intersects_shape(
  index : Int32,
  grid : Array[Int32],
  order : Array[Int32],
  dimensions : Array[UInt64],
  chunk_dimensions : Array[UInt32],
) -> Bool {
  let coordinates : Array[Int32] = Array::make(grid.length(), 0)
  let mut remaining = index
  let mut order_index = order.length()
  while order_index > 0 {
    order_index = order_index - 1
    let axis = order[order_index]
    let count = grid[axis]
    if count <= 0 {
      return false
    }
    coordinates[axis] = remaining % count
    remaining = remaining / count
  }
  for axis in 0..<dimensions.length() {
    if coordinates[axis].to_uint64() * chunk_dimensions[axis].to_uint64() >=
      dimensions[axis] {
      return false
    }
  }
  true
}

///|
fn validate_writer_chunk_size(
  chunk_dimensions : Array[UInt32],
  element_size : Int32,
) -> Unit raise Hdf5Error {
  let mut chunk_bytes = element_size.to_int64()
  for dimension in chunk_dimensions {
    guard chunk_bytes <= 0x7FFFFFFFL / dimension.to_int64() else {
      raise InvalidArgument("writer chunk is too large")
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
      let vlen_string_fill = fill is VariableStringValues(_) &&
        values is VariableStringValues(_)
      guard vlen_string_fill || !writer_values_contain_vlen(values) else {
        raise UnsupportedFeature(
          "custom variable-length sequence fills are not supported yet",
        )
      }
      guard writer_value_count(fill) == 1 && same_write_datatype(values, fill) else {
        raise InvalidArgument(
          "fill value must be one value of the dataset type",
        )
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
    raise InvalidArgument("a dataset cannot replace the root group")
  }
  guard !self.has_entry(normalized) else {
    raise InvalidArgument("writer object path already exists")
  }
  validate_write_values(dimensions, values)
  guard dimensions.length() > 0 &&
    chunk_dimensions.length() == dimensions.length() else {
    raise InvalidArgument("chunk dimensions must match a simple dataspace")
  }
  for axis in 0..<chunk_dimensions.length() {
    let dimension = chunk_dimensions[axis]
    guard dimensions[axis] > 0UL && dimension > 0U && dimension <= 0x7FFFFFFFU else {
      raise InvalidArgument("chunk dimension is invalid")
    }
  }
  chunk_grid(dimensions, chunk_dimensions) |> ignore
  validate_writer_chunk_size(
    chunk_dimensions,
    writer_value_element_size(values),
  )
  validate_write_filters(filters)
  validate_writer_filters_for_values(values, filters)
  let copied_fill = validated_write_fill(values, fill_value)
  self.ensure_groups(writer_parent_path(normalized))
  self.record_link(normalized)
  self.datasets.push({
    path: normalized,
    dimensions: dimensions.copy(),
    maximum_dimensions: dimensions.copy(),
    values: copy_write_values(values),
    storage: WritableChunked(chunk_dimensions.copy(), filters.copy()),
    fill_value: copied_fill,
    chunk_filter_masks: [],
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
    None => raise InvalidArgument("writer dataset does not exist")
  }
}

///|
fn writer_dataset_chunk_grid(
  dataset : WritableDataset,
) -> (Array[UInt32], Array[Hdf5WriteFilter], Array[UInt64]) raise Hdf5Error {
  match dataset.storage {
    WritableChunked(dimensions, filters) =>
      (dimensions, filters, dataset.dimensions.copy())
    WritableExtensible(dimensions, filters, unlimited_axis) => {
      let grid_dimensions = dataset.dimensions.copy()
      for axis in 0..<grid_dimensions.length() {
        if axis != unlimited_axis {
          grid_dimensions[axis] = dataset.maximum_dimensions[axis]
        }
      }
      (dimensions, filters, grid_dimensions)
    }
    WritableContiguous | WritableExternal(_) =>
      raise InvalidArgument("dataset does not use chunked storage")
  }
}

///|
/// Sets the HDF5 filter mask for one chunk of an existing chunked dataset.
///
/// Bit `n` set means filter `n` in the dataset pipeline is skipped for that
/// chunk. The mask is stored in the fixed-array or extensible-array chunk index.
pub fn Hdf5Writer::set_chunk_filter_mask(
  self : Hdf5Writer,
  path : String,
  chunk_index : Int32,
  filter_mask : UInt32,
) -> Unit raise Hdf5Error {
  let index = self.writer_dataset_index(path)
  let dataset = self.datasets[index]
  let (chunk_dimensions, filters, grid_dimensions) = writer_dataset_chunk_grid(
    dataset,
  )
  guard filters.length() > 0 else {
    raise InvalidArgument("dataset has no filters for a chunk mask")
  }
  guard filters.length() == 32 || filter_mask >> filters.length() == 0U else {
    raise InvalidArgument("chunk filter mask exceeds the filter pipeline")
  }
  let (_, chunk_count) = chunk_grid(grid_dimensions, chunk_dimensions)
  guard chunk_index >= 0 && chunk_index < chunk_count else {
    raise InvalidArgument("chunk index is outside the dataset grid")
  }
  let masks = dataset.chunk_filter_masks.copy()
  if masks.search_by(entry => entry.chunk_index == chunk_index)
    is Some(mask_index) {
    masks[mask_index] = { chunk_index, filter_mask, }
  } else {
    masks.push({ chunk_index, filter_mask, })
  }
  self.datasets[index] = {
    path: dataset.path,
    dimensions: dataset.dimensions,
    maximum_dimensions: dataset.maximum_dimensions,
    values: dataset.values,
    storage: dataset.storage,
    fill_value: dataset.fill_value,
    chunk_filter_masks: masks,
  }
}

///|
fn dataset_chunk_filter_mask(
  dataset : WritableDataset,
  chunk_index : Int32,
) -> UInt32 {
  match
    dataset.chunk_filter_masks.search_by(entry => {
      entry.chunk_index == chunk_index
    }) {
    Some(index) => dataset.chunk_filter_masks[index].filter_mask
    None => 0U
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
    raise InvalidArgument(
      "extensible dimensions must have matching nonzero ranks",
    )
  }
  let encoded_maximum : Array[UInt64] = []
  let mut unlimited_axis = -1
  for axis in 0..<dimensions.length() {
    guard dimensions[axis] <= 0x7FFFFFFFUL &&
      chunk_dimensions[axis] > 0U &&
      chunk_dimensions[axis] <= 0x7FFFFFFFU else {
      raise InvalidArgument("extensible dataset dimension is invalid")
    }
    match maximum_dimensions[axis] {
      Some(maximum) => {
        guard maximum > 0UL &&
          maximum <= 0x7FFFFFFFUL &&
          dimensions[axis] <= maximum else {
          raise InvalidArgument(
            "extensible dataset maximum dimension is invalid",
          )
        }
        encoded_maximum.push(maximum)
      }
      None => {
        guard unlimited_axis == -1 else {
          raise InvalidArgument(
            "extensible dataset requires one unlimited dimension",
          )
        }
        unlimited_axis = axis
        encoded_maximum.push(0xFFFFFFFFFFFFFFFFUL)
      }
    }
  }
  guard unlimited_axis >= 0 else {
    raise InvalidArgument("extensible dataset requires one unlimited dimension")
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
    raise InvalidArgument("a dataset cannot replace the root group")
  }
  guard !self.has_entry(normalized) else {
    raise InvalidArgument("writer object path already exists")
  }
  validate_write_values(dimensions, values)
  let (encoded_maximum, unlimited_axis) = validate_extensible_dimensions(
    dimensions, maximum_dimensions, chunk_dimensions,
  )
  validate_writer_chunk_size(
    chunk_dimensions,
    writer_value_element_size(values),
  )
  validate_write_filters(filters)
  validate_writer_filters_for_values(values, filters)
  let copied_fill = validated_write_fill(values, fill_value)
  let grid_dimensions = dimensions.copy()
  for axis in 0..<grid_dimensions.length() {
    if axis != unlimited_axis {
      grid_dimensions[axis] = encoded_maximum[axis]
    }
  }
  chunk_grid(grid_dimensions, chunk_dimensions) |> ignore
  self.ensure_groups(writer_parent_path(normalized))
  self.record_link(normalized)
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
    chunk_filter_masks: [],
  })
}

///|
fn default_structured_value(
  datatype : Datatype,
) -> DatasetValue raise Hdf5Error {
  match datatype.class {
    0 => if datatype.signed { SignedInteger(0L) } else { UnsignedInteger(0UL) }
    1 => FloatingPoint(0.0)
    2 => Time(0UL)
    3 => Text("")
    4 => Bitfield(0UL)
    5 => Opaque(Bytes::make(datatype.size, b'\x00'))
    6 =>
      Compound(
        datatype.compound_members.map(field => {
          name: field.name,
          value: default_structured_value(field.datatype),
        }),
      )
    7 =>
      if datatype.reference_kind == 1 {
        RegionReference(None, RegionEmpty)
      } else {
        ObjectReference(None)
      }
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
        raise InvalidArgument("enum datatype has no zero-valued default fill")
      }
      Enumeration(datatype.enum_members[index].name)
    }
    9 => Sequence([])
    10 => {
      let base = datatype.base_types[0]
      Elements(
        Array::make(datatype.size / base.size, default_structured_value(base)),
      )
    }
    _ =>
      raise InvalidArgument("writer structured datatype class is unsupported")
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
    ObjectReferenceValues(_) => ObjectReferenceValues([None])
    Complex64Values(_) => Complex64Values([(0.0, 0.0)])
    Complex128Values(_) => Complex128Values([(0.0, 0.0)])
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
    (ObjectReferenceValues(_), ObjectReferenceValues(items)) =>
      ObjectReferenceValues(Array::make(count, items[0]))
    (Complex64Values(_), Complex64Values(items)) =>
      Complex64Values(Array::make(count, items[0]))
    (Complex128Values(_), Complex128Values(items)) =>
      Complex128Values(Array::make(count, items[0]))
    (StructuredValues(datatype, _), StructuredValues(_, items)) =>
      StructuredValues(datatype, Array::make(count, items[0]))
    _ =>
      raise InvalidArgument("fill value must be one value of the dataset type")
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
    _ => raise InvalidArgument("only extensible datasets can be resized")
  }
  guard dimensions.length() == dataset.dimensions.length() else {
    raise InvalidArgument("resized dataset rank does not match")
  }
  for axis in 0..<dimensions.length() {
    guard dimensions[axis] <= 0x7FFFFFFFUL &&
      (
        dataset.maximum_dimensions[axis] == 0xFFFFFFFFFFFFFFFFUL ||
        dimensions[axis] <= dataset.maximum_dimensions[axis]
      ) else {
      raise InvalidArgument("resized dataset exceeds its maximum dimensions")
    }
  }
  let grid_dimensions = dimensions.copy()
  for axis in 0..<grid_dimensions.length() {
    if axis != unlimited_axis {
      grid_dimensions[axis] = dataset.maximum_dimensions[axis]
    }
  }
  chunk_grid(grid_dimensions, chunk_dimensions) |> ignore
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
    chunk_filter_masks: dataset.chunk_filter_masks,
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
          raise InvalidArgument(
            "appended value is outside the destination shape",
          )
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
        raise InvalidArgument("appended values must match the dataset type")
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
    (ObjectReferenceValues(output), ObjectReferenceValues(items)) =>
      ObjectReferenceValues(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (Complex64Values(output), Complex64Values(items)) =>
      Complex64Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    (Complex128Values(output), Complex128Values(items)) =>
      Complex128Values(
        overlay_array(
          output, items, source_dimensions, destination_dimensions, axis, offset,
        ),
      )
    _ => raise InvalidArgument("appended values must match the dataset type")
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
    _ => raise InvalidArgument("only extensible datasets can be appended")
  }
  guard dimensions.length() == dataset.dimensions.length() else {
    raise InvalidArgument("appended dataset rank does not match")
  }
  validate_write_values(dimensions, values)
  guard same_write_datatype(dataset.values, values) else {
    raise InvalidArgument("appended values must match the dataset type")
  }
  for axis in 0..<dimensions.length() {
    if axis != unlimited_axis && dimensions[axis] != dataset.dimensions[axis] {
      raise InvalidArgument(
        "appended block must match fixed dataset dimensions",
      )
    }
  }
  guard dataset.dimensions[unlimited_axis] <=
    0x7FFFFFFFUL - dimensions[unlimited_axis] else {
    raise InvalidArgument("appended dataset dimension is too large")
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
    chunk_filter_masks: resized.chunk_filter_masks,
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
    raise InvalidArgument("attribute target does not exist")
  }
  let encoded_name = @utf8.encode(name)
  guard encoded_name.length() > 0 && encoded_name.length() < 65535 else {
    raise InvalidArgument("attribute name is empty or too long")
  }
  for byte in encoded_name {
    guard byte != b'\x00' else {
      raise InvalidArgument("attribute names cannot contain NUL")
    }
  }
  guard self.attributes.search_by(attribute => {
      attribute.object_path == normalized && attribute.name == name
    })
    is None else {
    raise InvalidArgument("attribute name already exists")
  }
  validate_write_values(dimensions, values)
  let creation_order = self.object_attributes(normalized).length().to_uint64()
  self.attributes.push({
    object_path: normalized,
    name,
    dimensions: dimensions.copy(),
    values: copy_write_values(values),
    creation_order,
  })
}

///|
fn write_uint_width(
  buffer : @buffer.Buffer,
  value : UInt64,
  size : Int32,
  little_endian~ : Bool = true,
) -> Unit {
  for index in 0..<size {
    let shift = if little_endian { index * 8 } else { (size - 1 - index) * 8 }
    buffer.write_byte(((value >> shift) & 0xFFUL).to_byte())
  }
}

///|
fn encode_integer_datatype(
  signed : Bool,
  size : Int32,
  little_endian~ : Bool = true,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=12)
  let flags = (if signed { 8 } else { 0 }) | (if little_endian { 0 } else { 1 })
  buffer.write_byte(b'\x10')
  buffer.write_byte(flags.to_byte())
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(size.reinterpret_as_uint())
  buffer.write_uint16_le(0)
  buffer.write_uint16_le((size * 8).to_uint16())
  buffer.to_bytes()
}

///|
fn encode_float_datatype(size : Int32, little_endian~ : Bool = true) -> Bytes {
  let single = size == 4
  let buffer = @buffer.Buffer::Buffer(size_hint=20)
  buffer.write_byte(b'\x11')
  buffer.write_byte(if little_endian { b'\x20' } else { b'\x21' })
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
    0 =>
      encode_integer_datatype(
        datatype.signed,
        datatype.size,
        little_endian=datatype.little_endian,
      )
    1 =>
      encode_float_datatype(datatype.size, little_endian=datatype.little_endian)
    2 => {
      let buffer = @buffer.Buffer::Buffer(size_hint=12)
      buffer.write_byte(b'\x12')
      buffer.write_byte(if datatype.little_endian { b'\x00' } else { b'\x01' })
      buffer.write_byte(b'\x00')
      buffer.write_byte(b'\x00')
      buffer.write_uint_le(datatype.size.reinterpret_as_uint())
      buffer.write_uint16_le(datatype.bit_precision.to_uint16())
      buffer.to_bytes()
    }
    3 => encode_string_datatype(datatype.size)
    4 => {
      let buffer = @buffer.Buffer::Buffer(size_hint=14)
      buffer.write_byte(b'\x14')
      buffer.write_byte(if datatype.little_endian { b'\x00' } else { b'\x01' })
      buffer.write_byte(b'\x00')
      buffer.write_byte(b'\x00')
      buffer.write_uint_le(datatype.size.reinterpret_as_uint())
      buffer.write_uint16_le(datatype.bit_offset.to_uint16())
      buffer.write_uint16_le(datatype.bit_precision.to_uint16())
      buffer.to_bytes()
    }
    5 => {
      let tag = @utf8.encode(datatype.opaque_tag)
      let padded = if tag.length() == 0 {
        0
      } else {
        round_up_to_eight(tag.length() + 1)
      }
      let buffer = @buffer.Buffer::Buffer(size_hint=8 + padded)
      buffer.write_byte(b'\x15')
      buffer.write_byte(padded.to_byte())
      buffer.write_byte(b'\x00')
      buffer.write_byte(b'\x00')
      buffer.write_uint_le(datatype.size.reinterpret_as_uint())
      if padded > 0 {
        writer_write_bytes(buffer, tag)
        writer_write_zeros(buffer, padded - tag.length())
      }
      buffer.to_bytes()
    }
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
    7 => encode_reference_datatype(datatype.size, datatype.reference_kind)
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
        write_uint_width(
          buffer,
          raw,
          datatype.size,
          little_endian=datatype.base_types[0].little_endian,
        )
      }
      buffer.to_bytes()
    }
    9 => {
      let buffer = @buffer.Buffer::Buffer()
      buffer.write_byte(b'\x19')
      buffer.write_byte(b'\x00')
      buffer.write_byte(b'\x00')
      buffer.write_byte(b'\x00')
      buffer.write_uint_le(datatype.size.reinterpret_as_uint())
      writer_write_bytes(buffer, encode_writer_datatype(datatype.base_types[0]))
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
  heap~ : WritableGlobalHeap = empty_writable_heap(),
  objects~ : Array[WritableObject] = [],
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
      write_uint_width(
        buffer,
        raw,
        datatype.size,
        little_endian=datatype.little_endian,
      )
    }
    1 =>
      match value {
        FloatingPoint(number) =>
          if datatype.size == 4 {
            if datatype.little_endian {
              buffer.write_float_le(Float::from_double(number))
            } else {
              buffer.write_float_be(Float::from_double(number))
            }
          } else if datatype.little_endian {
            buffer.write_double_le(number)
          } else {
            buffer.write_double_be(number)
          }
        _ => ()
      }
    2 =>
      match value {
        Time(number) =>
          write_uint_width(
            buffer,
            number,
            datatype.size,
            little_endian=datatype.little_endian,
          )
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
    4 =>
      match value {
        Bitfield(number) =>
          write_uint_width(
            buffer,
            number << datatype.bit_offset,
            datatype.size,
            little_endian=datatype.little_endian,
          )
        _ => ()
      }
    5 =>
      match value {
        Opaque(bytes) => writer_write_bytes(buffer, bytes)
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
              heap~,
              objects~,
            )
            output.blit_from_bytes(field.offset, encoded, 0, encoded.length())
          }
          writer_write_bytes(buffer, output.unsafe_reinterpret_as_bytes())
        }
        _ => ()
      }
    7 =>
      if datatype.reference_kind == 0 {
        match value {
          ObjectReference(path) => {
            let address = match path {
              None => 0UL
              Some(target) => writer_reference_address(objects, target)
            }
            buffer.write_uint64_le(address)
          }
          _ => ()
        }
      } else {
        match value {
          RegionReference(path, selection) =>
            match path {
              None => writer_write_zeros(buffer, 12)
              Some(target) => {
                let payload = writer_region_reference_payload(
                  objects, target, selection,
                )
                buffer.write_uint64_le(heap.region_address)
                buffer.write_uint_le(
                  writable_region_heap_index(heap, payload).reinterpret_as_uint(),
                )
              }
            }
          _ => ()
        }
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
          write_uint_width(
            buffer,
            raw,
            datatype.size,
            little_endian=datatype.base_types[0].little_endian,
          )
        }
        _ => ()
      }
    9 =>
      match value {
        Sequence(items) =>
          if items.length() == 0 {
            buffer.write_uint_le(0U)
            buffer.write_uint64_le(0UL)
            buffer.write_uint_le(0U)
          } else {
            let encoded = @buffer.Buffer::Buffer()
            let base = datatype.base_types[0]
            for item in items {
              writer_write_bytes(
                encoded,
                encode_structured_value(base, item, heap~, objects~),
              )
            }
            let data = encoded.to_bytes()
            buffer.write_uint_le(items.length().reinterpret_as_uint())
            buffer.write_uint64_le(heap.address)
            buffer.write_uint_le(
              writable_sequence_heap_index(heap, data).reinterpret_as_uint(),
            )
          }
        _ => ()
      }
    10 =>
      match value {
        Elements(items) =>
          for item in items {
            writer_write_bytes(
              buffer,
              encode_structured_value(
                datatype.base_types[0],
                item,
                heap~,
                objects~,
              ),
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
  buffer.write_byte(b'\x19')
  buffer.write_byte(b'\x01')
  buffer.write_byte(b'\x01')
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(16U)
  writer_write_bytes(buffer, encode_integer_datatype(false, 1))
  buffer.to_bytes()
}

///|
fn encode_object_reference_datatype() -> Bytes {
  encode_reference_datatype(8, 0)
}

///|
fn encode_reference_datatype(size : Int32, reference_kind : Int32) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=8)
  buffer.write_byte(b'\x17')
  buffer.write_byte(reference_kind.to_byte())
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x00')
  buffer.write_uint_le(size.reinterpret_as_uint())
  buffer.to_bytes()
}

///|
fn empty_writable_heap() -> WritableGlobalHeap {
  {
    address: 0UL,
    region_address: 0UL,
    strings: [],
    regions: [],
    sequences: [],
  }
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
fn writable_sequence_heap_index(
  heap : WritableGlobalHeap,
  value : Bytes,
) -> Int32 raise Hdf5Error {
  match heap.sequences.search_by(item => item == value) {
    Some(index) => heap.strings.length() + index + 1
    None => {
      guard heap.address == 0UL else {
        raise InvalidField("variable sequence is missing from the global heap")
      }
      0
    }
  }
}

///|
fn writable_region_heap_index(
  heap : WritableGlobalHeap,
  value : Bytes,
) -> Int32 raise Hdf5Error {
  match heap.regions.search_by(item => item == value) {
    Some(index) => index + 1
    None => {
      guard heap.address == 0UL else {
        raise InvalidField("region reference is missing from the global heap")
      }
      0
    }
  }
}

///|
fn encode_complex_datatype(size : Int32) -> Bytes raise Hdf5Error {
  let component = Datatype::write_float(size)
  encode_writer_datatype(
    Datatype::write_compound([("r", component), ("i", component)]),
  )
}

///|
fn encode_write_values(
  values : Hdf5WriteValues,
  heap~ : WritableGlobalHeap = empty_writable_heap(),
  objects~ : Array[WritableObject] = [],
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
    ObjectReferenceValues(items) => {
      for item in items {
        let address = match item {
          None => 0UL
          Some(path) => writer_reference_address(objects, path)
        }
        payload.write_uint64_le(address)
      }
      (encode_object_reference_datatype(), 8)
    }
    Complex64Values(items) => {
      for (real, imaginary) in items {
        payload.write_float_le(real)
        payload.write_float_le(imaginary)
      }
      (encode_complex_datatype(4), 8)
    }
    Complex128Values(items) => {
      for (real, imaginary) in items {
        payload.write_double_le(real)
        payload.write_double_le(imaginary)
      }
      (encode_complex_datatype(8), 16)
    }
    StructuredValues(datatype, items) => {
      for item in items {
        writer_write_bytes(
          payload,
          encode_structured_value(datatype, item, heap~, objects~),
        )
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
fn encode_external_data_layout(size : UInt64) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=18)
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x01')
  buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  buffer.write_uint64_le(size)
  buffer.to_bytes()
}

///|
fn external_slot_total(slots : Array[Hdf5ExternalSlot]) -> UInt64 {
  let mut total = 0UL
  for slot in slots {
    total = total + slot.data_size
  }
  total
}

///|
fn external_name_offsets(
  slots : Array[Hdf5ExternalSlot],
) -> (Array[UInt64], Int32) raise Hdf5Error {
  let names : Array[String] = []
  let offsets : Array[UInt64] = []
  let mut data_size = 8
  for slot in slots {
    match names.search_by(name => name == slot.file_name) {
      Some(index) => offsets.push(offsets[index])
      None => {
        let encoded = @utf8.encode(slot.file_name)
        guard encoded.length() < 0x7FFFFFFF - data_size else {
          raise InvalidArgument("external data file names are too long")
        }
        names.push(slot.file_name)
        offsets.push(data_size.to_uint64())
        data_size = data_size + encoded.length() + 1
      }
    }
  }
  (offsets, writer_align(data_size))
}

///|
fn encode_external_local_heap(
  slots : Array[Hdf5ExternalSlot],
  address : UInt64,
) -> Bytes raise Hdf5Error {
  let (_, data_size) = external_name_offsets(slots)
  let data = FixedArray::make(data_size, b'\x00')
  let names : Array[String] = []
  let mut cursor = 8
  for slot in slots {
    if names.search_by(name => name == slot.file_name) is None {
      let encoded = @utf8.encode(slot.file_name)
      data.blit_from_bytes(cursor, encoded, 0, encoded.length())
      cursor = cursor + encoded.length() + 1
      names.push(slot.file_name)
    }
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=32 + data_size)
  writer_write_bytes(buffer, b"HEAP")
  buffer.write_byte(b'\x00')
  writer_write_zeros(buffer, 3)
  buffer.write_uint64_le(data_size.to_uint64())
  buffer.write_uint64_le(1UL)
  buffer.write_uint64_le(address + 32UL)
  writer_write_bytes(buffer, data.unsafe_reinterpret_as_bytes())
  buffer.to_bytes()
}

///|
fn encode_external_data_message(
  slots : Array[Hdf5ExternalSlot],
  heap_address : UInt64,
) -> Bytes raise Hdf5Error {
  let (offsets, _) = external_name_offsets(slots)
  guard slots.length() <= 65535 else {
    raise InvalidArgument("external dataset requires at most 65535 slots")
  }
  let buffer = @buffer.Buffer::Buffer()
  buffer.write_byte(b'\x01')
  writer_write_zeros(buffer, 3)
  buffer.write_uint16_le(slots.length().to_uint16())
  buffer.write_uint16_le(slots.length().to_uint16())
  buffer.write_uint64_le(heap_address)
  for index in 0..<slots.length() {
    buffer.write_uint64_le(offsets[index])
    buffer.write_uint64_le(slots[index].file_offset)
    buffer.write_uint64_le(slots[index].data_size)
  }
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
    buffer.write_byte(b'\x04')
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
fn encode_fill_value(
  fill : Hdf5WriteValues,
  heap~ : WritableGlobalHeap,
  objects~ : Array[WritableObject],
) -> Bytes raise Hdf5Error {
  let encoded = encode_write_values(fill, heap~, objects~)
  let buffer = @buffer.Buffer::Buffer(size_hint=6 + encoded.element_size)
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x22')
  buffer.write_uint_le(encoded.element_size.reinterpret_as_uint())
  writer_write_bytes(buffer, encoded.payload)
  buffer.to_bytes()
}

///|
fn writer_scaleoffset_layout(
  values : Hdf5WriteValues,
) -> (Int32, Bool, Int32) raise Hdf5Error {
  match values {
    Int8Values(_) => (1, true, 0)
    Int16Values(_) => (2, true, 0)
    Int32Values(_) => (4, true, 0)
    Int64Values(_) => (8, true, 0)
    UInt8Values(_) => (1, false, 0)
    UInt16Values(_) => (2, false, 0)
    UInt32Values(_) => (4, false, 0)
    UInt64Values(_) => (8, false, 0)
    Float32Values(_) => (4, false, 1)
    Float64Values(_) => (8, false, 1)
    _ =>
      raise UnsupportedFeature(
        "scale-offset writer filter requires fixed-size numeric values",
      )
  }
}

///|
fn scaleoffset_read_raw(data : Bytes, offset : Int32, size : Int32) -> UInt64 {
  let mut value = 0UL
  for index in 0..<size {
    value = value | (data[offset + index].to_uint64() << (index * 8))
  }
  value
}

///|
fn scaleoffset_bit_mask(size : Int32) -> UInt64 {
  if size == 8 {
    0xFFFFFFFFFFFFFFFFUL
  } else {
    (1UL << (size * 8)) - 1UL
  }
}

///|
fn scaleoffset_ceil_log2(value : UInt64) -> Int32 {
  let mut remaining = value - 1UL
  let mut bits = 0
  while remaining > 0UL {
    bits = bits + 1
    remaining = remaining >> 1
  }
  bits
}

///|
fn scaleoffset_min_max(
  data : Bytes,
  element_size : Int32,
  signed : Bool,
  fill_raw : UInt64,
) -> (UInt64, UInt64) {
  let bits = element_size * 8
  let count = data.length() / element_size
  let mut found = false
  let mut min_raw = 0UL
  let mut max_raw = 0UL
  let mut min_value = 0L
  let mut max_value = 0L
  for index in 0..<count {
    let raw = scaleoffset_read_raw(data, index * element_size, element_size)
    if raw == fill_raw {
      continue
    }
    if signed {
      let value = decode_signed_bits(raw, bits)
      if !found {
        min_raw = raw
        max_raw = raw
        min_value = value
        max_value = value
        found = true
      } else {
        if value < min_value {
          min_value = value
          min_raw = raw
        }
        if value > max_value {
          max_value = value
          max_raw = raw
        }
      }
    } else if !found {
      min_raw = raw
      max_raw = raw
      found = true
    } else {
      if raw < min_raw {
        min_raw = raw
      }
      if raw > max_raw {
        max_raw = raw
      }
    }
  }
  (min_raw, max_raw)
}

///|
fn scaleoffset_client_data(
  scale : Int32,
  datatype_class : Int32,
  element_size : Int32,
  signed : Bool,
  chunk_elements : Int32,
  fill : Bytes,
) -> Array[UInt32] {
  let parameters : Array[UInt32] = Array::make(20, 0U)
  parameters[0] = if datatype_class == 0 { 2U } else { 0U }
  parameters[1] = scale.reinterpret_as_uint()
  parameters[2] = chunk_elements.reinterpret_as_uint()
  parameters[3] = datatype_class.reinterpret_as_uint()
  parameters[4] = element_size.reinterpret_as_uint()
  parameters[5] = if signed { 1U } else { 0U }
  parameters[7] = 1U
  for index in 0..<element_size {
    parameters[8 + index / 4] = parameters[8 + index / 4] |
      (fill[index].to_uint() << (index % 4 * 8))
  }
  parameters
}

///|
fn scaleoffset_round_away(value : Double) -> Int64 {
  let truncated = value.to_int64()
  let fraction = value - truncated.to_double()
  if fraction >= 0.5 {
    truncated + 1L
  } else if fraction <= -0.5 {
    truncated - 1L
  } else {
    truncated
  }
}

///|
fn scaleoffset_finite32(value : Float) -> Bool {
  !value.is_nan() && !value.is_inf()
}

///|
fn scaleoffset_finite64(value : Double) -> Bool {
  !value.is_nan() && !value.is_inf()
}

///|
fn scaleoffset_write_packed_chunk(
  minbits : Int32,
  min_raw : UInt64,
  packed : FixedArray[UInt64],
) -> Bytes raise Hdf5Error {
  let count = packed.length()
  guard minbits >= 0 && minbits < 64 else {
    raise InvalidArgument("invalid scale-offset minimum bits")
  }
  let output_size = 21L + count.to_int64() * minbits.to_int64() / 8L + 1L
  guard output_size <= 0x7FFFFFFFL else {
    raise InvalidField("scale-offset writer chunk is too large")
  }
  let output = FixedArray::make(output_size.to_int(), b'\x00')
  let encoded_bits = minbits.reinterpret_as_uint()
  for index in 0..<4 {
    output[index] = ((encoded_bits >> (index * 8)) & 0xFFU).to_byte()
  }
  output[4] = 8
  for index in 0..<8 {
    output[5 + index] = ((min_raw >> (index * 8)) & 0xFFUL).to_byte()
  }
  let mut bit_index = 0
  for index in 0..<count {
    let value = packed[index]
    let mut remaining = minbits
    while remaining > 0 {
      let byte_index = bit_index / 8
      let bit_offset = bit_index % 8
      let available = 8 - bit_offset
      let take = if remaining < available { remaining } else { available }
      let part = (value >> (remaining - take)) & ((1UL << take) - 1UL)
      output[21 + byte_index] = (output[21 + byte_index].to_uint() |
      (part.to_uint() << (available - take))).to_byte()
      bit_index = bit_index + take
      remaining = remaining - take
    }
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn encode_scaleoffset_float32_chunk(
  data : Bytes,
  digits : Int32,
  fill : Bytes,
) -> Bytes raise Hdf5Error {
  guard data.length() % 4 == 0 && fill.length() == 4 else {
    raise InvalidField("invalid float32 scale-offset writer chunk")
  }
  guard digits >= 0 && digits <= 38 else {
    raise InvalidArgument("invalid float32 scale-offset decimal digits")
  }
  let count = data.length() / 4
  let fill_value = Float::reinterpret_from_uint(fill.unsafe_read_uint32_le(0))
  guard scaleoffset_finite32(fill_value) else {
    raise UnsupportedFeature("scale-offset fill value must be finite")
  }
  let factor = scaleoffset_float_factor32(digits)
  let tolerance = @math.powf(Float(10.0), -Float::from_int(digits))
  guard scaleoffset_finite32(factor) && scaleoffset_finite32(tolerance) else {
    raise UnsupportedFeature("float32 scale-offset factor is out of range")
  }
  let mut found = false
  let mut minimum = Float(0.0)
  let mut maximum = Float(0.0)
  for index in 0..<count {
    let value = Float::reinterpret_from_uint(
      data.unsafe_read_uint32_le(index * 4),
    )
    guard scaleoffset_finite32(value) else {
      raise UnsupportedFeature(
        "scale-offset does not support non-finite floating-point values",
      )
    }
    if Float::abs(value - fill_value) < tolerance {
      continue
    }
    if !found {
      minimum = value
      maximum = value
      found = true
    } else {
      if value < minimum {
        minimum = value
      }
      if value > maximum {
        maximum = value
      }
    }
  }
  let difference = maximum * factor - minimum * factor
  guard scaleoffset_finite32(difference) &&
    difference >= Float(0.0) &&
    difference.to_double() < 9223372036854775808.0 else {
    raise UnsupportedFeature("float32 scale-offset range is out of range")
  }
  let rounded_difference = scaleoffset_round_away(difference.to_double())
  guard rounded_difference >= 0L &&
    rounded_difference.to_double() < 9223372036854775808.0 else {
    raise UnsupportedFeature("float32 scale-offset range is out of range")
  }
  let span = rounded_difference.reinterpret_as_uint64() + 1UL
  let minbits = scaleoffset_ceil_log2(span + 1UL)
  guard minbits < 32 else {
    raise UnsupportedFeature(
      "scale-offset automatic bits require full precision",
    )
  }
  let marker = (1UL << minbits) - 1UL
  let minimum_scaled = minimum * factor
  guard scaleoffset_finite32(minimum_scaled) else {
    raise UnsupportedFeature("float32 scale-offset range is out of range")
  }
  let packed = FixedArray::make(count, 0UL)
  for index in 0..<count {
    let value = Float::reinterpret_from_uint(
      data.unsafe_read_uint32_le(index * 4),
    )
    if Float::abs(value - fill_value) < tolerance {
      packed[index] = marker
    } else {
      let scaled = value * factor - minimum_scaled
      guard scaleoffset_finite32(scaled) &&
        scaled >= Float(0.0) &&
        scaled.to_double() < 9223372036854775808.0 else {
        raise UnsupportedFeature("float32 scale-offset value is out of range")
      }
      let quantized = scaleoffset_round_away(scaled.to_double())
      guard quantized >= 0L && quantized.reinterpret_as_uint64() <= marker else {
        raise InvalidField("float32 scale-offset value is out of range")
      }
      packed[index] = quantized.reinterpret_as_uint64()
    }
  }
  scaleoffset_write_packed_chunk(
    minbits,
    minimum.reinterpret_as_uint().to_uint64(),
    packed,
  )
}

///|
fn encode_scaleoffset_float64_chunk(
  data : Bytes,
  digits : Int32,
  fill : Bytes,
) -> Bytes raise Hdf5Error {
  guard data.length() % 8 == 0 && fill.length() == 8 else {
    raise InvalidField("invalid float64 scale-offset writer chunk")
  }
  guard digits >= 0 && digits <= 308 else {
    raise InvalidArgument("invalid float64 scale-offset decimal digits")
  }
  let count = data.length() / 8
  let fill_value = fill.unsafe_read_uint64_le(0).reinterpret_as_double()
  guard scaleoffset_finite64(fill_value) else {
    raise UnsupportedFeature("scale-offset fill value must be finite")
  }
  let factor = scaleoffset_float_factor64(digits)
  let tolerance = @math.pow(10.0, -Double::from_int(digits))
  guard scaleoffset_finite64(factor) && scaleoffset_finite64(tolerance) else {
    raise UnsupportedFeature("float64 scale-offset factor is out of range")
  }
  let mut found = false
  let mut minimum = 0.0
  let mut maximum = 0.0
  for index in 0..<count {
    let value = data.unsafe_read_uint64_le(index * 8).reinterpret_as_double()
    guard scaleoffset_finite64(value) else {
      raise UnsupportedFeature(
        "scale-offset does not support non-finite floating-point values",
      )
    }
    if Double::abs(value - fill_value) < tolerance {
      continue
    }
    if !found {
      minimum = value
      maximum = value
      found = true
    } else {
      if value < minimum {
        minimum = value
      }
      if value > maximum {
        maximum = value
      }
    }
  }
  let difference = maximum * factor - minimum * factor
  guard scaleoffset_finite64(difference) &&
    difference >= 0.0 &&
    difference < 9223372036854775808.0 else {
    raise UnsupportedFeature("float64 scale-offset range is out of range")
  }
  let rounded_difference = scaleoffset_round_away(difference)
  guard rounded_difference >= 0L &&
    rounded_difference.to_double() < 9223372036854775808.0 else {
    raise UnsupportedFeature("float64 scale-offset range is out of range")
  }
  let span = rounded_difference.reinterpret_as_uint64() + 1UL
  let minbits = scaleoffset_ceil_log2(span + 1UL)
  guard minbits < 64 else {
    raise UnsupportedFeature(
      "scale-offset automatic bits require full precision",
    )
  }
  let marker = (1UL << minbits) - 1UL
  let minimum_scaled = minimum * factor
  guard scaleoffset_finite64(minimum_scaled) else {
    raise UnsupportedFeature("float64 scale-offset range is out of range")
  }
  let packed = FixedArray::make(count, 0UL)
  for index in 0..<count {
    let value = data.unsafe_read_uint64_le(index * 8).reinterpret_as_double()
    if Double::abs(value - fill_value) < tolerance {
      packed[index] = marker
    } else {
      let scaled = value * factor - minimum_scaled
      guard scaleoffset_finite64(scaled) &&
        scaled >= 0.0 &&
        scaled < 9223372036854775808.0 else {
        raise UnsupportedFeature("float64 scale-offset value is out of range")
      }
      let quantized = scaleoffset_round_away(scaled)
      guard quantized >= 0L && quantized.reinterpret_as_uint64() <= marker else {
        raise InvalidField("float64 scale-offset value is out of range")
      }
      packed[index] = quantized.reinterpret_as_uint64()
    }
  }
  scaleoffset_write_packed_chunk(
    minbits,
    minimum.reinterpret_as_uint64(),
    packed,
  )
}

///|
fn encode_scaleoffset_chunk(
  data : Bytes,
  scale : Int32,
  element_size : Int32,
  datatype_class : Int32,
  signed : Bool,
  fill : Bytes,
) -> Bytes raise Hdf5Error {
  guard element_size >= 1 &&
    element_size <= 8 &&
    data.length() % element_size == 0 &&
    fill.length() == element_size else {
    raise InvalidField("invalid scale-offset writer chunk")
  }
  if datatype_class == 1 {
    if element_size == 4 {
      return encode_scaleoffset_float32_chunk(data, scale, fill)
    }
    if element_size == 8 {
      return encode_scaleoffset_float64_chunk(data, scale, fill)
    }
    raise UnsupportedFeature(
      "scale-offset floating-point width is not supported",
    )
  }
  let bits = element_size * 8
  if scale == bits {
    return data
  }
  let count = data.length() / element_size
  let fill_raw = scaleoffset_read_raw(fill, 0, element_size)
  let (min_raw, max_raw) = scaleoffset_min_max(
    data, element_size, signed, fill_raw,
  )
  let value_mask = scaleoffset_bit_mask(element_size)
  let difference = (max_raw - min_raw) & value_mask
  let minbits = if scale == 0 {
    let span = difference + 1UL
    let computed = scaleoffset_ceil_log2(span + 1UL)
    guard computed < bits else {
      raise UnsupportedFeature(
        "scale-offset automatic bits require full precision",
      )
    }
    computed
  } else {
    scale
  }
  guard minbits >= 0 && minbits < bits else {
    raise InvalidArgument("invalid scale-offset minimum bits")
  }
  let output_size = 21L + count.to_int64() * minbits.to_int64() / 8L + 1L
  guard output_size <= 0x7FFFFFFFL else {
    raise InvalidField("scale-offset writer chunk is too large")
  }
  let output = FixedArray::make(output_size.to_int(), b'\x00')
  let encoded_bits = minbits.reinterpret_as_uint()
  for index in 0..<4 {
    output[index] = ((encoded_bits >> (index * 8)) & 0xFFU).to_byte()
  }
  output[4] = 8
  for index in 0..<8 {
    output[5 + index] = ((min_raw >> (index * 8)) & 0xFFUL).to_byte()
  }
  let packed_mask = if minbits == 0 { 0UL } else { (1UL << minbits) - 1UL }
  let mut bit_index = 0
  for index in 0..<count {
    let raw = scaleoffset_read_raw(data, index * element_size, element_size)
    let packed = if raw == fill_raw {
      packed_mask
    } else {
      (raw - min_raw) & value_mask & packed_mask
    }
    let mut remaining = minbits
    while remaining > 0 {
      let byte_index = bit_index / 8
      let bit_offset = bit_index % 8
      let available = 8 - bit_offset
      let take = if remaining < available { remaining } else { available }
      let part = (packed >> (remaining - take)) & ((1UL << take) - 1UL)
      output[21 + byte_index] = (output[21 + byte_index].to_uint() |
      (part.to_uint() << (available - take))).to_byte()
      bit_index = bit_index + take
      remaining = remaining - take
    }
  }
  output.unsafe_reinterpret_as_bytes()
}

///|
fn writer_chunk_element_count(
  chunk_dimensions : Array[UInt32],
) -> Int32 raise Hdf5Error {
  let mut count = 1L
  for dimension in chunk_dimensions {
    guard dimension > 0U && count <= 0x7FFFFFFFL / dimension.to_int64() else {
      raise InvalidField("writer chunk is too large")
    }
    count = count * dimension.to_int64()
  }
  count.to_int()
}

///|
fn encode_filter_pipeline(
  filters : Array[Hdf5WriteFilter],
  element_size : Int32,
  chunk_elements : Int32,
  values : Hdf5WriteValues,
  fill : Bytes,
) -> Bytes raise Hdf5Error {
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
      ScaleOffset(minbits) => {
        let (size, signed, datatype_class) = writer_scaleoffset_layout(values)
        guard size == element_size else {
          raise InvalidField("scale-offset datatype size mismatch")
        }
        let parameters = scaleoffset_client_data(
          minbits, datatype_class, size, signed, chunk_elements, fill,
        )
        buffer.write_uint16_le(6)
        buffer.write_uint16_le(1)
        buffer.write_uint16_le(20)
        for parameter in parameters {
          buffer.write_uint_le(parameter)
        }
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
  filter_mask : UInt32,
  element_size : Int32,
  values : Hdf5WriteValues,
  fill : Bytes,
) -> Bytes raise Hdf5Error {
  let mut output = data
  for index in 0..<filters.length() {
    if (filter_mask & (1U << index)) != 0U {
      continue
    }
    output = match filters[index] {
      Shuffle => shuffle_chunk(output, element_size)
      Deflate(level) => @zlib.zlib(output[:], level~)
      Fletcher32 => add_fletcher32(output)
      ScaleOffset(minbits) => {
        let (size, signed, datatype_class) = writer_scaleoffset_layout(values)
        guard size == element_size else {
          raise InvalidField("scale-offset datatype size mismatch")
        }
        encode_scaleoffset_chunk(
          output, minbits, size, datatype_class, signed, fill,
        )
      }
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
      raise InvalidArgument("writer chunk grid is too large")
    }
    grid.push(count.to_int())
    if count == 0UL {
      total = 0L
    } else {
      guard total <= 0x7FFFFFFFL / count.reinterpret_as_int64() else {
        raise InvalidArgument("writer chunk grid is too large")
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
  objects : Array[WritableObject],
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
    WritableContiguous | WritableExternal(_) =>
      raise InvalidField("expected chunked writer dataset")
  }
  let encoded = encode_write_values(dataset.values, heap~, objects~)
  let fill = match dataset.fill_value {
    Some(value) => encode_write_values(value, heap~, objects~).payload
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
  let always_allocate = writer_values_contain_vlen(dataset.values)
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
  if always_allocate {
    for index in 0..<chunk_count {
      if writer_chunk_intersects_shape(
          index,
          grid,
          order,
          dataset.dimensions,
          chunk_dimensions,
        ) {
        allocated[index] = true
      }
    }
  }
  let entries : Array[EncodedChunkEntry] = []
  for index in 0..<chunk_count {
    let filter_mask = dataset_chunk_filter_mask(dataset, index)
    let data = if allocated[index] {
      encode_chunk_filters(
        chunk_buffers[index].unsafe_reinterpret_as_bytes(),
        filters,
        filter_mask,
        encoded.element_size,
        dataset.values,
        fill,
      )
    } else {
      b""
    }
    entries.push({
      allocated: allocated[index],
      data,
      address: 0UL,
      filter_mask,
    })
  }
  entries
}

///|
fn encode_fixed_array_storage(
  start : UInt64,
  dataset : WritableDataset,
  heap : WritableGlobalHeap,
  objects : Array[WritableObject],
) -> Bytes raise Hdf5Error {
  let (_, filters) = match dataset.storage {
    WritableChunked(dimensions, filters) => (dimensions, filters)
    WritableExtensible(dimensions, filters, _) => (dimensions, filters)
    WritableContiguous | WritableExternal(_) =>
      raise InvalidField("expected chunked writer dataset")
  }
  let entries = encode_dataset_chunks(dataset, heap, objects)
  let filtered = filters.length() > 0
  let entry_size = if filtered { 14 } else { 8 }
  let page_bits = 10
  let page_entries = 1 << page_bits
  let paged = entries.length() > page_entries
  let page_count = if paged {
    (entries.length() + page_entries - 1) / page_entries
  } else {
    1
  }
  let bitmap_size = if paged { (page_count + 7) / 8 } else { 0 }
  let header_length = 28
  let block_address = writer_align(start.to_int() + header_length).to_uint64()
  let block_length = 14 +
    bitmap_size +
    (if paged { 4 + page_count * 4 } else { 4 }) +
    entries.length() * entry_size
  let mut chunk_address = writer_align(block_address.to_int() + block_length)
  for index in 0..<entries.length() {
    if entries[index].allocated {
      entries[index] = {
        allocated: true,
        data: entries[index].data,
        address: chunk_address.to_uint64(),
        filter_mask: entries[index].filter_mask,
      }
      chunk_address = writer_align(chunk_address + entries[index].data.length())
    }
  }

  let header = @buffer.Buffer::Buffer(size_hint=header_length)
  writer_write_bytes(header, b"FAHD")
  header.write_byte(b'\x00')
  header.write_byte(if filtered { b'\x01' } else { b'\x00' })
  header.write_byte(entry_size.to_byte())
  header.write_byte(page_bits.to_byte())
  header.write_uint64_le(entries.length().to_uint64())
  header.write_uint64_le(block_address)
  header.write_uint_le(metadata_checksum(header.to_bytes(), 0, header.length()))

  let block = @buffer.Buffer::Buffer(size_hint=block_length)
  writer_write_bytes(block, b"FADB")
  block.write_byte(b'\x00')
  block.write_byte(if filtered { b'\x01' } else { b'\x00' })
  block.write_uint64_le(start)
  if paged {
    for byte_index in 0..<bitmap_size {
      let remaining = page_count - byte_index * 8
      block.write_byte(
        if remaining >= 8 {
          b'\xff'
        } else {
          (0xFF << (8 - remaining)).to_byte()
        },
      )
    }
    block.write_uint_le(metadata_checksum(block.to_bytes(), 0, block.length()))
  }
  let mut entry_index = 0
  for _ in 0..<page_count {
    let page = @buffer.Buffer::Buffer()
    let page_end = if entry_index + page_entries < entries.length() {
      entry_index + page_entries
    } else {
      entries.length()
    }
    while entry_index < page_end {
      let entry = entries[entry_index]
      page.write_uint64_le(
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
        page.write_uint16_le(entry.data.length().to_uint16())
        page.write_uint_le(entry.filter_mask)
      }
      entry_index = entry_index + 1
    }
    if paged {
      writer_write_bytes(block, page.to_bytes())
      block.write_uint_le(metadata_checksum(page.to_bytes(), 0, page.length()))
    } else {
      writer_write_bytes(block, page.to_bytes())
    }
  }
  if !paged {
    block.write_uint_le(metadata_checksum(block.to_bytes(), 0, block.length()))
  }

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
fn writer_extensible_range_allocated(
  entries : Array[EncodedChunkEntry],
  start : Int64,
  count : Int64,
) -> Bool {
  if start >= entries.length().to_int64() {
    return false
  }
  let end = if start + count < entries.length().to_int64() {
    start + count
  } else {
    entries.length().to_int64()
  }
  for index in start.to_int()..<end.to_int() {
    if entries[index].allocated {
      return true
    }
  }
  false
}

///|
fn writer_write_extensible_entry(
  buffer : @buffer.Buffer,
  entries : Array[EncodedChunkEntry],
  index : Int64,
  filtered : Bool,
) -> Unit raise Hdf5Error {
  let allocated = index < entries.length().to_int64() &&
    entries[index.to_int()].allocated
  buffer.write_uint64_le(
    if allocated {
      entries[index.to_int()].address
    } else {
      0xFFFFFFFFFFFFFFFFUL
    },
  )
  if filtered {
    let size = if allocated { entries[index.to_int()].data.length() } else { 0 }
    guard size <= 65535 else {
      raise InvalidField("filtered writer chunk is too large")
    }
    buffer.write_uint16_le(size.to_uint16())
    buffer.write_uint_le(
      if allocated {
        entries[index.to_int()].filter_mask
      } else {
        0U
      },
    )
  }
}

///|
fn writer_extensible_data_block_length(
  count : Int64,
  entry_size : Int32,
  page_bits : Int32,
) -> Int64 {
  let page_size = 1L << page_bits
  if count <= page_size {
    18L + count * entry_size.to_int64() + 4L
  } else {
    let pages = (count + page_size - 1L) / page_size
    18L + 4L + count * entry_size.to_int64() + pages * 4L
  }
}

///|
fn encode_writer_extensible_data_block(
  header_address : UInt64,
  entries : Array[EncodedChunkEntry],
  entry_start : Int64,
  count : Int64,
  block_offset : UInt64,
  filtered : Bool,
  page_bits : Int32,
) -> Bytes raise Hdf5Error {
  let page_size = 1L << page_bits
  let paged = count > page_size
  let block = @buffer.Buffer::Buffer()
  writer_write_bytes(block, b"EADB")
  block.write_byte(b'\x00')
  block.write_byte(if filtered { b'\x01' } else { b'\x00' })
  block.write_uint64_le(header_address)
  block.write_uint_le(block_offset.to_uint())
  if paged {
    block.write_uint_le(metadata_checksum(block.to_bytes(), 0, block.length()))
  }
  let pages = if paged { (count + page_size - 1L) / page_size } else { 1L }
  for page in 0..<pages.to_int() {
    let page_start = page.to_int64() * page_size
    let page_count = if page_start + page_size < count {
      page_size
    } else {
      count - page_start
    }
    let page_buffer = @buffer.Buffer::Buffer()
    for offset in 0..<page_count.to_int() {
      writer_write_extensible_entry(
        page_buffer,
        entries,
        entry_start + page_start + offset.to_int64(),
        filtered,
      )
    }
    writer_write_bytes(block, page_buffer.to_bytes())
    if paged {
      block.write_uint_le(
        metadata_checksum(page_buffer.to_bytes(), 0, page_buffer.length()),
      )
    }
  }
  if !paged {
    block.write_uint_le(metadata_checksum(block.to_bytes(), 0, block.length()))
  }
  block.to_bytes()
}

///|
fn encode_extensible_array_storage(
  start : UInt64,
  dataset : WritableDataset,
  heap : WritableGlobalHeap,
  objects : Array[WritableObject],
) -> Bytes raise Hdf5Error {
  let (_, filters, _) = match dataset.storage {
    WritableExtensible(dimensions, filters, axis) => (dimensions, filters, axis)
    _ => raise InvalidField("expected extensible writer dataset")
  }
  let entries = encode_dataset_chunks(dataset, heap, objects)
  let filtered = filters.length() > 0
  let entry_size = if filtered { 14 } else { 8 }
  let max_bits = 32
  let index_elements = 4
  let min_elements = 16
  let min_pointers = 4
  let page_bits = 10
  let first_super = 2 * array_log2(min_pointers)
  let levels = 1 + max_bits - array_log2(min_elements)
  let direct_count = 2 * (min_pointers - 1)
  let header_length = 72
  let index_address = start + header_length.to_uint64()
  let index_length = 14 +
    index_elements * entry_size +
    (direct_count + levels - first_super) * 8 +
    4
  let direct_addresses = Array::make(direct_count, 0xFFFFFFFFFFFFFFFFUL)
  let block_addresses : Array[UInt64] = []
  let block_starts : Array[Int64] = []
  let block_counts : Array[Int64] = []
  let block_offsets : Array[UInt64] = []
  let super_addresses = Array::make(levels - first_super, 0xFFFFFFFFFFFFFFFFUL)
  let super_starts : Array[Int64] = []
  let super_counts : Array[Int64] = []
  let super_block_addresses : Array[Array[UInt64]] = []
  let super_bitmaps : Array[Bytes] = []
  let mut secondary_block_count = 0L
  let mut secondary_block_size = 0L
  let mut data_block_count = 0L
  let mut data_block_size = 0L
  let mut realized_elements = index_elements.to_int64()
  let mut metadata_end = writer_align(index_address.to_int() + index_length)
  let mut level_start = 0L
  let mut direct_start = 0
  for level in 0..<first_super {
    let blocks = 1L << (level / 2)
    let count = min_elements.to_int64() << ((level + 1) / 2)
    for block_index in 0..<blocks.to_int() {
      let entry_start = index_elements.to_int64() +
        level_start +
        block_index.to_int64() * count
      if entry_start >= entries.length().to_int64() {
        break
      }
      if writer_extensible_range_allocated(entries, entry_start, count) {
        let block_address = metadata_end.to_uint64()
        direct_addresses[direct_start + block_index] = block_address
        block_addresses.push(block_address)
        block_starts.push(entry_start)
        block_counts.push(count)
        block_offsets.push(
          (level_start +
          (direct_start.to_int64() + block_index.to_int64()) * count).reinterpret_as_uint64(),
        )
        let block_length = writer_extensible_data_block_length(
          count, entry_size, page_bits,
        )
        data_block_count = data_block_count + 1L
        data_block_size = data_block_size + block_length
        realized_elements = realized_elements + count
        metadata_end = writer_align(metadata_end + block_length.to_int())
      }
    }
    level_start = level_start + blocks * count
    direct_start = direct_start + blocks.to_int()
  }
  for level in first_super..<levels {
    if index_elements.to_int64() + level_start >= entries.length().to_int64() {
      break
    }
    let blocks = 1L << (level / 2)
    let count = min_elements.to_int64() << ((level + 1) / 2)
    if writer_extensible_range_allocated(
        entries,
        index_elements.to_int64() + level_start,
        blocks * count,
      ) {
      let page_size = 1L << page_bits
      let pages = if count > page_size { count / page_size } else { 0L }
      let bitmap_bytes = (pages + 7L) / 8L
      let bitmap = Array::make((blocks * bitmap_bytes).to_int(), b'\x00')
      let addresses = Array::make(blocks.to_int(), 0xFFFFFFFFFFFFFFFFUL)
      let super_address = metadata_end.to_uint64()
      super_addresses[level - first_super] = super_address
      let super_length = 18L + blocks * bitmap_bytes + blocks * 8L + 4L
      secondary_block_count = secondary_block_count + 1L
      secondary_block_size = secondary_block_size + super_length
      metadata_end = writer_align(metadata_end + super_length.to_int())
      for block_index in 0..<blocks.to_int() {
        let entry_start = index_elements.to_int64() +
          level_start +
          block_index.to_int64() * count
        if writer_extensible_range_allocated(entries, entry_start, count) {
          addresses[block_index] = metadata_end.to_uint64()
          if pages > 0L {
            for page in 0..<pages.to_int() {
              if writer_extensible_range_allocated(
                  entries,
                  entry_start + page.to_int64() * page_size,
                  page_size,
                ) {
                let bit = block_index.to_int64() * pages + page.to_int64()
                let byte_index = (bit / 8L).to_int()
                bitmap[byte_index] = (bitmap[byte_index].to_int() |
                (0x80 >> (bit % 8L).to_int())).to_byte()
              }
            }
          }
          let block_length = writer_extensible_data_block_length(
            count, entry_size, page_bits,
          )
          data_block_count = data_block_count + 1L
          data_block_size = data_block_size + block_length
          realized_elements = realized_elements + count
          metadata_end = writer_align(metadata_end + block_length.to_int())
        }
      }
      super_starts.push(level_start)
      super_counts.push(count)
      super_block_addresses.push(addresses)
      super_bitmaps.push(Bytes::from_array(bitmap))
    }
    level_start = level_start + blocks * count
  }
  guard entries.length().to_int64() <= index_elements.to_int64() + level_start else {
    raise InvalidField(
      "writer extensible-array index exceeds configured capacity",
    )
  }
  let mut chunk_address = metadata_end
  for index in 0..<entries.length() {
    if entries[index].allocated {
      entries[index] = {
        allocated: true,
        data: entries[index].data,
        address: chunk_address.to_uint64(),
        filter_mask: entries[index].filter_mask,
      }
      chunk_address = writer_align(chunk_address + entries[index].data.length())
    }
  }

  let header = @buffer.Buffer::Buffer(size_hint=header_length)
  writer_write_bytes(header, b"EAHD")
  header.write_byte(b'\x00')
  header.write_byte(if filtered { b'\x01' } else { b'\x00' })
  header.write_byte(entry_size.to_byte())
  header.write_byte(max_bits.to_byte())
  header.write_byte(index_elements.to_byte())
  header.write_byte(min_elements.to_byte())
  header.write_byte(min_pointers.to_byte())
  header.write_byte(page_bits.to_byte())
  let mut maximum_set = 0UL
  for index in 0..<entries.length() {
    if entries[index].allocated {
      maximum_set = (index + 1).to_uint64()
    }
  }
  header.write_uint64_le(secondary_block_count.reinterpret_as_uint64())
  header.write_uint64_le(secondary_block_size.reinterpret_as_uint64())
  header.write_uint64_le(data_block_count.reinterpret_as_uint64())
  header.write_uint64_le(data_block_size.reinterpret_as_uint64())
  header.write_uint64_le(maximum_set)
  header.write_uint64_le(realized_elements.reinterpret_as_uint64())
  header.write_uint64_le(index_address)
  header.write_uint_le(metadata_checksum(header.to_bytes(), 0, header.length()))

  let index = @buffer.Buffer::Buffer(size_hint=index_length)
  writer_write_bytes(index, b"EAIB")
  index.write_byte(b'\x00')
  index.write_byte(if filtered { b'\x01' } else { b'\x00' })
  index.write_uint64_le(start)
  for entry_index in 0..<index_elements {
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
      index.write_uint_le(
        if allocated {
          entries[entry_index].filter_mask
        } else {
          0U
        },
      )
    }
  }
  for address in direct_addresses {
    index.write_uint64_le(address)
  }
  for address in super_addresses {
    index.write_uint64_le(address)
  }
  index.write_uint_le(metadata_checksum(index.to_bytes(), 0, index.length()))

  let output = @buffer.Buffer::Buffer(size_hint=chunk_address - start.to_int())
  writer_write_bytes(output, header.to_bytes())
  writer_write_bytes(output, index.to_bytes())
  for block_index in 0..<block_addresses.length() {
    writer_write_zeros(
      output,
      block_addresses[block_index].to_int() - start.to_int() - output.length(),
    )
    writer_write_bytes(
      output,
      encode_writer_extensible_data_block(
        start,
        entries,
        block_starts[block_index],
        block_counts[block_index],
        block_offsets[block_index],
        filtered,
        page_bits,
      ),
    )
  }
  let mut planned_index = 0
  for super_index in 0..<super_addresses.length() {
    if super_addresses[super_index] == 0xFFFFFFFFFFFFFFFFUL {
      continue
    }
    writer_write_zeros(
      output,
      super_addresses[super_index].to_int() - start.to_int() - output.length(),
    )
    let super_block = @buffer.Buffer::Buffer()
    writer_write_bytes(super_block, b"EASB")
    super_block.write_byte(b'\x00')
    super_block.write_byte(if filtered { b'\x01' } else { b'\x00' })
    super_block.write_uint64_le(start)
    super_block.write_uint_le(
      super_starts[planned_index].reinterpret_as_uint64().to_uint(),
    )
    writer_write_bytes(super_block, super_bitmaps[planned_index])
    for address in super_block_addresses[planned_index] {
      super_block.write_uint64_le(address)
    }
    super_block.write_uint_le(
      metadata_checksum(super_block.to_bytes(), 0, super_block.length()),
    )
    writer_write_bytes(output, super_block.to_bytes())
    for block_index in 0..<super_block_addresses[planned_index].length() {
      let address = super_block_addresses[planned_index][block_index]
      if address == 0xFFFFFFFFFFFFFFFFUL {
        continue
      }
      writer_write_zeros(
        output,
        address.to_int() - start.to_int() - output.length(),
      )
      writer_write_bytes(
        output,
        encode_writer_extensible_data_block(
          start,
          entries,
          index_elements.to_int64() +
          super_starts[planned_index] +
          block_index.to_int64() * super_counts[planned_index],
          super_counts[planned_index],
          (super_starts[planned_index] +
          block_index.to_int64() * super_counts[planned_index]).reinterpret_as_uint64(),
          filtered,
          page_bits,
        ),
      )
    }
    planned_index = planned_index + 1
  }
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
fn encode_hard_link(
  name : String,
  address : UInt64,
  creation_order : UInt64,
  track_order : Bool,
) -> Bytes {
  let encoded = @utf8.encode(name)
  let buffer = @buffer.Buffer::Buffer(size_hint=20 + encoded.length())
  buffer.write_byte(b'\x01')
  buffer.write_byte(if track_order { b'\x15' } else { b'\x11' })
  if track_order {
    buffer.write_uint64_le(creation_order)
  }
  buffer.write_byte(b'\x01')
  buffer.write_uint16_le(encoded.length().to_uint16())
  writer_write_bytes(buffer, encoded)
  buffer.write_uint64_le(address)
  buffer.to_bytes()
}

///|
fn encode_soft_link(
  name : String,
  target : String,
  creation_order : UInt64,
  track_order : Bool,
) -> Bytes {
  let encoded_name = @utf8.encode(name)
  let encoded_target = @utf8.encode(target)
  let buffer = @buffer.Buffer::Buffer(
    size_hint=8 + encoded_name.length() + encoded_target.length(),
  )
  buffer.write_byte(b'\x01')
  buffer.write_byte(if track_order { b'\x1D' } else { b'\x19' })
  buffer.write_byte(b'\x01')
  if track_order {
    buffer.write_uint64_le(creation_order)
  }
  buffer.write_byte(b'\x01')
  buffer.write_uint16_le(encoded_name.length().to_uint16())
  writer_write_bytes(buffer, encoded_name)
  buffer.write_uint16_le(encoded_target.length().to_uint16())
  writer_write_bytes(buffer, encoded_target)
  buffer.to_bytes()
}

///|
fn encode_external_link(
  name : String,
  target_file : String,
  target_path : String,
  creation_order : UInt64,
  track_order : Bool,
) -> Bytes {
  let encoded_name = @utf8.encode(name)
  let encoded_file = @utf8.encode(target_file)
  let encoded_path = @utf8.encode(target_path)
  let info_size = 1 + encoded_file.length() + 1 + encoded_path.length() + 1
  let buffer = @buffer.Buffer::Buffer(
    size_hint=8 + encoded_name.length() + info_size,
  )
  buffer.write_byte(b'\x01')
  buffer.write_byte(if track_order { b'\x1D' } else { b'\x19' })
  buffer.write_byte(b'\x40')
  if track_order {
    buffer.write_uint64_le(creation_order)
  }
  buffer.write_byte(b'\x01')
  buffer.write_uint16_le(encoded_name.length().to_uint16())
  writer_write_bytes(buffer, encoded_name)
  buffer.write_uint16_le(info_size.to_uint16())
  buffer.write_byte(b'\x00')
  writer_write_bytes(buffer, encoded_file)
  buffer.write_byte(b'\x00')
  writer_write_bytes(buffer, encoded_path)
  buffer.write_byte(b'\x00')
  buffer.to_bytes()
}

///|
fn encode_attribute(
  attribute : WritableAttribute,
  heap : WritableGlobalHeap,
  objects : Array[WritableObject],
) -> Bytes raise Hdf5Error {
  let encoded = encode_write_values(attribute.values, heap~, objects~)
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
fn encode_compact_attribute_info() -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=18)
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x00')
  buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  buffer.to_bytes()
}

///|
let writer_dense_threshold : Int32 = 8

///|
let writer_dense_max_managed_size : Int32 = 65535

///|
let writer_dense_max_records : Int32 = 65535

///|
fn encode_dense_storage_info(
  group : Bool,
  max_creation_order : UInt64,
  heap_address : UInt64,
  index_address : UInt64,
  order_index_address : UInt64,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=28)
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x03')
  if group {
    buffer.write_uint64_le(max_creation_order)
  } else {
    buffer.write_uint16_le(max_creation_order.to_uint16())
  }
  buffer.write_uint64_le(heap_address)
  buffer.write_uint64_le(index_address)
  buffer.write_uint64_le(order_index_address)
  buffer.to_bytes()
}

///|
fn writer_dense_max_creation_order(
  entries : Array[WritableDenseHeapEntry],
) -> UInt64 {
  let mut maximum = 0UL
  for entry in entries {
    if entry.creation_order > maximum {
      maximum = entry.creation_order
    }
  }
  maximum
}

///|
fn writer_name_hash(name : String) -> UInt32 {
  let encoded = @utf8.encode(name)
  metadata_checksum(encoded, 0, encoded.length())
}

///|
fn Hdf5Writer::dense_link_entries(
  self : Hdf5Writer,
  object : WritableObject,
  objects : Array[WritableObject],
  track_order~ : Bool,
) -> Array[WritableDenseHeapEntry] raise Hdf5Error {
  let entries : Array[WritableDenseHeapEntry] = []
  for child in objects {
    if child.path != "/" && writer_parent_path(child.path) == object.path {
      let name = writer_basename(child.path)
      entries.push({
        name,
        hash: writer_name_hash(name),
        data: encode_hard_link(
          name,
          child.address,
          self.link_creation_order(child.path),
          track_order,
        ),
        creation_order: self.link_creation_order(child.path),
        flags: b'\x00',
        shared_id: None,
      })
    }
  }
  for link in self.soft_links {
    if link.parent_path == object.path {
      entries.push({
        name: link.name,
        hash: writer_name_hash(link.name),
        data: encode_soft_link(
          link.name,
          link.target,
          self.link_creation_order(
            writer_soft_link_path(link.parent_path, link.name),
          ),
          track_order,
        ),
        creation_order: self.link_creation_order(
          writer_soft_link_path(link.parent_path, link.name),
        ),
        flags: b'\x00',
        shared_id: None,
      })
    }
  }
  for link in self.external_links {
    if link.parent_path == object.path {
      entries.push({
        name: link.name,
        hash: writer_name_hash(link.name),
        data: encode_external_link(
          link.name,
          link.target_file,
          link.target_path,
          self.link_creation_order(
            writer_soft_link_path(link.parent_path, link.name),
          ),
          track_order,
        ),
        creation_order: self.link_creation_order(
          writer_soft_link_path(link.parent_path, link.name),
        ),
        flags: b'\x00',
        shared_id: None,
      })
    }
  }
  entries
}

///|
fn writer_shared_attribute(
  attributes : Array[WritableSharedAttribute],
  path : String,
  name : String,
) -> WritableSharedAttribute? {
  match
    attributes.search_by(attribute => {
      attribute.path == path && attribute.name == name
    }) {
    Some(index) => Some(attributes[index])
    None => None
  }
}

///|
fn encode_writer_shared_message(heap_id : Bytes) -> Bytes raise Hdf5Error {
  guard heap_id.length() == 8 else {
    raise InvalidField("writer shared-message heap ID has an invalid size")
  }
  let buffer = @buffer.Buffer::Buffer(size_hint=10)
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x01')
  writer_write_bytes(buffer, heap_id)
  buffer.to_bytes()
}

///|
fn Hdf5Writer::dense_attribute_entries(
  self : Hdf5Writer,
  path : String,
  heap : WritableGlobalHeap,
  objects : Array[WritableObject],
  shared~ : Array[WritableSharedAttribute] = [],
) -> Array[WritableDenseHeapEntry] raise Hdf5Error {
  let entries : Array[WritableDenseHeapEntry] = []
  for attribute in self.object_attributes(path) {
    let (data, flags, shared_id) = match
      writer_shared_attribute(shared, path, attribute.name) {
      Some(shared_attribute) =>
        (
          encode_writer_shared_message(shared_attribute.id),
          shared_message_flag.to_byte(),
          Some(shared_attribute.id),
        )
      None => (encode_attribute(attribute, heap, objects), b'\x00', None)
    }
    entries.push({
      name: attribute.name,
      hash: writer_name_hash(attribute.name),
      data,
      creation_order: attribute.creation_order,
      flags,
      shared_id,
    })
  }
  entries
}

///|
fn writer_dense_entries_fit(entries : Array[WritableDenseHeapEntry]) -> Bool {
  if entries.length() <= writer_dense_threshold ||
    entries.length() > writer_dense_max_records {
    return false
  }
  for entry in entries {
    if entry.data.length() > writer_dense_max_managed_size {
      return false
    }
  }
  true
}

///|
fn writer_sort_dense_entries(entries : Array[WritableDenseHeapEntry]) -> Unit {
  entries.sort_by((left, right) => {
    if left.hash < right.hash {
      -1
    } else if left.hash > right.hash {
      1
    } else if left.name < right.name {
      -1
    } else if left.name > right.name {
      1
    } else {
      0
    }
  })
}

///|
fn writer_next_power_of_two(
  value : Int32,
  minimum : Int32,
) -> Int32 raise Hdf5Error {
  let mut result = minimum
  while result < value {
    guard result <= 0x3FFFFFFF else {
      raise InvalidField("writer dense metadata is too large")
    }
    result = result * 2
  }
  result
}

///|
fn encode_writer_dense_heap_header(
  root_address : UInt64,
  id_length : Int32,
  start_block_size : Int32,
  managed_size : Int32,
  max_direct_size : Int32,
  max_heap_bits : Int32,
  record_count : Int32,
  max_managed_size : Int32,
  table_width : Int32,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=146)
  writer_write_bytes(buffer, b"FRHP")
  buffer.write_byte(b'\x00')
  buffer.write_uint16_le(id_length.to_uint16())
  buffer.write_uint16_le(0)
  buffer.write_byte(b'\x02')
  buffer.write_uint_le(max_managed_size.reinterpret_as_uint())
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  buffer.write_uint64_le(managed_size.to_uint64())
  buffer.write_uint64_le(managed_size.to_uint64())
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(record_count.to_uint64())
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(0UL)
  buffer.write_uint16_le(table_width.to_uint16())
  buffer.write_uint64_le(start_block_size.to_uint64())
  buffer.write_uint64_le(max_direct_size.to_uint64())
  buffer.write_uint16_le(max_heap_bits.to_uint16())
  buffer.write_uint16_le(1)
  buffer.write_uint64_le(root_address)
  buffer.write_uint16_le(0)
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  buffer.to_bytes()
}

///|
fn patched_zeroed_checksum(bytes : Bytes, checksum_offset : Int32) -> Bytes {
  let checksum = checksum_lookup3(
    bytes,
    0,
    bytes.length(),
    checksum_offset,
    metadata_checksum_size,
    0U,
  )
  let output = @buffer.Buffer::Buffer(size_hint=bytes.length())
  output.write_bytes(bytes.get_view(start=0, end=checksum_offset).unwrap())
  output.write_uint_le(checksum)
  output.write_bytes(
    bytes
    .get_view(
      start=checksum_offset + metadata_checksum_size,
      end=bytes.length(),
    )
    .unwrap(),
  )
  output.to_bytes()
}

///|
fn encode_writer_dense_direct_block(
  heap_address : UInt64,
  block_size : Int32,
  heap_offset_size : Int32,
  entries : Array[WritableDenseHeapEntry],
) -> (Bytes, Array[Bytes]) raise Hdf5Error {
  let buffer = @buffer.Buffer::Buffer(size_hint=block_size)
  writer_write_bytes(buffer, b"FHDB")
  buffer.write_byte(b'\x00')
  buffer.write_uint64_le(heap_address)
  write_uint_width(buffer, 0UL, heap_offset_size)
  let checksum_offset = buffer.length()
  buffer.write_uint_le(0U)
  let ids : Array[Bytes] = []
  for entry in entries {
    match entry.shared_id {
      Some(id) => {
        guard id.length() == 8 else {
          raise InvalidField(
            "writer shared-message heap ID has an invalid size",
          )
        }
        ids.push(id)
      }
      None => {
        let offset = buffer.length()
        guard offset.to_uint64() <= 0xFFFFFFFFUL &&
          entry.data.length().to_uint64() <= 0xFFFFUL else {
          raise InvalidField("writer dense heap object exceeds its identifier")
        }
        let id = @buffer.Buffer::Buffer(size_hint=1 + heap_offset_size + 2)
        id.write_byte(b'\x00')
        write_uint_width(id, offset.to_uint64(), heap_offset_size)
        id.write_uint16_le(entry.data.length().to_uint16())
        ids.push(id.to_bytes())
        writer_write_bytes(buffer, entry.data)
      }
    }
  }
  guard buffer.length() <= block_size else {
    raise InvalidField("writer dense heap block is too small")
  }
  writer_write_zeros(buffer, block_size - buffer.length())
  (patched_zeroed_checksum(buffer.to_bytes(), checksum_offset), ids)
}

///|
fn encode_writer_dense_shared_ids(
  entries : Array[WritableDenseHeapEntry],
) -> Array[Bytes] raise Hdf5Error {
  let ids : Array[Bytes] = []
  for entry in entries {
    match entry.shared_id {
      Some(id) => {
        guard id.length() == 8 else {
          raise InvalidField(
            "writer shared-message heap ID has an invalid size",
          )
        }
        ids.push(id)
      }
      None => raise InvalidField("writer dense heap has an unallocated object")
    }
  }
  ids
}

///|
fn encode_writer_dense_btree_header(
  node_type : Int32,
  node_size : Int32,
  record_size : Int32,
  root_address : UInt64,
  record_count : Int32,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=38)
  writer_write_bytes(buffer, b"BTHD")
  buffer.write_byte(b'\x00')
  buffer.write_byte(node_type.to_byte())
  buffer.write_uint_le(node_size.reinterpret_as_uint())
  buffer.write_uint16_le(record_size.to_uint16())
  buffer.write_uint16_le(0)
  buffer.write_byte(b'\x64')
  buffer.write_byte(b'\x28')
  buffer.write_uint64_le(root_address)
  buffer.write_uint16_le(record_count.to_uint16())
  buffer.write_uint64_le(record_count.to_uint64())
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  buffer.to_bytes()
}

///|
fn encode_writer_dense_btree_node(
  node_type : Int32,
  node_size : Int32,
  record_size : Int32,
  records : Array[Bytes],
) -> Bytes raise Hdf5Error {
  let buffer = @buffer.Buffer::Buffer(size_hint=node_size)
  writer_write_bytes(buffer, b"BTLF")
  buffer.write_byte(b'\x00')
  buffer.write_byte(node_type.to_byte())
  for record in records {
    guard record.length() == record_size else {
      raise InvalidField("writer dense B-tree record size mismatch")
    }
    writer_write_bytes(buffer, record)
  }
  guard buffer.length() + metadata_checksum_size <= node_size else {
    raise InvalidField("writer dense B-tree node is too small")
  }
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  writer_write_zeros(buffer, node_size - buffer.length())
  buffer.to_bytes()
}

///|
let writer_shared_max_managed_size : Int32 = 4096

///|
let writer_shared_table_size : Int32 = 38

///|
fn Hdf5Writer::shared_attribute_candidates(
  self : Hdf5Writer,
  objects : Array[WritableObject],
  heap : WritableGlobalHeap,
) -> Array[WritableSharedAttribute] raise Hdf5Error {
  let attributes : Array[WritableSharedAttribute] = []
  if !self.share_attributes {
    return attributes
  }
  let empty_id = Bytes::make(8, b'\x00')
  for object in objects {
    for attribute in self.object_attributes(object.path) {
      let data = encode_attribute(attribute, heap, objects)
      if data.length() <= writer_shared_max_managed_size {
        attributes.push({
          path: object.path,
          name: attribute.name,
          data,
          id: empty_id,
        })
      }
    }
  }
  guard attributes.length() <= 65535 else {
    raise InvalidArgument("writer has too many shared attributes")
  }
  attributes
}

///|
fn encode_writer_shared_heap(
  heap_address : UInt64,
  attributes : Array[WritableSharedAttribute],
) -> (Bytes, Array[Bytes]) raise Hdf5Error {
  guard attributes.length() > 0 else { return (b"", []) }
  let heap_offset_size = 5
  let direct_overhead = 4 + 1 + 8 + heap_offset_size + metadata_checksum_size
  let mut data_size = direct_overhead
  for attribute in attributes {
    guard attribute.data.length() <= writer_shared_max_managed_size else {
      raise InvalidField("writer shared attribute exceeds its managed heap")
    }
    data_size = data_size + attribute.data.length()
  }
  let block_size = writer_next_power_of_two(data_size, 1024)
  let max_direct_size = writer_next_power_of_two(block_size, 65536)
  let root_address = heap_address + 146UL
  let header = encode_writer_dense_heap_header(
    root_address,
    8,
    block_size,
    block_size,
    max_direct_size,
    40,
    attributes.length(),
    writer_shared_max_managed_size,
    4,
  )
  let entries = attributes.map(attribute => {
    name: attribute.name,
    hash: 0U,
    data: attribute.data,
    creation_order: 0UL,
    flags: b'\x00',
    shared_id: None,
  })
  let (direct_block, ids) = encode_writer_dense_direct_block(
    heap_address, block_size, heap_offset_size, entries,
  )
  let buffer = @buffer.Buffer::Buffer(
    size_hint=header.length() + direct_block.length(),
  )
  writer_write_bytes(buffer, header)
  writer_write_bytes(buffer, direct_block)
  (buffer.to_bytes(), ids)
}

///|
fn encode_writer_shared_table(heap_address : UInt64, count : Int32) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=writer_shared_table_size)
  writer_write_bytes(buffer, b"SMTB")
  buffer.write_byte(b'\x00')
  buffer.write_byte(b'\x00')
  buffer.write_uint16_le(0x1000U.to_uint16())
  buffer.write_uint_le(0U)
  buffer.write_uint16_le(65535U.to_uint16())
  buffer.write_uint16_le(0U.to_uint16())
  buffer.write_uint16_le(count.to_uint16())
  buffer.write_uint64_le(0xFFFFFFFFFFFFFFFFUL)
  buffer.write_uint64_le(heap_address)
  buffer.write_uint_le(metadata_checksum(buffer.to_bytes(), 0, buffer.length()))
  buffer.to_bytes()
}

///|
fn encode_writer_superblock_extension(table_address : UInt64) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=40)
  buffer.write_byte(b'\x01')
  buffer.write_byte(b'\x00')
  buffer.write_uint16_le(1U.to_uint16())
  buffer.write_uint_le(1U)
  buffer.write_uint_le(24U)
  buffer.write_uint_le(0U)
  buffer.write_uint16_le(0x000FU.to_uint16())
  buffer.write_uint16_le(16U.to_uint16())
  buffer.write_byte(b'\x00')
  writer_write_zeros(buffer, 3)
  buffer.write_byte(b'\x00')
  buffer.write_uint64_le(table_address)
  buffer.write_byte(b'\x01')
  writer_write_zeros(buffer, 6)
  buffer.to_bytes()
}

///|
fn Hdf5Writer::encode_dense_metadata(
  self : Hdf5Writer,
  object : WritableObject,
  objects : Array[WritableObject],
  heap : WritableGlobalHeap,
  heap_address : UInt64,
  kind : WritableDenseKind,
  shared~ : Array[WritableSharedAttribute] = [],
) -> (Bytes, Bytes, Bytes) raise Hdf5Error {
  let group = match kind {
    WritableDenseLinks => true
    WritableDenseAttributes => false
  }
  let entries = if group {
    self.dense_link_entries(object, objects, track_order=true)
  } else {
    self.dense_attribute_entries(object.path, heap, objects, shared~)
  }
  guard writer_dense_entries_fit(entries) else {
    raise InvalidField("dense metadata is not available for this object")
  }
  writer_sort_dense_entries(entries)
  let record_size = if group { 11 } else { 17 }
  let heap_offset_size = if group { 4 } else { 5 }
  let max_heap_bits = if group { 32 } else { 40 }
  let id_length = if group { 7 } else { 8 }
  let minimum_block_size = if group { 512 } else { 1024 }
  let direct_overhead = 4 + 1 + 8 + heap_offset_size + metadata_checksum_size
  let mut data_size = direct_overhead
  let mut local_count = 0
  for entry in entries {
    if entry.shared_id is None {
      local_count = local_count + 1
      data_size = data_size + entry.data.length()
    }
  }
  let block_size = writer_next_power_of_two(data_size, minimum_block_size)
  let max_direct_size = writer_next_power_of_two(block_size, 65536)
  let root_address = if local_count > 0 {
    heap_address + 146UL
  } else {
    0xFFFFFFFFFFFFFFFFUL
  }
  let index_address = if local_count > 0 {
    heap_address + 146UL + block_size.to_uint64()
  } else {
    heap_address + 146UL
  }
  let heap_header = encode_writer_dense_heap_header(
    root_address,
    id_length,
    block_size,
    if local_count > 0 {
      block_size
    } else {
      0
    },
    max_direct_size,
    max_heap_bits,
    local_count,
    writer_dense_max_managed_size,
    4,
  )
  let (direct_block, ids) = if local_count > 0 {
    encode_writer_dense_direct_block(
      heap_address, block_size, heap_offset_size, entries,
    )
  } else {
    (b"", encode_writer_dense_shared_ids(entries))
  }
  let records : Array[Bytes] = []
  for index in 0..<entries.length() {
    let record = @buffer.Buffer::Buffer(size_hint=record_size)
    if group {
      record.write_uint_le(entries[index].hash)
      writer_write_bytes(record, ids[index])
    } else {
      writer_write_bytes(record, ids[index])
      record.write_byte(entries[index].flags)
      record.write_uint_le(entries[index].creation_order.to_uint())
      record.write_uint_le(entries[index].hash)
    }
    guard record.length() == record_size else {
      raise InvalidField("writer dense B-tree record size mismatch")
    }
    records.push(record.to_bytes())
  }
  let order_indices : Array[Int32] = []
  for index in 0..<entries.length() {
    order_indices.push(index)
  }
  order_indices.sort_by((left, right) => {
    let left_order = entries[left].creation_order
    let right_order = entries[right].creation_order
    if left_order < right_order {
      -1
    } else if left_order > right_order {
      1
    } else if entries[left].name < entries[right].name {
      -1
    } else if entries[left].name > entries[right].name {
      1
    } else {
      0
    }
  })
  let order_record_size = if group { 15 } else { 13 }
  let order_records : Array[Bytes] = []
  for index in order_indices {
    let order_record = @buffer.Buffer::Buffer(size_hint=order_record_size)
    if group {
      order_record.write_uint64_le(entries[index].creation_order)
      writer_write_bytes(order_record, ids[index])
    } else {
      writer_write_bytes(order_record, ids[index])
      order_record.write_byte(entries[index].flags)
      order_record.write_uint_le(entries[index].creation_order.to_uint())
    }
    guard order_record.length() == order_record_size else {
      raise InvalidField("writer dense order B-tree record size mismatch")
    }
    order_records.push(order_record.to_bytes())
  }
  let node_size = writer_next_power_of_two(
    10 + records.length() * record_size,
    512,
  )
  let node_address = index_address + 38UL
  let btree_header = encode_writer_dense_btree_header(
    if group {
      5
    } else {
      8
    },
    node_size,
    record_size,
    node_address,
    records.length(),
  )
  let btree_node = encode_writer_dense_btree_node(
    if group {
      5
    } else {
      8
    },
    node_size,
    record_size,
    records,
  )
  let heap_bytes = @buffer.Buffer::Buffer(
    size_hint=heap_header.length() + direct_block.length(),
  )
  writer_write_bytes(heap_bytes, heap_header)
  writer_write_bytes(heap_bytes, direct_block)
  let index_bytes = @buffer.Buffer::Buffer(
    size_hint=btree_header.length() + btree_node.length(),
  )
  writer_write_bytes(index_bytes, btree_header)
  writer_write_bytes(index_bytes, btree_node)
  let order_index_address = index_address + index_bytes.length().to_uint64()
  let order_node_size = writer_next_power_of_two(
    10 + order_records.length() * order_record_size,
    512,
  )
  let order_node_address = order_index_address + 38UL
  let order_header = encode_writer_dense_btree_header(
    if group {
      6
    } else {
      9
    },
    order_node_size,
    order_record_size,
    order_node_address,
    order_records.length(),
  )
  let order_node = encode_writer_dense_btree_node(
    if group {
      6
    } else {
      9
    },
    order_node_size,
    order_record_size,
    order_records,
  )
  let order_bytes = @buffer.Buffer::Buffer(
    size_hint=order_header.length() + order_node.length(),
  )
  writer_write_bytes(order_bytes, order_header)
  writer_write_bytes(order_bytes, order_node)
  (heap_bytes.to_bytes(), index_bytes.to_bytes(), order_bytes.to_bytes())
}

///|
fn writer_dense_addresses(
  dense : Array[WritableDenseMetadata],
  path : String,
  kind : WritableDenseKind,
) -> (UInt64, UInt64, UInt64) {
  match
    dense.search_by(metadata => metadata.path == path && metadata.kind == kind) {
    Some(index) =>
      (
        dense[index].heap_address,
        dense[index].index_address,
        dense[index].order_index_address,
      )
    None => (0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL)
  }
}

///|
fn writer_external_heap_address(
  external : Array[WritableExternalHeap],
  path : String,
) -> UInt64 {
  match external.search_by(heap => heap.path == path) {
    Some(index) => external[index].address
    None => 0xFFFFFFFFFFFFFFFFUL
  }
}

///|
fn write_header_message(
  buffer : @buffer.Buffer,
  message_type : Int32,
  data : Bytes,
) -> Unit raise Hdf5Error {
  write_header_message_with_order(
    buffer,
    message_type,
    data,
    creation_order=false,
  )
}

///|
fn write_header_message_with_order(
  buffer : @buffer.Buffer,
  message_type : Int32,
  data : Bytes,
  creation_order~ : Bool,
) -> Unit raise Hdf5Error {
  guard data.length() <= 65535 else {
    raise InvalidField("object header message is too large")
  }
  buffer.write_byte(message_type.to_byte())
  buffer.write_uint16_le(data.length().to_uint16())
  buffer.write_byte(b'\x00')
  if creation_order {
    buffer.write_uint16_le(0)
  }
  writer_write_bytes(buffer, data)
}

///|
fn write_header_message_with_flags(
  buffer : @buffer.Buffer,
  message_type : Int32,
  data : Bytes,
  flags : Byte,
) -> Unit raise Hdf5Error {
  guard data.length() <= 65535 else {
    raise InvalidField("object header message is too large")
  }
  buffer.write_byte(message_type.to_byte())
  buffer.write_uint16_le(data.length().to_uint16())
  buffer.write_byte(flags)
  writer_write_bytes(buffer, data)
}

///|
fn split_object_header_messages(
  payload : Bytes,
) -> Array[Bytes] raise Hdf5Error {
  let messages : Array[Bytes] = []
  let reader = Reader::new(payload)
  while reader.remaining() > 0 {
    let start = reader.position()
    guard reader.remaining() >= 4 else {
      raise InvalidField("incomplete writer object header message")
    }
    let _message_type = reader.read_u8()
    let size = reader.read_u16_le().to_int()
    let _flags = reader.read_u8()
    reader.skip(size)
    messages.push(
      payload.get_view(start~, end=reader.position()).unwrap().to_owned(),
    )
  }
  messages
}

///|
fn message_with_creation_order(message : Bytes) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=message.length() + 2)
  writer_write_bytes(
    buffer,
    message.get_view(start=0, end=4).unwrap().to_owned(),
  )
  buffer.write_uint16_le(0)
  writer_write_bytes(
    buffer,
    message.get_view(start=4, end=message.length()).unwrap().to_owned(),
  )
  buffer.to_bytes()
}

///|
fn object_header_messages_size(messages : Array[Bytes]) -> Int32 {
  let mut size = 0
  for message in messages {
    size = size + message.length()
  }
  size
}

///|
fn plan_object_header_chunks(messages : Array[Bytes]) -> Array[Array[Bytes]] {
  let chunks : Array[Array[Bytes]] = []
  let mut index = 0
  let mut initial = true
  while index < messages.length() {
    let capacity = if initial { 65535 } else { 1024 * 1024 }
    let mut remaining_size = 0
    for position in index..<messages.length() {
      remaining_size = remaining_size + messages[position].length()
    }
    let chunk : Array[Bytes] = []
    if remaining_size <= capacity {
      for position in index..<messages.length() {
        chunk.push(messages[position])
      }
      index = messages.length()
    } else {
      let usable = capacity - 20
      let mut used = 0
      while index < messages.length() &&
            used + messages[index].length() <= usable {
        used = used + messages[index].length()
        chunk.push(messages[index])
        index = index + 1
      }
    }
    chunks.push(chunk)
    initial = false
  }
  if chunks.length() == 0 {
    chunks.push([])
  }
  chunks
}

///|
fn encode_object_continuation(address : UInt64, length : UInt64) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=16)
  buffer.write_uint64_le(address)
  buffer.write_uint64_le(length)
  buffer.to_bytes()
}

///|
fn encode_object_header(
  payload : Bytes,
  start : UInt64,
  creation_order~ : Bool = false,
) -> Bytes raise Hdf5Error {
  let raw_messages = split_object_header_messages(payload)
  let messages = if creation_order {
    raw_messages.map(message_with_creation_order)
  } else {
    raw_messages
  }
  let chunks = plan_object_header_chunks(messages)
  let chunk_bodies : Array[Int32] = []
  for index in 0..<chunks.length() {
    let continuation_size = if index + 1 < chunks.length() { 20 } else { 0 }
    chunk_bodies.push(
      object_header_messages_size(chunks[index]) + continuation_size,
    )
  }
  let addresses : Array[UInt64] = [start]
  let mut next_end = start.to_int() + 12 + chunk_bodies[0]
  for index in 1..<chunks.length() {
    let address = writer_align(next_end)
    addresses.push(address.to_uint64())
    next_end = address + 8 + chunk_bodies[index]
  }
  let output = @buffer.Buffer::Buffer(size_hint=next_end - start.to_int())
  let initial = @buffer.Buffer::Buffer(size_hint=chunk_bodies[0])
  for message in chunks[0] {
    writer_write_bytes(initial, message)
  }
  if chunks.length() > 1 {
    write_header_message_with_order(
      initial,
      msg_object_continuation,
      encode_object_continuation(
        addresses[1],
        (8 + chunk_bodies[1]).to_uint64(),
      ),
      creation_order~,
    )
  }
  guard initial.length() <= 65535 else {
    raise InvalidField("object header chunk is too large")
  }
  writer_write_bytes(output, b"OHDR")
  output.write_byte(b'\x02')
  output.write_byte(if creation_order { b'\x0D' } else { b'\x01' })
  output.write_uint16_le(initial.length().to_uint16())
  writer_write_bytes(output, initial.to_bytes())
  output.write_uint_le(metadata_checksum(output.to_bytes(), 0, output.length()))
  for index in 1..<chunks.length() {
    writer_write_zeros(
      output,
      addresses[index].to_int() - start.to_int() - output.length(),
    )
    let continuation = @buffer.Buffer::Buffer(size_hint=8 + chunk_bodies[index])
    writer_write_bytes(continuation, b"OCHK")
    for message in chunks[index] {
      writer_write_bytes(continuation, message)
    }
    if index + 1 < chunks.length() {
      write_header_message_with_order(
        continuation,
        msg_object_continuation,
        encode_object_continuation(
          addresses[index + 1],
          (8 + chunk_bodies[index + 1]).to_uint64(),
        ),
        creation_order~,
      )
    }
    continuation.write_uint_le(
      metadata_checksum(continuation.to_bytes(), 0, continuation.length()),
    )
    writer_write_bytes(output, continuation.to_bytes())
  }
  output.to_bytes()
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
fn writer_reference_address(
  objects : Array[WritableObject],
  path : String,
) -> UInt64 raise Hdf5Error {
  let canonical = canonical_writer_path(path)
  match objects.search_by(object => object.path == canonical) {
    Some(index) => objects[index].address
    None =>
      raise InvalidArgument(
        "object reference target does not exist: " + canonical,
      )
  }
}

///|
fn region_selection_max_value(selection : Hdf5RegionSelection) -> UInt64 {
  let mut maximum = 0UL
  let observe = fn(value : UInt64) { if value > maximum { maximum = value } }
  match selection {
    RegionEmpty | RegionAll => ()
    RegionPoints(points) => {
      observe(points.length().to_uint64())
      for point in points {
        for coordinate in point {
          observe(coordinate)
        }
      }
    }
    RegionHyperslab(start, stride, count, block) =>
      for values in [start, stride, count, block] {
        for value in values {
          observe(value)
        }
      }
    RegionBlocks(blocks) => {
      observe(blocks.length().to_uint64())
      for block in blocks {
        for value in block.start {
          observe(value)
        }
        for value in block.end {
          observe(value)
        }
      }
    }
  }
  maximum
}

///|
fn region_selection_encoding_width(selection : Hdf5RegionSelection) -> Int32 {
  let maximum = region_selection_max_value(selection)
  if maximum <= 0xFFFFUL {
    2
  } else if maximum <= 0xFFFFFFFFUL {
    4
  } else {
    8
  }
}

///|
fn encode_region_selection(
  dataspace : Dataspace,
  selection : Hdf5RegionSelection,
) -> Bytes raise Hdf5Error {
  validate_region_selection(dataspace, selection)
  let rank = dataspace.dimensions.length()
  let buffer = @buffer.Buffer::Buffer()
  match selection {
    RegionEmpty | RegionAll => {
      buffer.write_uint_le(if selection is RegionAll { 3U } else { 0U })
      buffer.write_uint_le(1U)
      buffer.write_uint_le(0U)
      buffer.write_uint_le(0U)
    }
    RegionPoints(points) => {
      guard points.length() <= 0x7FFFFFFF else {
        raise InvalidArgument("region reference contains too many points")
      }
      let width = region_selection_encoding_width(selection)
      buffer.write_uint_le(1U)
      buffer.write_uint_le(2U)
      buffer.write_byte(width.to_byte())
      buffer.write_uint_le(rank.reinterpret_as_uint())
      write_uint_width(buffer, points.length().to_uint64(), width)
      for point in points {
        for coordinate in point {
          write_uint_width(buffer, coordinate, width)
        }
      }
    }
    RegionHyperslab(start, stride, count, block) => {
      let width = region_selection_encoding_width(selection)
      buffer.write_uint_le(2U)
      buffer.write_uint_le(3U)
      buffer.write_byte(b'\x01')
      buffer.write_byte(width.to_byte())
      buffer.write_uint_le(rank.reinterpret_as_uint())
      for axis in 0..<rank {
        write_uint_width(buffer, start[axis], width)
        write_uint_width(buffer, stride[axis], width)
        write_uint_width(buffer, count[axis], width)
        write_uint_width(buffer, block[axis], width)
      }
    }
    RegionBlocks(blocks) => {
      guard blocks.length() <= 0x7FFFFFFF else {
        raise InvalidArgument("region reference contains too many blocks")
      }
      let width = region_selection_encoding_width(selection)
      buffer.write_uint_le(2U)
      buffer.write_uint_le(3U)
      buffer.write_byte(b'\x00')
      buffer.write_byte(width.to_byte())
      buffer.write_uint_le(rank.reinterpret_as_uint())
      write_uint_width(buffer, blocks.length().to_uint64(), width)
      for block in blocks {
        for axis in 0..<rank {
          write_uint_width(buffer, block.start[axis], width)
          write_uint_width(buffer, block.end[axis], width)
        }
      }
    }
  }
  buffer.to_bytes()
}

///|
fn writer_region_reference_payload(
  objects : Array[WritableObject],
  path : String,
  selection : Hdf5RegionSelection,
) -> Bytes raise Hdf5Error {
  let canonical = canonical_writer_path(path)
  let object = match objects.search_by(item => item.path == canonical) {
    Some(index) => objects[index]
    None =>
      raise InvalidArgument(
        "region reference target does not exist: " + canonical,
      )
  }
  guard object.kind is WritableDatasetObject(_) else {
    raise InvalidArgument(
      "region reference target is not a dataset: " + canonical,
    )
  }
  let dataspace = match object.dataspace {
    Some(dataspace) => dataspace
    None => raise InvalidField("writer dataset has no dataspace")
  }
  guard dataspace.kind != Null else {
    raise InvalidArgument("region reference target has a null dataspace")
  }
  let buffer = @buffer.Buffer::Buffer()
  buffer.write_uint64_le(object.address)
  writer_write_bytes(buffer, encode_region_selection(dataspace, selection))
  buffer.to_bytes()
}

///|
fn Hdf5Writer::object_payload(
  self : Hdf5Writer,
  object : WritableObject,
  objects : Array[WritableObject],
  heap : WritableGlobalHeap,
  dense : Array[WritableDenseMetadata],
  external : Array[WritableExternalHeap],
  external_addresses_final : Bool,
  shared~ : Array[WritableSharedAttribute] = [],
) -> Bytes raise Hdf5Error {
  let buffer = @buffer.Buffer::Buffer()
  match object.kind {
    WritableGroup => {
      let entries = self.dense_link_entries(object, objects, track_order=true)
      if writer_dense_entries_fit(entries) {
        let (heap_address, index_address, order_index_address) = writer_dense_addresses(
          dense,
          object.path,
          WritableDenseLinks,
        )
        write_header_message(
          buffer,
          msg_link_info,
          encode_dense_storage_info(
            true,
            writer_dense_max_creation_order(entries),
            heap_address,
            index_address,
            order_index_address,
          ),
        )
      } else {
        write_header_message(buffer, msg_link_info, encode_link_info())
        let compact_entries = self.dense_link_entries(
          object,
          objects,
          track_order=false,
        )
        for entry in compact_entries {
          write_header_message(buffer, msg_link, entry.data)
        }
      }
    }
    WritableDatasetObject(index) => {
      let dataset = self.datasets[index]
      let encoded = encode_write_values(dataset.values, heap~, objects~)
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
          write_header_message(
            buffer,
            msg_fill_value,
            encode_fill_value(fill, heap~, objects~),
          )
        None => ()
      }
      let filter_fill = match dataset.fill_value {
        Some(fill) => encode_write_values(fill, heap~, objects~).payload
        None => Bytes::make(encoded.element_size, b'\x00')
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
        WritableExternal(slots) => {
          let heap_address = if external_addresses_final {
            let address = writer_external_heap_address(external, object.path)
            guard address != 0xFFFFFFFFFFFFFFFFUL else {
              raise InvalidField("external data heap was not assigned")
            }
            address
          } else {
            0UL
          }
          write_header_message(
            buffer,
            msg_external_data_files,
            encode_external_data_message(slots, heap_address),
          )
          write_header_message(
            buffer,
            msg_data_layout,
            encode_external_data_layout(external_slot_total(slots)),
          )
        }
        WritableChunked(chunk_dimensions, filters) => {
          if filters.length() > 0 {
            write_header_message(
              buffer,
              msg_filter_pipeline,
              encode_filter_pipeline(
                filters,
                encoded.element_size,
                writer_chunk_element_count(chunk_dimensions),
                dataset.values,
                filter_fill,
              ),
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
              encode_filter_pipeline(
                filters,
                encoded.element_size,
                writer_chunk_element_count(chunk_dimensions),
                dataset.values,
                filter_fill,
              ),
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
  let attributes = self.object_attributes(object.path)
  if attributes.length() > 0 {
    let entries = self.dense_attribute_entries(
      object.path,
      heap,
      objects,
      shared~,
    )
    if writer_dense_entries_fit(entries) {
      let (heap_address, index_address, order_index_address) = writer_dense_addresses(
        dense,
        object.path,
        WritableDenseAttributes,
      )
      write_header_message(
        buffer,
        msg_attribute_info,
        encode_dense_storage_info(
          false,
          writer_dense_max_creation_order(entries),
          heap_address,
          index_address,
          order_index_address,
        ),
      )
    } else {
      write_header_message(
        buffer,
        msg_attribute_info,
        encode_compact_attribute_info(),
      )
      for entry in entries {
        write_header_message_with_flags(
          buffer,
          msg_attribute,
          entry.data,
          entry.flags,
        )
      }
    }
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
      dataspace: None,
    })
  }
  for index in 0..<writer.datasets.length() {
    let dataset = writer.datasets[index]
    objects.push({
      path: dataset.path,
      kind: WritableDatasetObject(index),
      address: 0UL,
      raw_address: 0UL,
      dataspace: Some({
        kind: if dataset.dimensions.length() == 0 {
          Scalar
        } else {
          Simple
        },
        dimensions: dataset.dimensions.copy(),
        maximum_dimensions: dataset.maximum_dimensions.copy(),
      }),
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
fn collect_heap_sequences(
  output : Array[Bytes],
  datatype : Datatype,
  value : DatasetValue,
  heap~ : WritableGlobalHeap,
  objects~ : Array[WritableObject],
) -> Unit raise Hdf5Error {
  match datatype.class {
    6 =>
      match value {
        Compound(fields) =>
          for index in 0..<fields.length() {
            collect_heap_sequences(
              output,
              datatype.compound_members[index].datatype,
              fields[index].value,
              heap~,
              objects~,
            )
          }
        _ => ()
      }
    9 =>
      match value {
        Sequence(items) =>
          if items.length() > 0 {
            let base = datatype.base_types[0]
            let encoded = @buffer.Buffer::Buffer()
            for item in items {
              writer_write_bytes(
                encoded,
                encode_structured_value(base, item, heap~, objects~),
              )
            }
            let data = encoded.to_bytes()
            if !output.contains(data) {
              guard output.length() < 65535 else {
                raise InvalidField("writer global heap has too many objects")
              }
              output.push(data)
            }
          }
        _ => ()
      }
    10 =>
      match value {
        Elements(items) =>
          for item in items {
            collect_heap_sequences(
              output,
              datatype.base_types[0],
              item,
              heap~,
              objects~,
            )
          }
        _ => ()
      }
    _ => ()
  }
}

///|
fn collect_heap_regions(
  output : Array[Bytes],
  datatype : Datatype,
  value : DatasetValue,
  objects : Array[WritableObject],
) -> Unit raise Hdf5Error {
  match datatype.class {
    6 =>
      match value {
        Compound(fields) =>
          for index in 0..<fields.length() {
            collect_heap_regions(
              output,
              datatype.compound_members[index].datatype,
              fields[index].value,
              objects,
            )
          }
        _ => ()
      }
    7 =>
      if datatype.reference_kind == 1 {
        match value {
          RegionReference(Some(target), selection) => {
            let payload = writer_region_reference_payload(
              objects, target, selection,
            )
            if !output.contains(payload) {
              guard output.length() < 65535 else {
                raise InvalidField("writer global heap has too many objects")
              }
              output.push(payload)
            }
          }
          _ => ()
        }
      }
    9 =>
      match value {
        Sequence(items) =>
          for item in items {
            collect_heap_regions(output, datatype.base_types[0], item, objects)
          }
        _ => ()
      }
    10 =>
      match value {
        Elements(items) =>
          for item in items {
            collect_heap_regions(output, datatype.base_types[0], item, objects)
          }
        _ => ()
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
    match dataset.fill_value {
      Some(fill) => collect_heap_strings(strings, fill)
      None => ()
    }
  }
  for attribute in self.attributes {
    collect_heap_strings(strings, attribute.values)
  }
  strings
}

///|
fn Hdf5Writer::global_heap_sequences(
  self : Hdf5Writer,
  heap~ : WritableGlobalHeap,
  objects~ : Array[WritableObject],
) -> Array[Bytes] raise Hdf5Error {
  let sequences : Array[Bytes] = []
  for dataset in self.datasets {
    match dataset.values {
      StructuredValues(datatype, items) =>
        for item in items {
          collect_heap_sequences(sequences, datatype, item, heap~, objects~)
        }
      _ => ()
    }
    match dataset.fill_value {
      Some(StructuredValues(datatype, items)) =>
        for item in items {
          collect_heap_sequences(sequences, datatype, item, heap~, objects~)
        }
      _ => ()
    }
  }
  for attribute in self.attributes {
    match attribute.values {
      StructuredValues(datatype, items) =>
        for item in items {
          collect_heap_sequences(sequences, datatype, item, heap~, objects~)
        }
      _ => ()
    }
  }
  sequences
}

///|
fn Hdf5Writer::global_heap_regions(
  self : Hdf5Writer,
  objects~ : Array[WritableObject],
) -> Array[Bytes] raise Hdf5Error {
  let regions : Array[Bytes] = []
  for dataset in self.datasets {
    match dataset.values {
      StructuredValues(datatype, items) =>
        for item in items {
          collect_heap_regions(regions, datatype, item, objects)
        }
      _ => ()
    }
    match dataset.fill_value {
      Some(StructuredValues(datatype, items)) =>
        for item in items {
          collect_heap_regions(regions, datatype, item, objects)
        }
      _ => ()
    }
  }
  for attribute in self.attributes {
    match attribute.values {
      StructuredValues(datatype, items) =>
        for item in items {
          collect_heap_regions(regions, datatype, item, objects)
        }
      _ => ()
    }
  }
  regions
}

///|
fn encode_heap_objects(objects : Array[Bytes]) -> Bytes raise Hdf5Error {
  let object_count = objects.length()
  if object_count == 0 {
    return b""
  }
  guard object_count <= 65535 else {
    raise InvalidField("writer global heap has too many objects")
  }
  let mut used_size = 16
  for item in objects {
    let length = item.length()
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
  let mut object_index = 0
  for item in objects {
    object_index = object_index + 1
    buffer.write_uint16_le(object_index.to_uint16())
    buffer.write_uint16_le(0)
    buffer.write_uint_le(0U)
    buffer.write_uint64_le(item.length().to_uint64())
    writer_write_bytes(buffer, item)
    writer_write_zeros(buffer, round_up_to_eight(item.length()) - item.length())
  }
  buffer.write_uint16_le(0)
  buffer.write_uint16_le(0)
  buffer.write_uint_le(0U)
  buffer.write_uint64_le((collection_size - used_size).to_uint64())
  writer_write_zeros(buffer, collection_size - buffer.length())
  buffer.to_bytes()
}

///|
fn encode_global_heap(heap : WritableGlobalHeap) -> Bytes raise Hdf5Error {
  let objects : Array[Bytes] = []
  for item in heap.strings {
    objects.push(@utf8.encode(item))
  }
  for item in heap.sequences {
    objects.push(item)
  }
  encode_heap_objects(objects)
}

///|
fn encode_region_heap(heap : WritableGlobalHeap) -> Bytes raise Hdf5Error {
  encode_heap_objects(heap.regions)
}

///|
fn assign_writable_addresses(
  writer : Hdf5Writer,
  templates : Array[WritableObject],
) -> WritableLayout raise Hdf5Error {
  let empty_heap = empty_writable_heap()
  let shared_candidates = writer.shared_attribute_candidates(
    templates, empty_heap,
  )
  let lengths : Array[Int32] = []
  for object in templates {
    lengths.push(
      encode_object_header(
        writer.object_payload(
          object,
          templates,
          empty_heap,
          [],
          [],
          false,
          shared=shared_candidates,
        ),
        0UL,
        creation_order=writer_dense_entries_fit(
          writer.dense_attribute_entries(
            object.path,
            empty_heap,
            templates,
            shared=shared_candidates,
          ),
        ),
      ).length(),
    )
  }
  let objects : Array[WritableObject] = []
  let mut offset = if shared_candidates.length() > 0 { 88 } else { 48 }
  for index in 0..<templates.length() {
    let object = templates[index]
    objects.push({
      path: object.path,
      kind: object.kind,
      address: offset.to_uint64(),
      raw_address: 0UL,
      dataspace: object.dataspace,
    })
    offset = writer_align(offset + lengths[index])
  }
  let strings = writer.global_heap_strings()
  let regions = writer.global_heap_regions(objects~)
  let region_address = if regions.length() > 0 {
    offset.to_uint64()
  } else {
    0UL
  }
  if regions.length() > 0 {
    offset = writer_align(offset + encode_heap_objects(regions).length())
  }
  let base_heap = {
    address: if strings.length() > 0 {
      offset.to_uint64()
    } else {
      0UL
    },
    region_address,
    strings,
    regions,
    sequences: [],
  }
  let sequences = writer.global_heap_sequences(heap=base_heap, objects~)
  let heap = {
    address: if base_heap.address == 0UL && sequences.length() > 0 {
      offset.to_uint64()
    } else {
      base_heap.address
    },
    region_address,
    strings,
    regions,
    sequences,
  }
  if heap.strings.length() > 0 || heap.sequences.length() > 0 {
    offset = writer_align(offset + encode_global_heap(heap).length())
  }
  let dense : Array[WritableDenseMetadata] = []
  for object in objects {
    let candidates : Array[(WritableDenseKind, Array[WritableDenseHeapEntry])] = []
    match object.kind {
      WritableGroup =>
        candidates.push(
          (
            WritableDenseLinks,
            writer.dense_link_entries(object, objects, track_order=true),
          ),
        )
      WritableDatasetObject(_) => ()
    }
    candidates.push(
      (
        WritableDenseAttributes,
        writer.dense_attribute_entries(
          object.path,
          heap,
          objects,
          shared=shared_candidates,
        ),
      ),
    )
    for (kind, dense_entries) in candidates {
      if writer_dense_entries_fit(dense_entries) {
        let (placeholder_heap, placeholder_index, placeholder_order_index) = writer.encode_dense_metadata(
          object,
          objects,
          heap,
          0UL,
          kind,
          shared=shared_candidates,
        )
        let heap_address = offset.to_uint64()
        let index_address = heap_address + placeholder_heap.length().to_uint64()
        dense.push({
          path: object.path,
          kind,
          heap_address,
          index_address,
          order_index_address: index_address +
          placeholder_index.length().to_uint64(),
        })
        offset = writer_align(
          offset +
          placeholder_heap.length() +
          placeholder_index.length() +
          placeholder_order_index.length(),
        )
      }
    }
  }
  let external : Array[WritableExternalHeap] = []
  for object in objects {
    match object.kind {
      WritableDatasetObject(dataset_index) => {
        let dataset = writer.datasets[dataset_index]
        match dataset.storage {
          WritableExternal(slots) => {
            let address = offset.to_uint64()
            external.push({ path: object.path, address, })
            offset = writer_align(
              offset + encode_external_local_heap(slots, address).length(),
            )
          }
          _ => ()
        }
      }
      WritableGroup => ()
    }
  }
  for index in 0..<objects.length() {
    match objects[index].kind {
      WritableDatasetObject(dataset_index) => {
        let dataset = writer.datasets[dataset_index]
        let has_storage = match dataset.storage {
          WritableContiguous =>
            encode_write_values(dataset.values, objects~).payload.length() > 0
          WritableChunked(_, _) => true
          WritableExtensible(_, _, _) => true
          WritableExternal(_) => false
        }
        if has_storage {
          objects[index] = {
            path: objects[index].path,
            kind: objects[index].kind,
            address: objects[index].address,
            raw_address: offset.to_uint64(),
            dataspace: objects[index].dataspace,
          }
          offset = match dataset.storage {
            WritableContiguous =>
              writer_align(
                offset +
                encode_write_values(dataset.values, heap~, objects~).payload.length(),
              )
            WritableChunked(_, _) =>
              writer_align(
                offset +
                encode_fixed_array_storage(
                  offset.to_uint64(),
                  dataset,
                  heap,
                  objects,
                ).length(),
              )
            WritableExtensible(_, _, _) =>
              writer_align(
                offset +
                encode_extensible_array_storage(
                  offset.to_uint64(),
                  dataset,
                  heap,
                  objects,
                ).length(),
              )
            WritableExternal(_) => offset
          }
        }
      }
      WritableGroup => ()
    }
  }
  let shared = if shared_candidates.length() > 0 {
    let table_address = offset.to_uint64()
    let heap_address = writer_align(
      table_address.to_int() + writer_shared_table_size,
    ).to_uint64()
    let attributes : Array[WritableSharedAttribute] = []
    for candidate in shared_candidates {
      let attribute = match
        writer
        .object_attributes(candidate.path)
        .search_by(attribute => attribute.name == candidate.name) {
        Some(index) => writer.object_attributes(candidate.path)[index]
        None => raise InvalidField("writer shared attribute disappeared")
      }
      let data = encode_attribute(attribute, heap, objects)
      guard data.length() <= writer_shared_max_managed_size else {
        raise InvalidField("writer shared attribute changed size")
      }
      attributes.push({
        path: candidate.path,
        name: candidate.name,
        data,
        id: b"",
      })
    }
    let (heap_bytes, ids) = encode_writer_shared_heap(heap_address, attributes)
    let resolved : Array[WritableSharedAttribute] = []
    for index in 0..<attributes.length() {
      resolved.push({
        path: attributes[index].path,
        name: attributes[index].name,
        data: attributes[index].data,
        id: ids[index],
      })
    }
    offset = writer_align(heap_address.to_int() + heap_bytes.length())
    Some({
      extension_address: 48UL,
      table_address,
      heap_address,
      attributes: resolved,
    })
  } else {
    None
  }
  { objects, heap, dense, external, shared, }
}

///|
fn encode_superblock(
  root_address : UInt64,
  extension_address : UInt64,
  eof : UInt64,
) -> Bytes {
  let buffer = @buffer.Buffer::Buffer(size_hint=48)
  writer_write_bytes(buffer, b"\x89HDF\r\n\x1a\n")
  buffer.write_byte(b'\x03')
  buffer.write_byte(b'\x08')
  buffer.write_byte(b'\x08')
  buffer.write_byte(b'\x00')
  buffer.write_uint64_le(0UL)
  buffer.write_uint64_le(extension_address)
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
  let shared_attributes = match layout.shared {
    Some(shared) => shared.attributes
    None => []
  }
  let encoded_headers : Array[Bytes] = []
  let mut eof = 48
  for object in objects {
    let creation_order = layout.dense.search_by(metadata => {
        metadata.path == object.path && metadata.kind == WritableDenseAttributes
      })
      is Some(_)
    let header = encode_object_header(
      self.object_payload(
        object,
        objects,
        layout.heap,
        layout.dense,
        layout.external,
        true,
        shared=shared_attributes,
      ),
      object.address,
      creation_order~,
    )
    encoded_headers.push(header)
    eof = writer_align(object.address.to_int() + header.length())
  }
  let region_heap_payload = encode_region_heap(layout.heap)
  if region_heap_payload.length() > 0 {
    eof = writer_align(
      layout.heap.region_address.to_int() + region_heap_payload.length(),
    )
  }
  let heap_payload = encode_global_heap(layout.heap)
  if heap_payload.length() > 0 {
    eof = writer_align(layout.heap.address.to_int() + heap_payload.length())
  }
  let dense_payloads : Array[(UInt64, Bytes)] = []
  for metadata in layout.dense {
    let object = find_writable_object(objects, metadata.path)
    let (heap_bytes, index_bytes, order_index_bytes) = self.encode_dense_metadata(
      object,
      objects,
      layout.heap,
      metadata.heap_address,
      metadata.kind,
      shared=shared_attributes,
    )
    guard heap_bytes.length().to_uint64() ==
      metadata.index_address - metadata.heap_address else {
      raise InvalidField("dense metadata address changed during encoding")
    }
    guard index_bytes.length().to_uint64() ==
      metadata.order_index_address - metadata.index_address else {
      raise InvalidField("dense order index address changed during encoding")
    }
    dense_payloads.push((metadata.heap_address, heap_bytes))
    dense_payloads.push((metadata.index_address, index_bytes))
    dense_payloads.push((metadata.order_index_address, order_index_bytes))
    eof = writer_align(
      metadata.order_index_address.to_int() + order_index_bytes.length(),
    )
  }
  let external_payloads : Array[(UInt64, Bytes)] = []
  for metadata in layout.external {
    let dataset = self.datasets[self.writer_dataset_index(metadata.path)]
    let slots = match dataset.storage {
      WritableExternal(slots) => slots
      _ => raise InvalidField("external data heap has the wrong dataset kind")
    }
    let payload = encode_external_local_heap(slots, metadata.address)
    external_payloads.push((metadata.address, payload))
    eof = writer_align(metadata.address.to_int() + payload.length())
  }
  let raw_payloads : Array[(UInt64, Bytes)] = []
  for object in objects {
    match object.kind {
      WritableDatasetObject(index) => {
        let dataset = self.datasets[index]
        let payload = match dataset.storage {
          WritableContiguous =>
            encode_write_values(dataset.values, heap=layout.heap, objects~).payload
          WritableChunked(_, _) =>
            encode_fixed_array_storage(
              object.raw_address,
              dataset,
              layout.heap,
              objects,
            )
          WritableExtensible(_, _, _) =>
            encode_extensible_array_storage(
              object.raw_address,
              dataset,
              layout.heap,
              objects,
            )
          WritableExternal(_) => b""
        }
        if payload.length() > 0 {
          raw_payloads.push((object.raw_address, payload))
          eof = writer_align(object.raw_address.to_int() + payload.length())
        }
      }
      WritableGroup => ()
    }
  }
  let shared_payloads : Array[(UInt64, Bytes)] = []
  if layout.shared is Some(shared) {
    let table = encode_writer_shared_table(
      shared.heap_address,
      shared.attributes.length(),
    )
    let (heap_bytes, ids) = encode_writer_shared_heap(
      shared.heap_address,
      shared.attributes,
    )
    for index in 0..<ids.length() {
      guard ids[index] == shared.attributes[index].id else {
        raise InvalidField("shared-message heap ID changed during encoding")
      }
    }
    guard table.length() <=
      shared.heap_address.to_int() - shared.table_address.to_int() else {
      raise InvalidField("shared-message table overlaps its heap")
    }
    shared_payloads.push((shared.table_address, table))
    shared_payloads.push((shared.heap_address, heap_bytes))
    let end = writer_align(shared.heap_address.to_int() + heap_bytes.length())
    if end > eof {
      eof = end
    }
  }
  let output = @buffer.Buffer::Buffer(size_hint=eof)
  writer_write_bytes(
    output,
    encode_superblock(
      find_writable_object(objects, "/").address,
      match layout.shared {
        Some(shared) => shared.extension_address
        None => 0xFFFFFFFFFFFFFFFFUL
      },
      eof.to_uint64(),
    ),
  )
  if layout.shared is Some(shared) {
    writer_write_zeros(
      output,
      shared.extension_address.to_int() - output.length(),
    )
    writer_write_bytes(
      output,
      encode_writer_superblock_extension(shared.table_address),
    )
  }
  for index in 0..<objects.length() {
    writer_write_zeros(
      output,
      objects[index].address.to_int() - output.length(),
    )
    writer_write_bytes(output, encoded_headers[index])
  }
  if region_heap_payload.length() > 0 {
    writer_write_zeros(
      output,
      layout.heap.region_address.to_int() - output.length(),
    )
    writer_write_bytes(output, region_heap_payload)
  }
  if heap_payload.length() > 0 {
    writer_write_zeros(output, layout.heap.address.to_int() - output.length())
    writer_write_bytes(output, heap_payload)
  }
  for (address, payload) in dense_payloads {
    writer_write_zeros(output, address.to_int() - output.length())
    writer_write_bytes(output, payload)
  }
  for (address, payload) in external_payloads {
    writer_write_zeros(output, address.to_int() - output.length())
    writer_write_bytes(output, payload)
  }
  for (address, payload) in raw_payloads {
    writer_write_zeros(output, address.to_int() - output.length())
    writer_write_bytes(output, payload)
  }
  for (address, payload) in shared_payloads {
    writer_write_zeros(output, address.to_int() - output.length())
    writer_write_bytes(output, payload)
  }
  writer_write_zeros(output, eof - output.length())
  output.to_bytes()
}
