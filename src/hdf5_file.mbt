///|
/// The kind of object stored under an HDF5 link.
pub(all) enum Hdf5ObjectKind {
  Group
  Dataset
  Unknown
} derive(Eq, @debug.Debug)

///|
pub extend Hdf5ObjectKind with Eq::{not_equal, equal}

///|
pub extend Hdf5ObjectKind with @debug.Debug::{to_repr}

///|
/// One named object in the file hierarchy.
pub(all) struct Hdf5Node {
  path : String
  address : UInt64
  kind : Hdf5ObjectKind
} derive(Eq, @debug.Debug)

///|
pub extend Hdf5Node with Eq::{not_equal, equal}

///|
pub extend Hdf5Node with @debug.Debug::{to_repr}

///|
/// A parsed HDF5 file backed by bytes or a random-access source.
pub(all) struct Hdf5File {
  bytes : Bytes
  superblock : Superblock
  source : &Hdf5Source?
} derive(@debug.Debug)

///|
pub extend Hdf5File with @debug.Debug::{to_repr}

///|
pub fn Hdf5File::parse(bytes : Bytes) -> Hdf5File raise Hdf5Error {
  { bytes, superblock: parse_superblock(bytes), source: None, }
}

///|
/// Parses an HDF5 file from a random-access source.
///
pub fn Hdf5File::parse_source(source : &Hdf5Source) -> Hdf5File raise Hdf5Error {
  {
    bytes: b"",
    superblock: parse_source_superblock(source),
    source: Some(source),
  }
}

///|
fn Hdf5File::materialized_bytes(self : Hdf5File) -> Bytes raise Hdf5Error {
  match self.source {
    None => self.bytes
    Some(source) => {
      let length = source.length()
      guard length <= 0x7FFFFFFFUL else {
        raise UnsupportedFeature(
          "materializing sources larger than 2 GiB is not supported",
        )
      }
      source.read_at(0UL, length.to_int())
    }
  }
}

///|
pub fn Hdf5File::root_address(self : Hdf5File) -> UInt64 {
  self.superblock.root_group_object_header
}

///|
pub fn Hdf5File::object_header_at(
  self : Hdf5File,
  address : UInt64,
) -> ObjectHeader raise Hdf5Error {
  match self.source {
    None =>
      parse_object_header(
        self.bytes,
        address,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
    Some(source) =>
      parse_source_object_header(
        source,
        address,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
  }
}

///|
fn object_kind(header : ObjectHeader) -> Hdf5ObjectKind {
  if header.first(msg_dataspace) is Some(_) &&
    header.first(msg_datatype) is Some(_) &&
    header.first(msg_data_layout) is Some(_) {
    Dataset
  } else if header.first(msg_symbol_table) is Some(_) ||
    header.first(msg_link_info) is Some(_) ||
    header.first(msg_link) is Some(_) {
    Group
  } else {
    Unknown
  }
}

///|
/// Reads the links stored in a group object header, supporting both the
/// symbol-table (v1) and link-message (v2) group encodings.
pub fn Hdf5File::links(
  self : Hdf5File,
  header : ObjectHeader,
) -> Array[GroupLink] raise Hdf5Error {
  match header.first(msg_symbol_table) {
    Some(message) => {
      let symbol_table = parse_symbol_table_message(
        message,
        self.superblock.offset_size,
        self.superblock.length_size,
      )
      match self.source {
        None =>
          read_group_links_v1(
            self.bytes,
            symbol_table.btree_address,
            symbol_table.heap_address,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.superblock.base_address,
          )
        Some(source) =>
          read_source_group_links_v1(
            source,
            symbol_table.btree_address,
            symbol_table.heap_address,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.superblock.base_address,
          )
      }
    }
    None =>
      match self.source {
        Some(source) =>
          read_source_group_links_v2(
            source,
            header,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.superblock.base_address,
          )
        None =>
          read_group_links_v2(
            self.bytes,
            header,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.superblock.base_address,
          )
      }
  }
}

///|
fn split_path(path : String) -> Array[String] {
  let segments : Array[String] = []
  for part in path.split("/") {
    let segment = part.to_owned()
    if segment != "" && segment != "." {
      segments.push(segment)
    }
  }
  segments
}

///|
fn join_path(parent : String, name : String) -> String {
  if parent == "/" || parent == "" {
    "/" + name
  } else {
    parent + "/" + name
  }
}

///|
pub fn Hdf5File::resolve(
  self : Hdf5File,
  path : String,
) -> Hdf5Node raise Hdf5Error {
  self.resolve_from("/", self.root_address(), path, 0)
}

///|
fn Hdf5File::resolve_from(
  self : Hdf5File,
  base_path : String,
  base_address : UInt64,
  path : String,
  depth : Int32,
) -> Hdf5Node raise Hdf5Error {
  guard depth < 16 else {
    raise InvalidField("too many soft-link indirections")
  }
  let segments = split_path(path)
  if segments.length() == 0 {
    let header = self.object_header_at(base_address)
    return {
      path: base_path,
      address: base_address,
      kind: object_kind(header),
    }
  }
  let mut current_path = base_path
  let mut current_address = base_address
  let mut index = 0
  while index < segments.length() {
    let header = self.object_header_at(current_address)
    let links = self.links(header)
    let segment = segments[index]
    let link = match links.search_by(link => link.name == segment) {
      Some(found) => links[found]
      None => raise InvalidField("path component not found: " + segment)
    }
    match link.link_target {
      Some(target) => {
        if link.cache_type == 64U {
          raise UnsupportedFeature("external links are not supported yet")
        }
        let rebased = if target.has_prefix("/") {
          target
        } else {
          join_path(current_path, target)
        }
        let mut remaining = ""
        let mut next = index + 1
        while next < segments.length() {
          remaining = join_path(remaining, segments[next])
          next = next + 1
        }
        let full_path = if remaining == "" {
          rebased
        } else {
          join_path(rebased, remaining)
        }
        return self.resolve_from("/", self.root_address(), full_path, depth + 1)
      }
      None => {
        current_path = join_path(current_path, segment)
        current_address = link.object_header_address
        index = index + 1
      }
    }
  }
  let header = self.object_header_at(current_address)
  { path: current_path, address: current_address, kind: object_kind(header), }
}

///|
pub fn Hdf5File::open_dataset(
  self : Hdf5File,
  path : String,
) -> Dataset raise Hdf5Error {
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
  match self.source {
    Some(source) if source_selection_supported(metadata) => {
      let rank = metadata.dataspace.dimensions.length()
      let start = Array::make(rank, 0UL)
      let stride = Array::make(rank, 1UL)
      let selected = read_source_dataset_selection(
        source,
        header,
        metadata,
        Hyperslab(start, metadata.dataspace.dimensions.copy(), stride),
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
      {
        dataspace: metadata.dataspace,
        datatype: metadata.datatype,
        layout: metadata.layout,
        values: selected.values,
      }
    }
    _ =>
      read_dataset(
        self.materialized_bytes(),
        header,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
  }
}

///|
/// Reads a dataset's shape, datatype and storage layout without loading its
/// values.
pub fn Hdf5File::open_dataset_metadata(
  self : Hdf5File,
  path : String,
) -> DatasetMetadata raise Hdf5Error {
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  let header = self.object_header_at(node.address)
  read_dataset_metadata(
    header,
    self.superblock.offset_size,
    self.superblock.length_size,
  )
}

///|
/// Reads a hyperslab or point selection from a dataset.
pub fn Hdf5File::read_dataset_selection(
  self : Hdf5File,
  path : String,
  selection : DatasetSelection,
) -> DatasetSelectionResult raise Hdf5Error {
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
  match self.source {
    Some(source) if source_selection_supported(metadata) =>
      read_source_dataset_selection(
        source,
        header,
        metadata,
        selection,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
    _ =>
      read_dataset_selection(
        self.materialized_bytes(),
        header,
        metadata,
        selection,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
  }
}

///|
pub fn Hdf5File::attributes(
  self : Hdf5File,
  path : String,
) -> Array[Attribute] raise Hdf5Error {
  let node = self.resolve(path)
  let header = self.object_header_at(node.address)
  match self.source {
    Some(source) =>
      read_source_attributes(
        source,
        header,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
    None =>
      read_attributes(
        self.bytes,
        header,
        self.superblock.offset_size,
        self.superblock.length_size,
        self.superblock.base_address,
      )
  }
}

///|
fn Hdf5File::walk_group(
  self : Hdf5File,
  path : String,
  address : UInt64,
  visited : Array[UInt64],
  nodes : Array[Hdf5Node],
  depth : Int32,
) -> Unit raise Hdf5Error {
  guard depth < 128 else { raise InvalidField("group hierarchy is too deep") }
  if visited.contains(address) {
    return
  }
  visited.push(address)
  let header = self.object_header_at(address)
  for link in self.links(header) {
    let child_path = join_path(path, link.name)
    match link.link_target {
      Some(_) =>
        if link.cache_type == 64U {
          nodes.push({ path: child_path, address: 0UL, kind: Unknown, })
        } else {
          let child = self.resolve(child_path) catch {
            _ => { path: child_path, address: 0UL, kind: Unknown, }
          }
          nodes.push({
            path: child_path,
            address: child.address,
            kind: child.kind,
          })
        }
      None => {
        let child_header = self.object_header_at(link.object_header_address)
        let kind = object_kind(child_header)
        nodes.push({
          path: child_path,
          address: link.object_header_address,
          kind,
        })
        if kind == Group {
          self.walk_group(
            child_path,
            link.object_header_address,
            visited,
            nodes,
            depth + 1,
          )
        }
      }
    }
  }
}

///|
/// Lists the full object hierarchy below the root group.
pub fn Hdf5File::walk(self : Hdf5File) -> Array[Hdf5Node] raise Hdf5Error {
  let nodes : Array[Hdf5Node] = []
  self.walk_group("/", self.root_address(), [], nodes, 0)
  nodes
}
