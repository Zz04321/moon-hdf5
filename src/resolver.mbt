///|
/// A concrete object reached through the resolver.
///
/// `file` is the registered source name, or `<root>` for the source passed to
/// `Hdf5Resolver::new`. `path` is the canonical path inside that source after
/// following soft and external links.
pub(all) struct Hdf5ResolvedNode {
  file : String
  path : String
  address : UInt64
  kind : Hdf5ObjectKind
} derive(Eq, @debug.Debug)

///|
pub extend Hdf5ResolvedNode with Eq::{not_equal, equal}

///|
pub extend Hdf5ResolvedNode with @debug.Debug::{to_repr}

///|
priv struct ResolverSource {
  name : String
  source : &Hdf5Source
}

///|
/// Resolves paths across a root HDF5 source and registered external-link
/// targets. Sources are kept in memory and parsed on demand.
pub struct Hdf5Resolver {
  priv sources : Array[ResolverSource]
}

///|
fn resolver_root_name() -> String {
  "<root>"
}

///|
fn resolver_file_name(name : String) -> String raise Hdf5Error {
  guard name != "" else { raise InvalidArgument("source name is empty") }
  for byte in @utf8.encode(name) {
    guard byte != b'\x00' else {
      raise InvalidArgument("source names cannot contain NUL")
    }
  }
  let segments : Array[String] = []
  for part in name.split("/") {
    let segment = part.to_owned()
    if segment == "" || segment == "." {
      continue
    }
    if segment == ".." {
      if segments.length() > 0 {
        segments.pop() |> ignore
      }
    } else {
      segments.push(segment)
    }
  }
  let mut normalized = ""
  for index in 0..<segments.length() {
    if index > 0 {
      normalized = normalized + "/"
    }
    normalized = normalized + segments[index]
  }
  guard normalized != "" else { raise InvalidArgument("source name is empty") }
  normalized
}

///|
fn resolver_parent_directory(name : String) -> String {
  let segments : Array[String] = []
  for part in name.split("/") {
    let segment = part.to_owned()
    if segment != "" && segment != "." {
      segments.push(segment)
    }
  }
  if segments.length() > 0 {
    segments.pop() |> ignore
  }
  let mut path = ""
  for index in 0..<segments.length() {
    if index > 0 {
      path = path + "/"
    }
    path = path + segments[index]
  }
  path
}

///|
fn resolver_basename(name : String) -> String {
  let segments : Array[String] = []
  for part in name.split("/") {
    let segment = part.to_owned()
    if segment != "" && segment != "." {
      segments.push(segment)
    }
  }
  if segments.length() == 0 {
    ""
  } else {
    segments[segments.length() - 1]
  }
}

///|
fn resolver_object_segments(path : String) -> Array[String] {
  let segments : Array[String] = []
  for part in path.split("/") {
    let segment = part.to_owned()
    if segment == "" || segment == "." {
      continue
    }
    if segment == ".." {
      if segments.length() > 0 {
        segments.pop() |> ignore
      }
    } else {
      segments.push(segment)
    }
  }
  segments
}

///|
fn resolver_object_path(segments : Array[String]) -> String {
  if segments.length() == 0 {
    return "/"
  }
  let mut path = ""
  for segment in segments {
    path = path + "/" + segment
  }
  path
}

///|
fn resolver_canonical_object_path(base : String, target : String) -> String {
  let segments : Array[String] = []
  if !target.has_prefix("/") {
    for segment in resolver_object_segments(base) {
      segments.push(segment)
    }
  }
  for part in target.split("/") {
    let segment = part.to_owned()
    if segment == "" || segment == "." {
      continue
    }
    if segment == ".." {
      if segments.length() > 0 {
        segments.pop() |> ignore
      }
    } else {
      segments.push(segment)
    }
  }
  resolver_object_path(segments)
}

///|
fn resolver_append_path(
  base : String,
  segments : Array[String],
  start : Int32,
) -> String {
  let result = resolver_object_segments(base)
  for index in start..<segments.length() {
    let segment = segments[index]
    if segment != "" && segment != "." {
      result.push(segment)
    }
  }
  resolver_object_path(result)
}

///|
fn resolver_visit_key(file : String, path : String) -> String {
  file + "|" + path
}

///|
fn Hdf5Resolver::source_index(self : Hdf5Resolver, name : String) -> Int32? {
  for index in 0..<self.sources.length() {
    if self.sources[index].name == name {
      return Some(index)
    }
  }
  None
}

///|
/// Creates a resolver for a root source. External targets must be registered
/// with `add_source` before they can be reached.
pub fn Hdf5Resolver::new(source : &Hdf5Source) -> Hdf5Resolver {
  { sources: [{ name: resolver_root_name(), source, }], }
}

///|
/// Registers an external-link target under a stable source name.
pub fn Hdf5Resolver::add_source(
  self : Hdf5Resolver,
  name : String,
  source : &Hdf5Source,
) -> Unit raise Hdf5Error {
  let normalized = resolver_file_name(name)
  guard normalized != resolver_root_name() else {
    raise InvalidArgument("<root> is reserved for the root source")
  }
  guard self.source_index(normalized) is None else {
    raise InvalidArgument("source name is already registered: " + normalized)
  }
  self.sources.push({ name: normalized, source, })
}

///|
/// Opens the root source or a registered external-link target.
pub fn Hdf5Resolver::open(
  self : Hdf5Resolver,
  name : String,
) -> Hdf5File raise Hdf5Error {
  let normalized = if name == resolver_root_name() {
    name
  } else {
    resolver_file_name(name)
  }
  match self.source_index(normalized) {
    Some(index) => Hdf5File::parse_source(self.sources[index].source)
    None => raise InvalidField("source is not registered: " + name)
  }
}

///|
fn Hdf5Resolver::external_source_name(
  self : Hdf5Resolver,
  referring_file : String,
  target_file : String,
) -> String raise Hdf5Error {
  let normalized = resolver_file_name(target_file)
  guard self.source_index(normalized) is None else { return normalized }
  let directory = if referring_file == resolver_root_name() {
    ""
  } else {
    resolver_parent_directory(referring_file)
  }
  if directory != "" && !target_file.has_prefix("/") {
    let relative = resolver_file_name(directory + "/" + target_file)
    guard self.source_index(relative) is None else { return relative }
  }
  let basename = resolver_basename(normalized)
  if basename != normalized {
    guard self.source_index(basename) is None else { return basename }
  }
  raise InvalidField(
    "external link target file is not registered: " + target_file,
  )
}

///|
fn Hdf5Resolver::resolve_from(
  self : Hdf5Resolver,
  file_name : String,
  file : Hdf5File,
  base_path : String,
  base_address : UInt64,
  path : String,
  depth : Int32,
  visited : Array[String],
) -> Hdf5ResolvedNode raise Hdf5Error {
  guard depth < 64 else { raise InvalidField("too many link indirections") }
  let canonical = resolver_canonical_object_path(base_path, path)
  let key = resolver_visit_key(file_name, canonical)
  guard !visited.contains(key) else {
    raise InvalidField("soft or external link cycle detected at " + key)
  }
  visited.push(key)
  let segments = resolver_object_segments(canonical)
  if segments.length() == 0 {
    let header = file.object_header_at(base_address)
    return {
      file: file_name,
      path: "/",
      address: base_address,
      kind: object_kind(header),
    }
  }
  let mut current_path = "/"
  let mut current_address = file.root_address()
  let mut index = 0
  while index < segments.length() {
    let header = file.object_header_at(current_address)
    let links = file.links(header)
    let segment = segments[index]
    let link = match links.search_by(link => link.name == segment) {
      Some(found) => links[found]
      None => raise InvalidField("path component not found: " + segment)
    }
    match link.link_target {
      Some(target) => {
        let target_path = match link.external_path {
          Some(value) => value
          None => ""
        }
        if link.cache_type == 64U {
          let target_file = self.external_source_name(file_name, target)
          let target_source = self.open(target_file)
          let target_root = resolver_canonical_object_path("/", target_path)
          let resolved_path = resolver_append_path(
            target_root,
            segments,
            index + 1,
          )
          return self.resolve_from(
            target_file,
            target_source,
            "/",
            target_source.root_address(),
            resolved_path,
            depth + 1,
            visited,
          )
        }
        let rebased = resolver_canonical_object_path(current_path, target)
        let resolved_path = resolver_append_path(rebased, segments, index + 1)
        return self.resolve_from(
          file_name,
          file,
          "/",
          file.root_address(),
          resolved_path,
          depth + 1,
          visited,
        )
      }
      None => {
        current_path = resolver_append_path(current_path, segments, index)
        current_address = link.object_header_address
        index = index + 1
      }
    }
  }
  let header = file.object_header_at(current_address)
  {
    file: file_name,
    path: current_path,
    address: current_address,
    kind: object_kind(header),
  }
}

///|
/// Resolves a path in the root source, following soft and external links.
pub fn Hdf5Resolver::resolve(
  self : Hdf5Resolver,
  path : String,
) -> Hdf5ResolvedNode raise Hdf5Error {
  let root = self.open(resolver_root_name())
  self.resolve_from(
    resolver_root_name(),
    root,
    "/",
    root.root_address(),
    path,
    0,
    [],
  )
}

///|
/// Reads a dataset through the resolver, including external-link targets.
pub fn Hdf5Resolver::read_dataset(
  self : Hdf5Resolver,
  path : String,
) -> Dataset raise Hdf5Error {
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  self.open(node.file).read_dataset(node.path)
}

///|
/// Reads dataset metadata through the resolver.
pub fn Hdf5Resolver::read_dataset_metadata(
  self : Hdf5Resolver,
  path : String,
) -> DatasetMetadata raise Hdf5Error {
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  self.open(node.file).read_dataset_metadata(node.path)
}

///|
/// Reads a dataset selection through the resolver.
pub fn Hdf5Resolver::read_dataset_selection(
  self : Hdf5Resolver,
  path : String,
  selection : DatasetSelection,
) -> DatasetSelectionResult raise Hdf5Error {
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  self.open(node.file).read_dataset_selection(node.path, selection)
}

///|
/// Creates a bounded-memory dataset iterator through the resolver.
pub fn Hdf5Resolver::chunk_iterator(
  self : Hdf5Resolver,
  path : String,
  tile_dimensions : Array[UInt64],
) -> DatasetChunkIterator raise Hdf5Error {
  let node = self.resolve(path)
  guard node.kind == Dataset else {
    raise InvalidField("object is not a dataset: " + path)
  }
  self.open(node.file).chunk_iterator(node.path, tile_dimensions)
}

///|
/// Reads attributes through the resolver.
pub fn Hdf5Resolver::attributes(
  self : Hdf5Resolver,
  path : String,
) -> Array[Attribute] raise Hdf5Error {
  let node = self.resolve(path)
  self.open(node.file).attributes(node.path)
}
