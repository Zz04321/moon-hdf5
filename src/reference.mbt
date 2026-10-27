///|
/// The object reached by a non-null HDF5 object reference.
///
/// `path` is the first hard-link path found by a bounded hierarchy walk. It is
/// `None` when the target is valid but has no named hard link.
pub(all) struct Hdf5ReferenceTarget {
  address : UInt64
  kind : Hdf5ObjectKind
  path : String?
} derive(Eq, @debug.Debug)

///|
pub extend Hdf5ReferenceTarget with Eq::{not_equal, equal}

///|
pub extend Hdf5ReferenceTarget with @debug.Debug::{to_repr}

///|
fn reference_is_null(reference : Bytes) -> Bool {
  let mut index = 0
  while index < reference.length() {
    if reference.unsafe_get(index) != b'\x00' {
      return false
    }
    index = index + 1
  }
  true
}

///|
fn Hdf5File::find_reference_path_in_group(
  self : Hdf5File,
  path : String,
  address : UInt64,
  target : UInt64,
  visited : Array[UInt64],
  depth : Int32,
) -> String? raise Hdf5Error {
  guard depth < 128 else { raise InvalidField("group hierarchy is too deep") }
  if visited.contains(address) {
    return None
  }
  visited.push(address)
  let header = self.object_header_at(address)
  for link in self.links(header) {
    match link.link_target {
      Some(_) => ()
      None => {
        let child_path = join_path(path, link.name)
        if link.object_header_address == target {
          return Some(child_path)
        }
        let child_kind = object_kind(
          self.object_header_at(link.object_header_address),
        ) catch {
          _ => Unknown
        }
        if child_kind == Group {
          let found = self.find_reference_path_in_group(
            child_path,
            link.object_header_address,
            target,
            visited,
            depth + 1,
          )
          if found is Some(found_path) {
            return Some(found_path)
          }
        }
      }
    }
  }
  None
}

///|
fn Hdf5File::find_reference_path(
  self : Hdf5File,
  target : UInt64,
) -> String? raise Hdf5Error {
  if target == self.root_address() {
    return Some("/")
  }
  let visited : Array[UInt64] = []
  self.find_reference_path_in_group(
    "/",
    self.root_address(),
    target,
    visited,
    0,
  )
}

///|
/// Resolves an eight-byte HDF5 object reference to its target object.
///
/// The all-zero null reference returns `None`. Region references (12 bytes),
/// malformed references and dangling addresses raise an error. References are
/// local to this file; use `Hdf5Resolver` to obtain a file that contains a
/// remotely linked dataset before resolving its stored references.
pub fn Hdf5File::resolve_object_reference(
  self : Hdf5File,
  reference : Bytes,
) -> Hdf5ReferenceTarget? raise Hdf5Error {
  guard reference.length() == 8 else {
    if reference.length() == 12 {
      raise UnsupportedFeature("region reference resolution is not supported")
    }
    raise InvalidField(
      "object reference must be 8 bytes, found " +
      reference.length().to_string(),
    )
  }
  if reference_is_null(reference) {
    return None
  }
  let address = reference.unsafe_read_uint64_le(0)
  let header = self.object_header_at(address) catch {
    error =>
      raise InvalidField(
        "object reference target is not a valid object header: " +
        error.message(),
      )
  }
  let kind = object_kind(header)
  let path = self.find_reference_path(address)
  Some({ address, kind, path, })
}
