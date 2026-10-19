# Design

The library operates on in-memory `Bytes`; filesystem access belongs to the caller.

| Layer | Files | Responsibility |
| --- | --- | --- |
| Framing | reader, superblock, object_header | Byte reads, addresses, messages |
| Navigation | heap, symbol_table, link, hdf5_file | Groups, links and paths |
| Metadata | dataspace, datatype, data_layout | Shapes, types and storage |
| Values | dataset, chunked, direct_chunks, filter, fill_value, global_heap, attribute | Value decoding |

Addresses are resolved relative to the superblock base. Modern group link-info
messages are checked for allocated heap/index addresses before returning compact
links. Chunked reads initialize the output with fill bytes before copying allocated
chunks. The low-level `read_chunked_payload` defaults to zeros; the high-level
reader supplies parsed fill values. Unallocated contiguous data uses the same
fill initialization; empty/null data does not dereference addresses. Allocated
storage is read as stored, and external dataset storage is explicitly rejected.

The high-level reader parses filter-pipeline messages v1/v2 and reverses active
filters for each v1 B-tree chunk. Masked-out filters are skipped, including
unknown optional filters. Deflate uses a zlib wrapper (not a gzip container),
Adler-32 verification and an output limit derived from chunk dimensions.
Shuffle restores byte planes; Fletcher32 verifies and removes its trailer.
Decoded chunks must exactly match the full chunk shape, including edge padding.
The public low-level `read_chunked_payload` remains for unfiltered payloads.

Layout v4 compact/contiguous storage follows the v3 encoding. Single-chunk
indexes store a direct address and, when filtered, a byte length and mask.
Implicit indexes address an early-allocated array of unfiltered chunks; the
maximum dataspace dimensions determine storage strides, while current dimensions
determine which chunks to copy. Fixed arrays use the same maximum-dimension
strides, with raw or filtered entries and optional paged storage. Uninitialized
pages and undefined chunk addresses retain fill values. Array headers, backlinks,
entry counts and storage bounds are checked; metadata checksums are not verified.
Extensible arrays support one unlimited dimension, index-block entries, direct
data blocks, super blocks and paged data blocks. The unlimited dimension is moved
first for storage indexing; finite dimensions use their maximum chunk grids.
Undefined pointers and uninitialized pages retain fill values. Configuration,
block signatures, backlinks, offsets and bounds are checked. Metadata checksums
remain unverified.

V2 B-tree chunk indexes support raw and filtered records in leaf and internal
nodes, including datasets with multiple unlimited dimensions. Child-pointer
widths follow each level's capacity. Traversal validates subtree counts, node
bounds and strictly increasing chunk coordinates, with bounded depth. Missing
chunks retain fill values. Header and node checksums remain unverified.
For array indexes, low-level `read_chunked_payload` accepts an optional
`length_size` (default 8); pass the superblock's value for other length widths.

`DataLayout` now exposes `chunk_index : ChunkIndex` (`BTreeV1`, `Single`,
`Implicit`, `FixedArray`, `ExtensibleArray`, or `BTreeV2`). Array and B-tree
configuration structs expose their layout creation parameters.
Callers constructing this public struct directly must supply that
field; callers using `parse_data_layout` need no changes.

V2 continuation lengths include the OCHK signature and trailing checksum;
both are excluded from message parsing. Message prefixes are four bytes, or
six with creation-order tracking; shorter trailing gaps are ignored. Checksum
bytes must be present but their values are not currently verified.

Recognized unsupported features raise `UnsupportedFeature`; malformed fields
normally raise `InvalidField` or `UnexpectedEof`. Unsupported attribute values
can be represented as `None`. Detection is incomplete: not all unsupported
inputs are rejected, and metadata checksums are not verified. See [audit.md](audit.md).

Numeric metadata is checked before decoding: integers require full-width bits
with zero offset; floats require standard IEEE binary32/binary64 fields.
Attribute-info heap/index addresses determine dense storage independently of
creation-order tracking. Unsupported numeric layouts raise an error, including
when used in attributes; the inspector prints attribute errors with their path.

Public interfaces are generated in `src/pkg.generated.mbti`. Tests use committed
h5py fixtures and synthetic messages. Local validation covers wasm and native;
manual CI currently runs only default-target checks/tests. Regeneration
instructions are in [testdata/README.md](../testdata/README.md).

The inspector writes diagnostics to stderr and returns 0/1/2 for success,
data errors and usage errors respectively. Per-dataset failures do not prevent
other datasets from being inspected, but make the overall result unsuccessful.
