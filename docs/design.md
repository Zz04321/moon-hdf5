# Design

The parser accepts in-memory `Bytes` or a caller-provided random-access
`Hdf5Source`. Superblocks, object headers, group structures and dense metadata
use bounded source reads. Dataset values, attributes and selections from every
supported storage layout also use bounded reads. Variable-length values fetch
their global heap objects on demand. Source-backed file addresses remain
`UInt64`; only individual read lengths and in-memory buffers narrow to `Int32`.

| Layer | Files | Responsibility |
| --- | --- | --- |
| Framing | reader, superblock, object_header | Byte reads, addresses, messages |
| Navigation | heap, symbol_table, link, hdf5_file | Groups, links and paths |
| Dense metadata | fractal_heap, v2_metadata | Fractal heaps and metadata v2 B-trees |
| Metadata | dataspace, datatype, data_layout | Shapes, types and storage |
| Values | dataset, chunked, direct_chunks, filter, fill_value, global_heap, attribute | Value decoding |
| Writing | writer | New-file encoding for groups, datasets and attributes |

Dataset metadata parsing is separate from storage reads and value decoding.
`read_dataset_metadata` returns the dataspace, datatype and layout without
loading contiguous or chunked payloads; `read_dataset` reuses that metadata
path before decoding all values. Dataset selections validate their rank and
bounds, preserve point order and duplicates, and read only the selected elements
from compact or contiguous storage. For chunked storage, every supported index
is validated while only chunks intersecting the selection are read and decoded.

Addresses are resolved relative to the superblock base. Modern group link-info
messages select compact or dense storage from allocated heap/index addresses.
Dense groups and attributes use a fractal heap for message bodies and a metadata
v2 B-tree for lookup. Name indexes are sorted for deterministic output, while
creation-order indexes retain their recorded order. Fractal heap direct and
indirect blocks, managed/tiny/direct-huge/indirect-huge heap IDs, and
leaf/internal v2 B-tree nodes are bounds checked. Filtered heaps decode each
direct block and huge object through the heap's stored pipeline before
validating direct-block checksums. Indirect huge records are located through
the heap's v2 B-tree. Chunked reads initialize the output with fill bytes before copying
allocated chunks. Unallocated contiguous data uses
the same fill initialization; empty/null data does not dereference addresses.
Allocated storage is read as stored, and external dataset storage is explicitly
rejected. Byte cursors, object headers, message constants and chunk-index
records remain internal implementation details.

The high-level reader parses filter-pipeline messages v1/v2 and reverses active
filters for each v1 B-tree chunk. Masked-out filters are skipped, including
unknown optional filters. Deflate uses a zlib wrapper (not a gzip container),
Adler-32 verification and an output limit derived from chunk dimensions.
Shuffle restores byte planes; Fletcher32 verifies and removes its trailer.
Decoded chunks must exactly match the full chunk shape, including edge padding.

Layout v4 compact/contiguous storage follows the v3 encoding. Single-chunk
indexes store a direct address and, when filtered, a byte length and mask.
Implicit indexes address an early-allocated array of unfiltered chunks; the
maximum dataspace dimensions determine storage strides, while current dimensions
determine which chunks to copy. Fixed arrays use the same maximum-dimension
strides, with raw or filtered entries and optional paged storage. Uninitialized
pages and undefined chunk addresses retain fill values. Array headers, backlinks,
entry counts and storage bounds are checked. Header, data-block and page
checksums are verified with HDF5 lookup3.
Extensible arrays support one unlimited dimension, index-block entries, direct
data blocks, super blocks and paged data blocks. The unlimited dimension is moved
first for storage indexing; finite dimensions use their maximum chunk grids.
Undefined pointers and uninitialized pages retain fill values. Configuration,
block signatures, backlinks, offsets and bounds are checked. Header, index-block,
super-block, data-block and page checksums are verified.

V2 B-tree chunk indexes support raw and filtered records in leaf and internal
nodes, including datasets with multiple unlimited dimensions. Child-pointer
widths follow each level's capacity. Traversal validates subtree counts, node
bounds and strictly increasing chunk coordinates, with bounded depth. Missing
chunks retain fill values. Header and node checksums are verified.
`DataLayout` exposes a stable `DatasetLayoutKind` and a copied chunk shape.
Format-specific index records and their creation parameters remain private so
the public API does not depend on HDF5's on-disk index encodings.

V2 continuation lengths include the OCHK signature and trailing checksum;
both are excluded from message parsing. Message prefixes are four bytes, or
six with creation-order tracking; shorter trailing gaps are ignored. The OCHK
checksum covers the signature and message payload and is verified.

Recognized unsupported features raise `UnsupportedFeature`; malformed fields
normally raise `InvalidField` or `UnexpectedEof`. Unsupported attribute values
can be represented as `None`. Detection is incomplete: not all unsupported
inputs are rejected. See [audit.md](audit.md).

Invalid selections and writer inputs raise `InvalidArgument`, keeping caller
errors distinct from malformed file metadata.

Numeric metadata is checked before decoding: integers require full-width bits
with zero offset; floats require standard IEEE binary32/binary64 fields.
Enum datatypes recursively parse an integer base type and require the enum and
base sizes to agree. Member names use the version 1/2 8-byte padding rule or
the version 3 single-NUL encoding; values follow the base type's signedness,
width and byte order. Datasets and attributes decode enum payloads to member
names and reject values with no matching member. Empty member lists, empty or
unterminated names and non-integer bases are rejected.
The remaining standard datatype classes use recursive `Datatype` metadata and
`DatasetValue` values. Compound members retain names and byte offsets; array
dimensions and vlen sequence bases retain their nested datatypes. Bitfields
apply their declared offset and precision, opaque values preserve bytes and
tags, and complex values decode their floating-point real and imaginary parts.
References remain encoded bytes because resolving object and region references
requires a separate navigation API. HDF5 2.0 datatype-message v5 encodings for
compound, array and complex types are accepted alongside earlier encodings.
Fixed strings honor null-terminated, null-padded and space-padded forms plus
ASCII/UTF-8 character sets. Callers that construct the public `Datatype` record
directly must initialize its recursive metadata fields.
Attribute-info heap/index addresses determine dense storage independently of
creation-order tracking, and dense records are read through the same fractal
heap and metadata v2 B-tree helpers as groups. Unsupported numeric layouts raise
an error, including when used in attributes; the inspector prints attribute
errors with their path.

Public interfaces are generated in `src/pkg.generated.mbti`. Tests use committed
h5py fixtures and synthetic messages. Local validation covers wasm and native;
Manual CI checks, builds and tests both wasm and native targets. Regeneration
instructions are in
[testdata/README.md](../testdata/README.md).

The command-line inspector supports whole-file traversal, metadata-only dataset
inspection, hyperslab or ordered-point reads, and simple statistics. Coordinate
vectors use comma-separated unsigned integers; point lists use semicolons. The
metadata, selection and writer demo executables keep each public workflow small
enough to use as a standalone example.

The inspector writes diagnostics to stderr and returns 0/1/2 for success,
data errors and usage errors respectively. Per-dataset failures do not prevent
other datasets from being inspected, but make the overall result unsuccessful.

The new-file writer emits a superblock v3, object headers v2, compact hard, soft
and external-link messages, contiguous or chunked dataset layouts and inline
attributes.
Object headers are split on complete message boundaries: the initial OHDR chunk
uses a 16-bit size, and additional messages continue through checksummed OCHK
chunks referenced by continuation messages. Objects with attributes include a
compact attribute-info message with undefined dense-storage addresses, allowing
HDF5 readers to report attribute counts correctly. The writer calculates all
object, continuation and raw-data addresses before encoding checksummed
metadata. Soft links may use absolute targets or targets relative to their
containing group and may remain dangling. External links use HDF5's version-zero
external-link information block with null-terminated target file and object-path
strings; the reader parses those fields but never opens another file. The public
builder validates absolute paths, duplicate objects, links and attributes,
dataspace rank and element counts, fixed-string widths and encoded link-message
sizes before allocating output.
Supported values are little-endian signed/unsigned integers at 8, 16, 32 and 64
bits, IEEE float32/64, and fixed-width or variable-length UTF-8 strings. Variable
strings use 16-byte descriptors and a shared global heap collection; empty
strings use the zero descriptor and repeated strings share one heap object.
Filters remain limited to fixed-size writer values because HDF5 1.14.6 rejects
filter pipelines on variable-length datasets.
Narrow integer constructors validate every value before truncating it to the
encoded width. Structured values reuse the reader's recursive `Datatype` and
`DatasetValue` model. The writer encodes version-three compound, enum and array
datatype descriptions; compound members are packed in declaration order, enums
use a standard integer
base, and arrays can recursively contain any supported fixed-size writer type.
Finite chunked datasets use v4 layouts and fixed-array indexes. Large fixed
arrays use paged data blocks with allocation bitmaps and per-page checksums.
Datasets with one unlimited dimension use a v4 extensible-array index with
inline entries, direct data blocks, super blocks and paged data blocks. Page
allocation bitmaps and metadata checksums follow the HDF5 extensible-array
layout. Header statistics describe the allocated blocks and realized element
capacity. Fixed dimensions use their maximum extents when mapping chunk
coordinates, so indexes remain stable as the current shape changes. Fill-only
chunks retain undefined addresses, except that variable-length datasets
allocate every in-shape chunk so empty strings never rely on undefined vlen
fill storage. Allocated chunks can run through shuffle, zlib-wrapped deflate and
Fletcher32 before their stored sizes are indexed.

`resize_dataset` rebuilds the row-major value array while preserving the
overlapping coordinate region. `append_dataset` grows the unlimited dimension
and copies a shape-compatible block into the new region. Shared generic array
helpers perform both operations for every writer datatype. Both operations
mutate the in-memory builder before complete-file encoding; modifying an
existing file in place remains separate work.

Structured datasets, fills and inline attributes share one recursive encoder.
Validation checks nesting depth, member names and order, integer widths, enum
names and values, array element counts, compound bounds and exact encoded sizes
before address assignment. Caller-owned datatype and value arrays are copied
when objects are added to the builder.
