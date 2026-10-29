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
| Navigation | heap, symbol_table, link, hdf5_file, resolver | Groups, links, paths and cross-file resolution |
| Dense metadata | fractal_heap, v2_metadata | Fractal heaps and metadata v2 B-trees |
| Metadata | dataspace, datatype, data_layout | Shapes, types and storage |
| Values | dataset, selection, region_reference, reference, chunked, direct_chunks, filter, fill_value, global_heap, attribute, streaming | Value decoding, reference resolution and bounded tile iteration |
| Writing | writer | New-file encoding for groups, datasets and attributes |

Dataset metadata parsing is separate from storage reads and value decoding.
`read_dataset_metadata` returns the dataspace, datatype and layout without
loading contiguous or chunked payloads; `read_dataset` reuses that metadata
path before decoding all values. Dataset selections validate their rank and
bounds, preserve point order and duplicates, and read only the selected elements
from compact or contiguous storage. For chunked storage, every supported index
is validated while only chunks intersecting the selection are read and decoded.

`Hdf5Resolver` is the file-manager boundary for multiple `Hdf5Source` values.
The source passed to `new` is named `<root>`; `add_source` registers named
external-link targets. Resolution follows hard, soft and external links, tries
an exact target name, then a name relative to the referring source directory,
then the basename. A resolved node records its source, canonical object path,
address and kind. An explicit visited set rejects soft/external-link cycles and
a depth bound limits indirection chains. `Hdf5File` retains its single-source
behavior and still reports external links as unresolved.

External raw-data storage follows the same boundary. `Hdf5File` rejects it
because the dataset bytes live outside the parsed file. `Hdf5Resolver` parses
the external data message, resolves each registered filename with the same
exact/relative/basename rules used for external links, and concatenates slots in
declaration order while applying each slot's file offset. Full reads and
selection reads are supported across slot boundaries. `DatasetChunkIterator`
uses the resolver-aware selection path, so tile iteration is bounded by one tile
even when a tile crosses external files. This post-v1.0 extension is limited to
contiguous, unfiltered storage; remote URLs and external chunked layouts are
rejected. A registered source shorter than its declared slot contributes its
available prefix and zero-fills the remainder, matching HDF5 external-storage
reads. The writer exposes the same storage subset through
`Hdf5Writer::add_external_dataset`. It validates fixed-size values and exact
slot-size coverage, emits the external-data message plus a local filename heap,
and gives the data-layout message an undefined address with the combined byte
length. The caller owns, creates and populates the referenced external files;
the encoder neither copies their bytes nor embeds them in the HDF5 file.

`DatasetChunkIterator` divides a dataset into caller-selected tiles without
materializing the complete dataset. Each tile is read through the same validated
selection path, so memory is bounded by the tile result plus a decoded storage
chunk when filters require whole-chunk decoding. Storage indexes are consulted
for each tile; this favors bounded memory and early termination over a single
full-index traversal. Scalar datasets produce one empty-coordinate tile, and
datasets with a zero-length axis produce no tiles. `Hdf5Resolver::chunk_iterator`
provides the same iterator after resolving a path to its concrete source.

Addresses are resolved relative to the superblock base. Modern group link-info
messages select compact or dense storage from allocated heap/index addresses.
Dense groups and attributes use a fractal heap for message bodies and a metadata
v2 B-tree for lookup. Name indexes are sorted for deterministic output, while
creation-order indexes retain their recorded order. Fractal heap direct and
indirect blocks, managed/tiny/direct-huge/indirect-huge heap IDs, and
leaf/internal v2 B-tree nodes are bounds checked. Filtered heaps decode each
direct block and huge object through the heap's stored pipeline before
validating direct-block checksums. Indirect huge records are located through
the heap's v2 B-tree. `Hdf5File::walk` follows every named hard-link path, so
aliases to the same group are all visible. It tracks addresses on the current
ancestor path rather than using a global visited set: recursive aliases stop,
while sibling aliases retain their distinct paths. A depth bound also rejects
excessively deep hierarchies. Chunked reads initialize the output with fill bytes before copying
allocated chunks. Unallocated contiguous data uses
the same fill initialization; empty/null data does not dereference addresses.
Allocated storage is read as stored. Single-file reads explicitly reject
external dataset storage, which is available only through `Hdf5Resolver` as
described above. Byte cursors, object headers, message constants and chunk-index
records remain internal implementation details.

The high-level reader parses filter-pipeline messages v1/v2 and reverses active
filters for each v1 B-tree chunk. Masked-out filters are skipped, including
unknown optional filters. Deflate uses a zlib wrapper (not a gzip container),
Adler-32 verification and an output limit derived from chunk dimensions.
Shuffle restores byte planes; Fletcher32 verifies and removes its trailer.
Integer scale-offset reads HDF5 1.14.6's 21-byte chunk header, unpacks
most-significant-bit-first values, applies the stored minimum and restores the
declared byte order. The all-ones packed value maps to the declared fill value;
an unstored fill maps to zero. Configured full precision is accepted only when
HDF5 bypasses filtering and stores raw element bytes. Decimal floating-point
scale-offset reads f32/f64 headers and reconstructs each value as
`packed / 10^digits + minimum`, using single- or double-precision arithmetic to
match HDF5, before applying the declared byte order. Decoded chunks must exactly
match the full chunk shape, including edge padding.

The writer emits scale-offset as the first filter in a chunk pipeline. Integer
values use automatic or fixed bit counts; f32/f64 values use HDF5 DSCALE with a
decimal digit count. The writer excludes fill values from the minimum and
maximum, writes the 21-byte header and packs differences most-significant-bit
first. Fill values become the all-ones marker. A configured full-width integer
filter stores raw element bytes. Automatic full-range chunks are rejected
because HDF5 1.14.6 cannot decode the header form it emits for that case, so
callers must select the raw full-width path explicitly. Floating-point
scale-offset rejects non-finite data and Fletcher32 pipelines.

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

Fixed-point metadata accepts 1–8-byte storage, any positive precision contained
in the storage unit, and the declared padding flags. Decoding shifts out the bit
offset, masks the precision and sign-extends signed values from their effective
width; the same path covers datasets, attributes, fills and nested values.
IEEE binary16/32/64 use direct decoding. Other binary floats use the declared
sign, exponent and mantissa positions and sizes, exponent bias and implied-MSB
normalization, then convert to `Double`; storage is limited to 8 bytes and 64
bits of precision. Enum datatypes recursively parse an integer base type and
require the enum and base sizes to agree. Member names use
the version 1/2 8-byte padding rule or the version 3 single-NUL encoding; values
follow the base type's signedness, offset, precision and byte order. Datasets
and attributes decode enum payloads to member names and reject values with no
matching member. Empty member lists, empty or unterminated names and non-integer
bases are rejected.
The remaining standard datatype classes use recursive `Datatype` metadata and
`DatasetValue` values. Compound members retain names and byte offsets; array
dimensions and vlen sequence bases retain their nested datatypes. Bitfields
apply their declared offset and precision, opaque values preserve bytes and
tags, and complex values decode their floating-point real and imaginary parts.
Reference values remain encoded bytes in the value tree so the decoder stays
layout-agnostic. `Hdf5File::resolve_object_reference` interprets the
file-address-width (4- or 8-byte) object-reference form as an address, validates
its object header and finds a deterministic hard-link path with a bounded
hierarchy walk. Null references return `None`; dangling addresses are rejected
explicitly.

`Hdf5File::resolve_region_reference` reads the global-heap object named by a
legacy dataset-region reference, resolves its target dataset in the same file
and parses the serialized dataspace selection. It supports none/all selections,
point-selection versions 1–2 and hyperslab versions 1–3, including regular
stride/count/block selections and irregular block lists. Regular hyperslab axes
are stored as interleaved start/stride/count/block vectors in version 3. Rank,
bounds, counts, encoding widths and trailing bytes are validated. The serialized
reference is `offset_size + 4` bytes, so the same decoder handles the 4- and
8-byte-address fixtures.
`read_region_reference` expands the selection to linear indices and reuses the
normal dataset selection path, so compact, contiguous and chunked storage share
the same reading and filter behavior, including filtered chunks. Regular
hyperslabs report their `count * block` shape; points and irregular blocks
report a flat element count. Version-4 encoded references are not supported.
HDF5 2.0 datatype-message v5 encodings for compound, array and complex types are
accepted alongside earlier encodings.
Fixed strings honor null-terminated, null-padded and space-padded forms plus
ASCII/UTF-8 character sets. Callers that construct the public `Datatype` record
directly must initialize its recursive metadata fields.
Attribute-info heap/index addresses determine dense storage independently of
creation-order tracking, and dense records are read through the same fractal
heap and metadata v2 B-tree helpers as groups. Numeric storage wider than 8
bytes and binary-float normalizations other than implied MSB raise an error,
including when used in attributes; the inspector prints attribute errors with
their path.

The reader resolves file-wide shared object header messages through the
superblock-extension SOHM table, its SMTB indexes and their fractal heaps.
Wrapper versions 1–3 cover both SOHM heap IDs and committed object-header
targets. A dense attribute B-tree record marked shared reads its payload from
the SOHM attribute heap rather than the object's own attribute heap. Attribute
flags may additionally share the embedded datatype and dataspace. Resolution is
cycle-checked and malformed indexes, heap IDs and checksums are rejected. The
writer can emit the same whole-attribute heap-ID form when sharing is enabled.

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

The new-file writer emits a superblock v3, object headers v2, compact or dense
hard, soft and external-link metadata, contiguous or chunked dataset layouts and
compact or dense attributes.
Object headers are split on complete message boundaries: the initial OHDR chunk
uses a 16-bit size, and additional messages continue through checksummed OCHK
chunks referenced by continuation messages. Small attribute sets include a
compact attribute-info message with undefined dense-storage addresses, allowing
HDF5 readers to report attribute counts correctly.

Groups and objects with attributes use dense storage for 9–65,535 links or
attributes when every encoded heap object is at most 65,535 bytes. Each collection has
one FRHP header, one checksummed FHDB direct block, one BTHD header and one BTLF
name index, plus a creation-order B-tree (type 6 for links or type 9 for
attributes) sharing the same heap IDs. Dense link records contain a `lookup3`
name hash and a 7-byte heap ID; dense attribute records contain an 8-byte heap
ID, flags, the creation order and a `lookup3` name hash. Link and attribute
collections are planned separately, so a group can use both. Link order starts
at zero in each group, attribute order starts at zero on each object, and
creation-order tracking is advertised through link-info, attribute-info and
object-header flags. The writer does not emit dense indirect or huge heap
blocks or a free-space manager.
Sets outside those bounds retain link or attribute messages in compact form and
use OCHK continuations when the object header exceeds one chunk.

`Hdf5Writer::set_attribute_sharing(true)` adds a v3 superblock extension, one
combined SMTB index and one FRHP/SOHM managed-object heap. Compact whole
attributes store a 10-byte version-3 heap-ID wrapper in the object header. A
dense attribute B-tree record uses the SOHM heap ID directly and advertises the
shared flag, so the object's local attribute heap contains only unshared
messages and may be an empty FRHP. Attributes whose encoded message exceeds
4 KiB remain inline or dense, while their deduplicated datatype and dataspace
messages are stored in the same SOHM heap and referenced by attribute flags
`0x01` and `0x02`. `Hdf5Writer::set_fill_sharing(true)` also stores each custom
dataset fill message of at most 4 KiB in the same heap and marks the object
header fill message as shared. The SMTB message-type mask is composed from the
message kinds actually emitted. The writer does not deduplicate equal whole
attributes and does not emit committed-target messages.

The writer calculates all object, continuation, dense-metadata, global-heap and
raw-data addresses before encoding checksummed metadata. Soft links may use
absolute targets or targets relative to their containing group and may remain
dangling. External links use HDF5's version-zero
external-link information block with null-terminated target file and object-path
strings; `Hdf5File` parses those fields, while `Hdf5Resolver` opens registered
targets. The public
builder validates absolute paths, duplicate objects, links and attributes,
dataspace rank and element counts, fixed-string widths and encoded link-message
sizes before allocating output.
Supported values are signed/unsigned integers stored in 1–8 bytes in either byte
order, IEEE float32/64 in either byte order, fixed-width or variable-length
UTF-8 strings, time, bitfield, opaque, legacy object references, legacy
dataset-region references and variable-length sequence values. Object-reference
values hold target paths and the null reference; addresses are resolved after
the final object layout is known, so forward references are valid.
`Datatype::write_object_reference` emits the class-7 object-reference datatype,
and `ObjectReference` values can be nested in compounds, fixed arrays and
variable-length sequences as well as stored directly in datasets and
attributes. `Datatype::write_region_reference` emits the 12-byte class-7 region
form. `RegionReference` values encode null, empty, all, point, regular or
irregular hyperslab and block selections against same-file datasets, including
forward targets and targets in filtered chunks. Region-reference records are
kept in a separate global heap collection from strings and sequence payloads;
mixing them can make HDF5 1.14.6 corrupt its heap bookkeeping.
Variable strings and sequences use 16-byte descriptors and a shared global heap
collection; empty values use the zero descriptor and repeated payloads share one
heap object. Sequence bases must be fixed-size datatypes. Custom variable-length
string fills use the same descriptor and heap collection; variable-length
sequence fills remain unsupported. A variable-length sequence whose base is or
contains a region reference is rejected, because HDF5 1.14.6 crashes while
converting that otherwise encodable value.
Filter pipelines remain limited to fixed-size writer values because HDF5 1.14.6
rejects them on variable-length datasets. Integer scale-offset supports
automatic or fixed bit counts. Floating-point scale-offset supports f32/f64
decimal digits. Scale-offset must precede shuffle, deflate and supported
Fletcher32 combinations. `set_chunk_filter_mask` selects filters to skip for one
finite or extensible chunk. The mask is applied while encoding that chunk and
stored in the fixed-array or extensible-array index, so the reader and HDF5
apply the same inverse pipeline. Masks are validated against the pipeline width
and chunk grid before the builder is mutated.
Narrow integer constructors validate every value before truncating it to the
encoded width. Structured values reuse the reader's recursive `Datatype` and
`DatasetValue` model. The writer emits version-three compound, enum and array
datatype messages and version-one time, bitfield, opaque and variable-length
sequence messages. Compound members are packed in declaration order, enums use
an integer base, arrays can recursively contain any supported fixed-size writer
type, and sequence bases are encoded recursively under the same fixed-size
rule. Bitfield values are written at their declared bit offset and validated
against their precision. Each numeric member preserves its declared byte order,
including enum-member tables and values nested inside compounds or arrays.
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

Structured datasets, fills and compact or dense attributes share one recursive
encoder.
Validation checks nesting depth, member names and order, integer widths, enum
names and values, array element counts, compound bounds and exact encoded sizes
before address assignment. Caller-owned datatype and value arrays are copied
when objects are added to the builder.
