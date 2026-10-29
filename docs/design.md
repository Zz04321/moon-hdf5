# Design

moon-hdf5 separates parsing, metadata interpretation, value decoding and
new-file encoding. The same datatype and value model is used by the reader and
writer so structured values do not require parallel implementations.

## Layers

| Layer | Responsibility |
| --- | --- |
| Framing | Superblocks, object headers, messages, address widths and checksums |
| Navigation | Groups, links, paths, dense metadata and cross-file resolution |
| Metadata | Dataspaces, datatypes, layouts, filters, fills and shared messages |
| Values | Numeric/structured decoding, selections, references and chunk iteration |
| Sources | In-memory bytes, bounded random-access reads and external-link registries |
| Writer | Validation, address planning and complete-file encoding |

## Reader

`Hdf5File::parse` accepts bytes; `parse_source` accepts an `Hdf5Source` and keeps
file addresses as `UInt64`. Superblocks, object headers, links, dense indexes,
heaps and selected values use bounded source reads. A full dataset or selection
result is materialized in memory, while `chunk_iterator` bounds memory to one
tile plus an intersecting storage chunk.

Dataset metadata and values are separate paths. `read_dataset_metadata` returns
the dataspace, datatype and layout without touching payload storage.
`read_dataset` reuses that path before decoding values. Selections validate rank,
bounds and stride, preserve point order and duplicates, and decode only
intersecting allocated chunks.

Groups use symbol tables or link messages. Modern groups select compact or dense
storage from link-info addresses; dense groups store links in an FRHP fractal
heap with metadata v2 B-tree name and creation-order indexes. Attributes use the
same dense machinery. Every named hard-link path is discovered, with recursion
stopped when an address repeats in the current ancestor chain.

Chunk indexes are normalized behind `DataLayout` and `DatasetLayoutKind`.
Supported readers cover v1 B-trees, single-chunk, implicit, fixed-array,
extensible-array and v2 B-tree indexes. Headers, block signatures, backlinks,
entry counts, coordinates, offsets and checksums are validated. Missing or
undefined chunks use fill values.

Filters are decoded through a shared pipeline model: shuffle, deflate,
Fletcher32, integer scale-offset and f32/f64 decimal scale-offset. Per-chunk
masks follow HDF5 filter order. Non-finite scale-offset inputs are rejected.

Datatype decoding recursively handles fixed-point layouts, IEEE and normalized
binary floats, fixed/vlen strings, compound, enum, array, time, bitfield, opaque,
sequence, reference and complex classes. Fixed-point values honor byte order,
bit offset, precision, sign extension and padding. Variable-length values fetch
global-heap objects on demand.

Legacy object references resolve to an object header and deterministic hard-link
path. Legacy dataset-region references resolve a global-heap selection against a
same-file dataset; selected-value reads reuse the normal selection and filter
path. Null references return `None`; dangling targets are rejected.

## External sources

`Hdf5Resolver` owns named `Hdf5Source` values. The primary source is `<root>`;
`add_source` registers external-link targets. Resolution follows hard, soft and
external links, tries exact, relative and basename matches, rejects cycles and
applies an indirection bound.

Contiguous external raw-data storage uses the same registry. Slots are
concatenated in declaration order with each file offset applied, so reads may
cross files and element boundaries. A short source contributes its available
prefix and zero-fills the declared remainder. `Hdf5File` leaves both external
links and external raw data unresolved.

## Shared messages

The reader resolves the superblock-extension SOHM table through its SMTB indexes
and fractal heaps. Wrapper versions 1–3 cover SOHM heap IDs and committed
object-header targets. Dense shared attribute records use SOHM heap IDs directly;
attribute flags may also share embedded datatype or dataspace messages. Cycles,
malformed indexes and bad checksums are rejected.

The writer can emit:

- whole attribute and custom fill messages through one SOHM table and managed
  heap, interning equal encoded attributes;
- named committed datatypes as ordinary datatype objects referenced through
  wrapper version 3 storage type 2;
- the SMTB type mask for the message kinds actually emitted.

Field-level attribute sharing is not emitted because every attribute the writer
can represent fits the whole-message path.

## Writer

`Hdf5Writer` is an in-memory builder. Adding an object validates paths, names,
shapes, datatypes, values, filters, fill values and link sizes immediately, and
copies caller-owned arrays. Encoding then computes the complete layout before
writing checksummed metadata.

The output uses superblock v3 and v2 object headers. The initial OHDR chunk uses
a 16-bit size; additional messages continue through checksummed OCHK chunks.
Object, continuation, dense-metadata, global-heap, external-slot and raw-data
addresses are assigned before final encoding.

Dense link and attribute collections are used for 9–65,535 records when every
encoded heap object is at most 65,535 bytes. Each collection has one FRHP header,
one FHDB direct block, one BTHD header and BTLF name index, plus a type 6 or 9
creation-order index sharing the same heap IDs. Larger sets remain compact and
use OCHK continuation when needed. Indirect/huge heap blocks and a free-space
manager are not emitted.

The writer supports contiguous datasets, contiguous external raw-data slots,
finite chunked datasets with fixed-array indexes, and datasets with one unlimited
dimension using extensible-array indexes. Large indexes use paged or super-block
forms. Fill-only chunks keep undefined addresses except for variable-length
datasets, which allocate in-shape chunks so empty strings do not depend on
undefined vlen storage.

Integer scale-offset supports automatic or fixed bit widths; floating-point
scale-offset supports f32/f64 decimal digits. Scale-offset must lead the
pipeline, and floating-point scale-offset cannot be combined with Fletcher32.
`set_chunk_filter_mask` records per-chunk skips in the fixed-array or
extensible-array index.

Structured values use one recursive encoder. Validation covers nesting depth,
member names and offsets, enum values, array dimensions, compound bounds and
exact encoded sizes. References store paths and resolve to addresses after final
layout, so forward targets work. Region references use a dedicated global heap
because mixing them with string or sequence heap objects can corrupt HDF5 1.14.6
bookkeeping.

Variable-length strings and sequences use 16-byte descriptors and one shared
global heap; repeated payloads share heap objects. Sequence bases must be
fixed-size. Variable-length sequence fills and sequence bases containing region
references are rejected because HDF5 1.14.6 cannot safely consume them.

## Errors and verification

Unsupported structures raise `UnsupportedFeature`; malformed metadata normally
raises `InvalidField` or `UnexpectedEof`; invalid caller input raises
`InvalidArgument`. Detection is not exhaustive.

Tests run against committed h5py/hdf5storage fixtures on wasm and native. The
release gate also runs documentation/package checks, the writer demo and
independent h5py verification with HDF5 1.14.6. Fixture regeneration is in
[testdata/README.md](../testdata/README.md); audit evidence is in
[audit.md](audit.md).
