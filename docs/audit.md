# Code audit — 2026-10-29

Scope: current worktree, reader and writer, with emphasis on the HDF5 1.14.6
v1.0 delivery profile and post-v1.0 extensions.

## Verdict

The release gate passes on the current worktree:

- wasm: 227 tests passed;
- native: 227 tests passed;
- interface, formatting, `moon check`, `moon doc` and package checks passed;
- writer demo verified by h5py 3.13.0/HDF5 1.14.6;
- numeric-layout, scale-offset, vlen-fill, NetCDF-4, external-storage,
  shared-message, hard-link, MATLAB-compatible and reference fixtures verified.

The implementation remains a documented HDF5 subset. Known unsupported
structures raise errors; detection is not exhaustive.

## Verification coverage

| Area | Evidence |
| --- | --- |
| Framing and navigation | Superblock v0–v3, user blocks, compact/dense groups, hard/soft links, source-backed reads |
| Dense metadata | FRHP direct/indirect blocks, huge objects, v2 B-tree leaf/internal nodes and SOHM heap IDs |
| Datasets | Compact, contiguous, external raw data and every supported chunk index |
| Filters | Shuffle, deflate, Fletcher32, integer/f32/f64 scale-offset, masks and fill markers |
| Datatypes | Fixed-point, IEEE/normalized floats, strings, compound, enum, array, time, bitfield, opaque, sequence, references, complex |
| References | 4-/8-byte object references and legacy region references against compact, contiguous and filtered targets |
| Shared messages | Table/indexes/heaps, wrappers v1–v3, field sharing, fills, whole attributes and committed targets |
| Writer | Superblock v3, OHDR/OCHK, dense metadata, chunk indexes, SOHM, references and external slots |
| Producers | h5py, hdf5storage and netCDF4 fixtures; HDF5 1.14.6 and 2.0 datatype messages |

## Reader findings

Address conversion remains relative to the superblock base. Object headers,
continuation chunks, dense metadata, heap blocks, B-tree nodes and chunk indexes
are bounds checked before use. Metadata checksums are verified. Group traversal
stops recursive hard-link aliases at the current ancestor chain.

Dataset metadata and payload reads are separate. Selection reads validate rank,
bounds and stride and decode only intersecting allocated chunks. Missing chunks
use fill values. `chunk_iterator` bounds memory by one tile plus an intersecting
storage chunk and supports early termination.

Variable-length values read global-heap objects on demand. Dataset-region
references resolve a selection against a same-file dataset and reuse the normal
selection and filter path. Null references return `None`; dangling addresses are
rejected. Region results remain memory resident.

`Hdf5Resolver` is the multi-source boundary. It follows chained external links,
rejects cycles, reports unregistered sources and reads contiguous external
raw-data slots across file and element boundaries. `Hdf5File` intentionally does
not dereference external targets or external raw data.

Numeric storage wider than 8 bytes and binary-float normalizations other than
implied MSB are rejected. Attribute value decoding propagates unsupported
datatype errors instead of reporting an empty value. HDF5 2.0 datatype-message
v5 is accepted as forward-compatibility evidence and is outside the 1.14.6
baseline.

## Writer findings

The builder validates absolute paths, duplicates, datatype nesting, shapes,
value sizes, filters, fill values and encoded link/message limits before
encoding. Caller-owned arrays are copied. All object, continuation,
dense-metadata, global-heap, external-slot and raw-data addresses are assigned
before checksummed output is written.

The writer emits a v3 superblock and v2 object headers. Large compact headers use
checksummed OCHK continuations. Dense collections use one FHDB direct block,
one BTHD/BTLF leaf per index and type 6/9 creation-order indexes. They are limited
to 9–65,535 records with encoded heap objects up to 65,535 bytes; indirect/huge
blocks and a free-space manager are not emitted.

Finite chunked datasets use fixed-array indexes; one unlimited dimension uses
extensible-array indexes, including inline entries, direct/super blocks and paged
data blocks. Fill-only chunks keep undefined addresses, except that variable-length
datasets allocate in-shape chunks so empty strings do not depend on undefined
vlen fill storage.

Integer and floating-point scale-offset, filter masks and fixed-size filter
pipelines are encoded in the index. Variable-length datasets reject filters,
matching HDF5 1.14.6. Variable-length sequence fills are rejected. Sequence
bases must be fixed-size; bases containing region references are rejected because
HDF5 1.14.6 cannot safely consume them.

Structured values use the reader's recursive datatype model. References resolve
after final layout, allowing forward targets. Region references use a dedicated
global heap because mixing them with string or sequence payloads can corrupt HDF5
bookkeeping.

## SOHM review

The reader resolves the superblock-extension SOHM table, SMTB indexes and fractal
heaps. Wrapper versions 1–3 cover heap IDs and committed object-header targets.
Dense attribute records referencing SOHM read the payload from the shared heap.
Attribute datatype/dataspace flags are resolved separately. Cycles, malformed
indexes and checksum failures are rejected.

The writer optionally emits one SOHM table and one managed-object heap for whole
attributes and custom fills up to 65,535 bytes. Equal encoded attributes reuse one
heap ID. Named committed datatypes are written as datatype objects and referenced
through wrapper version 3 storage type 2; datasets and attributes with matching
encoded datatypes use the shared flag and target address. This path also works
without a superblock-extension SOHM table.

The remaining SOHM boundary is one table and one heap. Multiple tables, multiple
heaps and heap extension across multiple blocks are not emitted.

## Residual risks

- Detection of unsupported input is not exhaustive.
- Whole-dataset and single-selection results are memory resident.
- External raw data requires caller registration and caller-managed files.
- Dense and SOHM output use compact single-block writer forms; larger producer
  structures are reader-side features unless explicitly listed as writer output.
- Version-4 encoded references and high-level netCDF-4/MATLAB semantics remain
  outside the profile.

## Milestones

- 2026-10-19: initial parser, framing, groups, layouts, filters and CLI audit.
- 2026-10-20: selection reads, datatype decoding, writer MVP and examples.
- 2026-10-21: release gate, source reads, soft/external links.
- 2026-10-23: HDF5 1.14.6 datatypes, dense writer metadata, resolver/streaming,
  NetCDF-4 producer interoperability and v1.0 profile.
- 2026-10-27: numeric layouts, scale-offset, SOHM reader, external raw data,
  references, big-endian and structured writer extensions.
- 2026-10-28: hard-link traversal, MATLAB-compatible fixture and vlen string fills.
- 2026-10-29: structured references, region references, chunk masks, external
  slots, creation order, complex values, large dense objects, SOHM output,
  fill/attribute deduplication and committed datatypes.
