# Compatibility

moon-hdf5 targets the HDF5 1.14.6 common-file profile. “Verified” means the
repository contains a fixture, test or h5py check for that direction. This is a
documented subset, not complete HDF5 support.

## v1.0 profile

The v1.0 profile is the verified rows in the reader and writer tables below,
including their boundaries. The external raw-data table is a separate
post-v1.0 extension. HDF5 2.0 datatype fixtures are forward-compatibility
evidence, not part of the 1.14.6 baseline.

Outside the profile: virtual datasets, SWMR, parallel HDF5, third-party filters,
in-place editing, netCDF-4/MATLAB high-level semantics, version-4 encoded
references, numeric storage wider than 8 bytes and non-implied-MSB binary
floats.

The release gate is `scripts/release_check.sh`: formatting, documentation and
package checks; wasm/native tests; writer demo; and independent h5py
verification with h5py 3.13.0/HDF5 1.14.6.

## Reader

| Area | Verified | Boundary |
| --- | --- | --- |
| Framing | Superblocks v0–v3, user blocks, 4- and 8-byte addresses | Other address/length combinations are not fully tested |
| Groups | Symbol-table, compact and dense groups, creation-order indexes, soft links, all named hard-link paths | Traversal stops at an address already in the current ancestor chain |
| Cross-file links | `Hdf5Resolver` follows chained external links and relative targets | `Hdf5File` alone does not open external targets |
| Attributes | Inline v1–v3, dense, SOHM wrappers, shared datatype/dataspace fields | Unsupported attribute datatypes fail the read instead of returning empty values |
| Layouts and indexes | Compact, contiguous, chunked; v1 B-tree, single-chunk, implicit, fixed-array, extensible-array and v2 B-tree indexes | Unsupported index variants are rejected |
| Filters | Deflate, shuffle, Fletcher32, integer and f32/f64 decimal scale-offset, masks and fill markers | Other scale modes and required unknown filters are rejected |
| Numeric values | 1–8-byte fixed-point with byte order, bit offset/precision and padding; IEEE float16/32/64; implied-MSB binary floats | Wider storage and other normalizations are rejected |
| Structured values | Compound, enum, array, vlen sequence, bitfield, opaque, time, reference, complex; datatype messages v1–v5 | Reference values decode as bytes; resolution is separate |
| References | Legacy object and dataset-region references under 4- and 8-byte addresses; compact, contiguous and filtered targets; point, hyperslab, block, all and empty selections | Targets need a named hard link and, for regions, must be datasets in the same file; version-4 references are unsupported |
| Fill values | Default, fixed-size, structured and custom vlen-string fills | Undefined policies and sequence fills may be rejected |
| Selections | Hyperslabs, ordered points and bounded-memory tile iteration | One selection result is memory resident |
| Shared messages | SOHM table/indexes/heaps, wrappers v1–v3, committed targets, fills, field sharing and dense shared records | Filter-pipeline sharing is resolvable but not fixture-verified |
| Dense metadata | Managed, tiny, direct-huge and indirect-huge heap objects; metadata v2 B-trees | Broader non-matrix interoperability remains untested |
| Producer files | netCDF4 1.7.5/libnetcdf 4.10.1/HDF5 2.2.0 structures; hdf5storage 0.2.2 MATLAB metadata | Verifies encoded HDF5 structures, not complete netCDF-4 or MATLAB semantics |

Fixtures cover superblock v0/v2/v3 matrices, HDF5 1.14.6 default/`latest` and
HDF5 2.0 datatypes, scale-offset, hard links, dense/fractal heaps, references,
shared messages and producer files.

## External raw data

This post-v1.0 reader extension is resolver-only.

| Area | Verified | Boundary |
| --- | --- | --- |
| External raw data | Registered 4-/8-byte-address files, multiple slots, offsets, element-unaligned boundaries, full/selection reads and chunk iteration | `Hdf5File` cannot dereference it; filters, chunked external layouts and remote URLs are unsupported; short sources are zero-filled |

`Hdf5Writer::add_external_dataset` writes the same contiguous, unfiltered subset
for fixed-size values. Callers create the external files and exact slot sizes.

## Writer

`src/cmd/write_demo` and `scripts/verify_writer_output.py` check the output with
h5py 3.13.0/HDF5 1.14.6.

| Area | Verified | Boundary |
| --- | --- | --- |
| Framing | Superblock v3 and v2 object headers with OCHK continuations | Older output formats are not selected |
| Groups and attributes | Compact/dense collections, creation order, continued headers, inline/dense attributes | Dense limits below |
| Dense metadata | 9–65,535 records with encoded objects up to 65,535 bytes; FRHP/FHDB, BTHD/BTLF and type 6/9 indexes; optional one SOHM table/heap | One direct block and leaf per index; no huge/indirect blocks or free-space manager; larger compact sets use OCHK |
| Datasets | Contiguous, external raw data, paged fixed-array, extensible-array and v2 B-tree chunks | No virtual storage; external storage is unfiltered fixed-size only |
| Numeric values | int/uint 1–8 bytes and IEEE float32/64 in either byte order | Nonstandard floating-point output is absent |
| Strings and structures | Fixed/vlen UTF-8; compound, enum, array, time, bitfield, opaque, references, complex and sequences | Complex output is little-endian canonical `r`/`i`; sequence bases are fixed-size and cannot contain region references |
| References | Legacy 8-byte objects and 12-byte regions in datasets, attributes, compounds and arrays | Same-file targets only; version-4 references and regions inside sequences are absent |
| Filters | Shuffle, deflate, Fletcher32, integer and f32/f64 decimal scale-offset, per-chunk masks | Fixed-size values only; scale-offset must lead; floating scale-offset cannot use Fletcher32 |
| Fills | Default, fixed-size/structured and vlen-string fills; optional SOHM | Sequence fills are rejected |
| Shared messages | One SOHM table/heap for whole attributes and fills; equal attributes reuse one message; committed datatypes use wrapper v3 type 2 | Multiple SOHM tables/heaps are not written |
| Extension | One or more unlimited axes with in-memory resize; append for one unlimited axis | Multiple unlimited axes use a v2 B-tree and require `resize_dataset`; existing files are not edited in place |

## Reproduce

```sh
moon test --target wasm
moon test --target native
moon run src/cmd/write_demo --target native -- output.h5
python scripts/verify_writer_output.py output.h5
```

Ordinary MoonBit tests need no Python or native HDF5. Fixture generation is in
[testdata/README.md](../testdata/README.md).
