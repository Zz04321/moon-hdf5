# Compatibility

This document separates formats that moon-hdf5 reads from formats its writer
creates. “Verified” means the repository contains a repeatable fixture, test or
h5py check for that direction. It is not a claim of complete HDF5 support.

## h5py files read by moon-hdf5

The committed fixtures are generated with h5py and then read by wasm and native
MoonBit tests.

| Area | Verified input | Current boundary |
| --- | --- | --- |
| File framing | Superblocks v0–v3, user blocks, 4-byte and 8-byte addresses | Other address/length combinations are not comprehensively tested |
| Groups | Symbol-table groups, compact links, dense links, creation-order indexes, soft links | External links are rejected |
| Attributes | Inline messages v1–v3 and dense attributes | Unsupported attribute datatypes may be reported without values |
| Dataset layouts | Compact, contiguous and chunked | External and virtual dataset storage are rejected |
| Chunk indexes | V1 B-tree, single chunk, implicit, fixed array, extensible array and v2 B-tree | Unsupported index variants are rejected |
| Filters | Deflate, shuffle and Fletcher32, pipelines, skip masks and checksum failures | Other required filters are rejected; unknown optional skipped filters can be ignored |
| Numeric values | Standard signed/unsigned 8/16/32/64-bit integers and IEEE float32/64, both byte orders | Nonstandard precision, offsets and floating layouts are rejected |
| Strings | Fixed strings with supported padding and ASCII/UTF-8, plus variable-length UTF-8 | Other character encodings are rejected |
| Structured values | Compound, enum, fixed array, vlen sequence, bitfield, opaque, time, reference and HDF5 2.0 complex | References are returned as encoded bytes and are not resolved |
| Fill values | Default and fixed-size fills for unallocated contiguous or chunked storage | Undefined/disabled policies and custom vlen fills are rejected when decoding requires them |
| Selections | Hyperslabs and ordered points across supported layouts and chunk indexes | Inputs and selected values remain memory resident |
| Dense metadata | Managed/tiny/direct-huge/indirect-huge fractal-heap objects and metadata v2 B-trees | Broader non-matrix interoperability remains to be tested |

The cross-version object matrix is stored in `interop_earliest.h5`,
`interop_v108.h5`, `interop_v110.h5` and `interop_32.h5`. HDF5 2.0 datatype
messages are covered by `datatypes_v200.h5`. Fixture generation commands and
versions are documented in [testdata/README.md](../testdata/README.md).

## moon-hdf5 files read by h5py

`src/cmd/write_demo` creates one file containing the cases below, and
`scripts/verify_writer_output.py` opens it with h5py and checks values plus
metadata. The current local verification uses h5py 3.12.1 with HDF5 1.14.4;
earlier writer stages were also checked with h5py 3.16.0 and HDF5 2.0.0.

| Area | Verified output | Current boundary |
| --- | --- | --- |
| File framing | Superblock v3 and object headers v2 | Writer does not select older format versions |
| Groups | Nested groups and compact hard links | Dense link creation and soft/external link writing are absent |
| Attributes | Inline numeric, fixed/vlen string and structured attributes | Dense attribute writing is absent |
| Dataset layouts | Contiguous, fixed-array chunked and extensible-array chunked | Fixed-array index is limited to 1024 chunks; extensible-array index to 128 slots |
| Numeric values | Little-endian int/uint8–64 and IEEE float32/64 | Big-endian and nonstandard numeric output are absent |
| Strings | Fixed-width and variable-length UTF-8, including empty and repeated strings | Variable-length sequence output is absent |
| Structured values | Packed compound, enum and fixed array values | Reference, bitfield, opaque, time and complex output are absent |
| Filters | Shuffle, deflate and Fletcher32 in HDF5 pipeline order | Additional filters and per-chunk caller masks are absent |
| Fill values | Default and custom fixed-size fills, including structured fills | Custom variable-length fills are rejected |
| Extensible datasets | One unlimited axis, in-memory resize and append | MoonBit does not modify an existing file in place |

The h5py verification also checks that generated extensible datasets expose the
expected maximum shape and can subsequently be resized and written by h5py.

## Reproduce

```sh
moon test --target wasm
moon test --target native
moon run src/cmd/write_demo --target native -- output.h5
python scripts/verify_writer_output.py output.h5
```

The Python verification requires h5py and NumPy. Ordinary MoonBit tests use
committed fixtures and do not require Python or a native HDF5 installation.
