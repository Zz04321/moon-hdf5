# Test fixtures

Tests use committed, generated `.h5` files without Python or native HDF5.
Older fixtures have no committed generator; these scripts only regenerate
the listed regression files.

## Regenerate (POSIX shell)

```sh
python3 -m venv /tmp/moon-hdf5-fixtures
/tmp/moon-hdf5-fixtures/bin/pip install h5py==3.16.0 numpy==2.5.4
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_group_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_fill_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_metadata_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_header_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_contiguous_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_filter_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_v4_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_fixed_array_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_extensible_array_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_btree_v2_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_interop_matrix.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_fractal_heap_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_enum_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_datatype_fixtures.py
```

Run from the repository root. Scripts reopen files and verify values with h5py.
Timestamps can vary; tests compare semantics, not binary hashes.

The HDF5 1.14.6 baseline fixture uses its own pinned environment:

```sh
python3 -m venv /tmp/moon-hdf5-h5py
/tmp/moon-hdf5-h5py/bin/pip install h5py==3.13.0 numpy==2.5.4
/tmp/moon-hdf5-h5py/bin/python scripts/generate_datatype_fixtures.py v114
/tmp/moon-hdf5-h5py/bin/python scripts/generate_metadata_fixtures.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_external_storage_fixture.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_shared_messages_fixture.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_reference_fixtures.py
/tmp/moon-hdf5-h5py/bin/python \
  scripts/verify_numeric_layouts_fixture.py testdata/numeric_layouts.h5
```

The generator refuses to produce the v114 fixtures from any other HDF5 version.

The NetCDF-4 interoperability fixture uses a separate producer environment:

```sh
python3 -m venv /tmp/moon-hdf5-netcf
/tmp/moon-hdf5-netcf/bin/pip install netCDF4==1.7.5 numpy==2.5.4
/tmp/moon-hdf5-netcf/bin/python scripts/generate_netcdf4_fixture.py
/tmp/moon-hdf5-h5py/bin/python \
  scripts/verify_netcdf4_fixture.py testdata/netcdf4_1_7_5.h5
/tmp/moon-hdf5-h5py/bin/python \
  scripts/verify_external_storage_fixture.py testdata/external_storage.h5
/tmp/moon-hdf5-h5py/bin/python \
  scripts/verify_shared_messages_fixture.py testdata/shared_messages.h5
```

The file is written by netCDF4 1.7.5 with libnetcdf 4.10.1 and HDF5 2.2.0.
The independent verifier uses h5py 3.13.0 with HDF5 1.14.6. Regeneration is
not part of the release gate.

| File | Cases |
| --- | --- |
| group_storage.h5 | Empty, compact (2 links), dense (12 links); with/without creation-order indexing |
| fill_v2.h5 | Earliest/v108 bounds, v2 fill messages, v1 chunk B-trees |
| fill_v3.h5 | v108/v108 bounds, v3 fill messages, v1 chunk B-trees |
| contiguous_fill_v2.h5, contiguous_fill_v3.h5 | Unallocated numeric/string/default fills, scalar/empty/null datasets and allocated partial writes, with legacy and v108 bounds |
| numeric_layouts.h5 | All standard integer widths and IEEE float16/32/64 in both byte orders; signed little- and big-endian fixed-point values with nonstandard precision/offset, an unsigned 20-bit value, binary16 normals/subnormals/zero/special values, custom attributes, custom fills, and an unsupported float exponent bias |
| attribute_storage.h5 | Compact and dense attributes, with/without creation-order tracking |
| layout_v4.h5 | v110 bounds; raw/filtered/unallocated/single-chunk masks, implicit edge and bounded-resizable grids, compact/contiguous data; basic fixed array |
| fixed_array.h5 | Raw/filtered edge chunks, bounded maximum strides, skipped filters, fills, paged arrays with uninitialized pages and a partial final page |
| extensible_array.h5 | Raw/filtered growing and reordered unlimited dimensions, direct/super blocks, sparse pages, skipped filters, unallocated fills and shrink-to-empty |
| extensible_array_32.h5 | Raw/filtered extensible arrays with 4-byte addresses and 8-byte lengths |
| btree_v2.h5 | Raw/filtered v2 B-tree chunks, multiple unlimited dimensions, depth-2 trees, edge/sparse chunks, skipped filters, unallocated fills and shrink-to-empty |
| filters_v1.h5, filters_v2.h5 | Deflate/shuffle/Fletcher32 alone and combined, big-endian edge chunks, odd Fletcher32 byte counts, missing chunks, skipped filters and corrupt checksum; pipeline v1/v2 |
| header_continuations.h5 | Seven root datasets containing `[42]`, v108 bounds, chained object-header continuations with trailing gaps |
| interop_earliest.h5, interop_v108.h5, interop_v110.h5 | Matching object matrix across earliest/v108/v110 bounds: compact and dense groups, compact and dense attributes, compact/contiguous/chunked storage, fixed/unlimited dimensions, single chunks, filters, sparse chunks and skipped filter masks |
| interop_32.h5 | Raw and filtered fixed-array chunks with 4-byte addresses and 8-byte lengths |
| fractal_heap_advanced.h5 | Filtered dense group with 1200 links plus filtered/raw long-name huge objects; indirect heap blocks and huge v2 B-trees |
| enums_v108.h5, enums_v110.h5 | Signed/unsigned/big-endian enum datasets and enum attributes; v1/v2 padded names under v108 bounds and v3 unpadded names under v110 bounds |
| datatypes_v114.h5 | HDF5 1.14.6 `latest` bounds; v4 compound, enum, fixed array and vlen-compound types, including a compound attribute |
| datatypes_v114_default.h5 | HDF5 1.14.6 default bounds with the same datatype matrix using earlier datatype-message versions |
| datatypes_v200.h5 | Nested compound and array types, vlen sequences and UTF-8 strings, bitfield, opaque, time, object/region references, HDF5 2.0 complex types, structured attributes and fills |
| netcdf4_1_7_5.h5 | Real netCDF4 1.7.5/libnetcdf 4.10.1/HDF5 2.2.0 output: groups, dimension variables, shuffled/deflated chunks with a v1 B-tree index, UTF-8 strings, numeric values and NetCDF dimension-reference attributes |
| shared_messages.h5 | HDF5 1.14.6 SOHM tables and heap IDs: internally shared datatype/dataspace fields, whole shared attributes, shared fills, dense attribute B-tree records, and a committed datatype used by a dataset and attribute |
| references.h5, references_v108.h5 | HDF5 1.14.6 `latest` and v108 object references to a dataset, group, child dataset, root group and committed datatype, plus an attribute reference, null reference, deleted target and region reference |
| external_storage.h5, external_storage_32.h5 | HDF5 1.14.6 external raw data split across two files and read at a nonzero offset under 8-byte addresses and lengths, plus a 4-byte-address/8-byte-length legacy file; supporting binary slot files are committed beside them |

Group datasets `value_NN` contain `[NN, NN + 100]`. Compact groups must read
normally. Dense groups are indexed by fractal heaps and metadata v2 B-trees;
listing and looking up their children must return all 12 links in order.

Fill datasets `/numbers/{signed,big_endian,floating}` and `/other/text` have
shape `(3, 5)`, chunks `(2, 3)`, and two allocated chunks. Fills are -7, -1234,
1.5 and `none`; positions `(0, 0)` / `(2, 4)` hold 42 / 99 or `yes` / `end`.
`/other/unallocated` is four -9 values without allocated storage;
`/other/default` is four zeros; `/other/partial_default` is `[5, 0, 0, 0]`.

Small groups isolate fill behavior; header continuation regressions use their
own fixture. See [the audit](../docs/audit.md) for resolved and remaining issues.
The reader side of the [compatibility matrix](../docs/compatibility.md) maps
these fixtures to the supported HDF5 structures they verify.

The interop matrix files contain identical object names and values, so the
same expectations run against superblock v0, v2 and v3. The v110 file covers
single-chunk, fixed-array, extensible-array and v2 B-tree chunk indexes; the
earliest and v108 files use v1 B-trees.

`fractal_heap_advanced.h5` stores `/filtered` with deflate and Fletcher32 heap
filters, enough links to allocate indirect blocks, and a 5005-byte link name
whose link message is a filtered huge object. `/raw` repeats the long link
without heap filters so both huge v2 B-tree record types are exercised.

`enums_v108.h5` and `enums_v110.h5` store `/colors` (`<i4`), `/unsigned`
(`<u1`) and `/big_endian` (`>i2`) enum datasets, plus scalar and array enum
attributes on `/colors`. `EIGHTCHR` exercises the eight-byte name boundary;
`LONG_NAME_XYZ` exercises names longer than one padding block.

`datatypes_v200.h5` uses HDF5 2.0 bounds so compound, array and complex types
exercise datatype-message v5. Structured values are also covered through point
selections. Reference values are checked as encoded bytes and object references
are resolved by MoonBit; the generator also dereferences them with h5py.

`references.h5` and `references_v108.h5` use HDF5 1.14.6 `latest` and v108
bounds. They cover dataset, group, root, committed-datatype and attribute
references, a null reference, a deleted-target reference and a region reference.
MoonBit tests resolve the valid object references in memory and through
`Hdf5Source`, reject the dangling address, and leave the 12-byte region
reference encoded.

`datatypes_v114.h5` and `datatypes_v114_default.h5` exercise HDF5 1.14.6
datatype messages under both `latest` (version 4) and default (earlier
versions) bounds, independently of the HDF5 2.0 fixtures.

`netcdf4_1_7_5.h5` is a producer-interoperability fixture, not a complete
netCDF data-model implementation test. `/temperature` is a `(4, 3, 5)` float
variable with `(2, 2, 3)` chunks, shuffle and deflate filters, and a v1 B-tree
chunk index. The MoonBit test also checks a hyperslab, UTF-8 station names,
groups, scalar attributes and NetCDF dimension-reference attributes.

`shared_messages.h5` configures three SOHM indexes for datatype/dataspace,
attribute and fill-value messages. Small attributes keep their message inline
while sharing the embedded datatype and dataspace; larger attributes share the
whole message. Dense groups contain 13 shared attributes each, exercising SOHM
heap IDs in metadata v2 B-tree records. The file also covers committed datatype
targets, a chunked dataset and both in-memory and source-backed reads.

`external_storage.h5` stores `/multi` as `int32[8]` across
`external_storage_0.bin` (12 bytes) and `external_storage_1.bin` (20 bytes).
`/offset` reads `int32[4]` from `external_storage_offset.bin` at byte offset 16.
`/unaligned` deliberately splits an `int32` element across 10-byte and 22-byte
external slots.
MoonBit tests read full values and selections through `Hdf5Resolver`, while the
resolver's chunk iterator checks a tile crossing the two-file boundary. The
independent h5py verifier checks the external-slot declarations and decoded
values. `external_storage_32.h5` repeats `/legacy` with 4-byte addresses and
lengths. These fixtures verify the resolver-only contiguous external-storage
extension, not complete external-storage support.
