# Test fixtures

Tests use committed `.h5` and `.mat` fixtures and need no Python or native HDF5.
Tests compare semantics because timestamps and binary layouts can vary.

## Regenerate

Run from the repository root. The pinned HDF5 1.14.6 environment is:

```sh
python3 -m venv /tmp/moon-hdf5-h5py
/tmp/moon-hdf5-h5py/bin/pip install h5py==3.13.0 numpy==2.5.4 hdf5storage==0.2.2

/tmp/moon-hdf5-h5py/bin/python scripts/generate_datatype_fixtures.py v114
/tmp/moon-hdf5-h5py/bin/python scripts/generate_metadata_fixtures.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_external_storage_fixture.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_shared_messages_fixture.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_reference_fixtures.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_scaleoffset_fixture.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_vlen_fill_fixture.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_hard_link_fixture.py
/tmp/moon-hdf5-h5py/bin/python scripts/generate_matlab73_fixture.py
```

The v114 generator refuses other HDF5 versions. A newer HDF5 2.x environment can
regenerate the remaining structural fixtures:

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

The NetCDF-4 producer fixture needs netCDF4 1.7.5/libnetcdf 4.10.1:

```sh
python3 -m venv /tmp/moon-hdf5-netcf
/tmp/moon-hdf5-netcf/bin/pip install netCDF4==1.7.5 numpy==2.5.4
/tmp/moon-hdf5-netcf/bin/python scripts/generate_netcdf4_fixture.py
```

Independent verifiers run with the HDF5 1.14.6 environment:

```sh
/tmp/moon-hdf5-h5py/bin/python scripts/verify_numeric_layouts_fixture.py testdata/numeric_layouts.h5
/tmp/moon-hdf5-h5py/bin/python scripts/verify_scaleoffset_fixture.py testdata/scaleoffset.h5
/tmp/moon-hdf5-h5py/bin/python scripts/verify_vlen_fill_fixture.py testdata/vlen_fills.h5
/tmp/moon-hdf5-h5py/bin/python scripts/verify_external_storage_fixture.py testdata/external_storage.h5
/tmp/moon-hdf5-h5py/bin/python scripts/verify_shared_messages_fixture.py testdata/shared_messages.h5
/tmp/moon-hdf5-h5py/bin/python scripts/verify_hard_links_fixture.py testdata/hard_links.h5
/tmp/moon-hdf5-h5py/bin/python scripts/verify_matlab73_fixture.py testdata/matlab73_compatible.mat
/tmp/moon-hdf5-h5py/bin/python scripts/verify_netcdf4_fixture.py testdata/netcdf4_1_7_5.h5
```

Regeneration is not part of the release gate; `scripts/release_check.sh`
verifies the committed files.

## Fixture map

| Files | Coverage |
| --- | --- |
| `group_storage.h5`, `attribute_storage.h5` | Empty/compact/dense groups and attributes, with and without creation order |
| `fill_v2.h5`, `fill_v3.h5`, `contiguous_fill_v*.h5` | Fill-message versions, unallocated and partially allocated contiguous storage |
| `numeric_layouts.h5` | Integer widths/byte orders, IEEE float16/32/64, nonstandard precision/offset/padding and normalized binary floats |
| `scaleoffset.h5` | Integer and f32/f64 decimal scale-offset, big-endian values, fill markers, pipeline and full-precision cases |
| `layout_v4.h5` | Compact/contiguous/chunked layouts, filter masks and all common chunk indexes |
| `fixed_array*.h5`, `extensible_array*.h5`, `btree_v2*.h5` | Paged/super-block fixed and extensible arrays; v2 B-tree chunk indexes |
| `fractal_heap_advanced.h5` | Filtered/raw huge objects, indirect blocks and v2 B-tree heap records |
| `enums_v108.h5`, `enums_v110.h5` | Enum bases, byte orders, name padding and enum attributes |
| `datatypes_v114*.h5`, `datatypes_v200.h5` | HDF5 1.14.6 and 2.0 datatype-message versions, including compound/array/complex |
| `interop_earliest.h5`, `interop_v108.h5`, `interop_v110.h5`, `interop_32.h5` | Cross-version object matrix and 4-/8-byte addresses |
| `references*.h5`, `references_layouts_32.h5` | Object/region references, null/dangling targets, layouts and selection versions |
| `shared_messages.h5` | SOHM table/indexes/heaps, field and whole-message sharing, dense shared records and committed targets |
| `external_storage*.h5`, `external_storage_*.bin` | Multi-file/offset/unaligned raw-data slots and 4-byte-address storage |
| `vlen_fills.h5` | Default/custom variable-length string fills from producer files |
| `hard_links.h5` | Named hard-link aliases and ancestor-cycle traversal |
| `netcdf4_1_7_5.h5` | netCDF group, dimension, chunk, filter, UTF-8 and reference-attribute structures |
| `matlab73_compatible.mat` | hdf5storage metadata, numeric/logical/char/struct and empty arrays |
