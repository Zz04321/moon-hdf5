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
```

Run from the repository root. Scripts reopen files and verify values with h5py.
Timestamps can vary; tests compare semantics, not binary hashes.

| File | Cases |
| --- | --- |
| group_storage.h5 | Empty, compact (2 links), dense (12 links); with/without creation-order indexing |
| fill_v2.h5 | Earliest/v108 bounds, v2 fill messages, v1 chunk B-trees |
| fill_v3.h5 | v108/v108 bounds, v3 fill messages, v1 chunk B-trees |
| contiguous_fill_v2.h5, contiguous_fill_v3.h5 | Unallocated numeric/string/default fills, scalar/empty/null datasets and allocated partial writes, with legacy and v108 bounds |
| numeric_layouts.h5 | All standard integer widths and IEEE floats in both byte orders; unsupported integer precision/offset and float exponent bias; nonstandard numeric attributes |
| attribute_storage.h5 | Compact and dense attributes, with/without creation-order tracking |
| layout_v4.h5 | v110 bounds; raw/filtered/unallocated/single-chunk masks, implicit edge and bounded-resizable grids, compact/contiguous data; basic fixed array |
| fixed_array.h5 | Raw/filtered edge chunks, bounded maximum strides, skipped filters, fills, paged arrays with uninitialized pages and a partial final page |
| extensible_array.h5 | Raw/filtered growing and reordered unlimited dimensions, direct/super blocks, sparse pages, skipped filters, unallocated fills and shrink-to-empty |
| extensible_array_32.h5 | Raw/filtered extensible arrays with 4-byte addresses and 8-byte lengths |
| btree_v2.h5 | Raw/filtered v2 B-tree chunks, multiple unlimited dimensions, depth-2 trees, edge/sparse chunks, skipped filters, unallocated fills and shrink-to-empty |
| filters_v1.h5, filters_v2.h5 | Deflate/shuffle/Fletcher32 alone and combined, big-endian edge chunks, odd Fletcher32 byte counts, missing chunks, skipped filters and corrupt checksum; pipeline v1/v2 |
| header_continuations.h5 | Seven root datasets containing `[42]`, v108 bounds, chained object-header continuations with trailing gaps |

Group datasets `value_NN` contain `[NN, NN + 100]`. Compact groups must read
normally; listing or looking up children of dense groups must raise
`UnsupportedFeature`.

Fill datasets `/numbers/{signed,big_endian,floating}` and `/other/text` have
shape `(3, 5)`, chunks `(2, 3)`, and two allocated chunks. Fills are -7, -1234,
1.5 and `none`; positions `(0, 0)` / `(2, 4)` hold 42 / 99 or `yes` / `end`.
`/other/unallocated` is four -9 values without allocated storage;
`/other/default` is four zeros; `/other/partial_default` is `[5, 0, 0, 0]`.

Small groups isolate fill behavior; header continuation regressions use their
own fixture. See [the audit](../docs/audit.md) for resolved and remaining issues.
