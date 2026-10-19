# Test fixtures

Tests use committed, generated `.h5` files without Python or native HDF5.
Older fixtures have no committed generator; these scripts only regenerate
the group/fill/metadata regression files.

## Regenerate (POSIX shell)

```sh
python3 -m venv /tmp/moon-hdf5-fixtures
/tmp/moon-hdf5-fixtures/bin/pip install h5py==3.16.0 numpy==2.5.4
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_group_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_fill_fixtures.py
/tmp/moon-hdf5-fixtures/bin/python scripts/generate_metadata_fixtures.py
```

Run from the repository root. Scripts reopen files and verify values with h5py.
Timestamps can vary; tests compare semantics, not binary hashes.

| File | Cases |
| --- | --- |
| group_storage.h5 | Empty, compact (2 links), dense (12 links); with/without creation-order indexing |
| fill_v2.h5 | Earliest/v108 bounds, v2 fill messages, v1 chunk B-trees |
| fill_v3.h5 | v108/v108 bounds, v3 fill messages, v1 chunk B-trees |
| numeric_layouts.h5 | All standard integer widths and IEEE floats in both byte orders; unsupported integer precision/offset and float exponent bias; nonstandard numeric attributes |
| attribute_storage.h5 | Compact and dense attributes, with/without creation-order tracking |

Group datasets `value_NN` contain `[NN, NN + 100]`. Compact groups must read
normally; listing or looking up children of dense groups must raise
`UnsupportedFeature`.

Fill datasets `/numbers/{signed,big_endian,floating}` and `/other/text` have
shape `(3, 5)`, chunks `(2, 3)`, and two allocated chunks. Fills are -7, -1234,
1.5 and `none`; positions `(0, 0)` / `(2, 4)` hold 42 / 99 or `yes` / `end`.
`/other/unallocated` is four -9 values without allocated storage;
`/other/default` is four zeros; `/other/partial_default` is `[5, 0, 0, 0]`.

Small groups isolate fill behavior; the independent object-header EOF defect
is recorded in [the audit](../docs/audit.md).
