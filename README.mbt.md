# moon-hdf5

A pure MoonBit, read-only HDF5 reader. The library accepts file bytes and
exposes groups, datasets and attributes without a native HDF5 dependency.
This is an early implementation with known compatibility gaps, not a complete
HDF5 implementation. See [audit findings](docs/audit.md).

## Run from source

Requires the [MoonBit toolchain](https://www.moonbitlang.com/download);
the native backend also needs a C compiler.

```sh
git clone https://github.com/Zz04321/moon-hdf5.git
cd moon-hdf5
moon update
moon run cmd/main --target native -- testdata/basic_v0.h5
```

Replace the final argument with your file path. Output `int[10] 0` means ten
integers with first value zero. The inspector loads complete datasets and reports
attribute-read errors, but does not return failure exit codes for handled errors.

As checked on 2026-10-19, Mooncakes search does not find this module; use the
checkout until a registry release is verified.

## Library API

Within this checkout, use imports as in [cmd/main/moon.pkg](cmd/main/moon.pkg):

```moonbit nocheck
///|
import {
  "Zz04321/moon-hdf5" @lib,
  "moonbitlang/x/fs",
}
```

In a function that handles or propagates errors:

```moonbit nocheck
///|
let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes("data.h5"))

///|
let dataset = file.open_dataset("/group/values")

///|
let shape = dataset.dataspace.dimensions

///|
let attributes = file.attributes("/group/values")

///|
let nodes = file.walk()
```

Values are `Integers`, `UnsignedIntegers`, `Floats` or `Strings`.
`resolve(path)` locates objects without decoding values. See the
[inspector](cmd/main/main.mbt) for error handling and the
[generated interface](pkg.generated.mbti) for all public functions.

## Scope

| Area | Implemented subset |
| --- | --- |
| Framing | Superblocks v0–v3, user blocks, object headers v1/v2 |
| Groups | Symbol tables, compact link messages, hard/soft links |
| Layout | Compact, contiguous, unfiltered v1 B-tree chunked data |
| Values | Standard int/uint8–64, IEEE float32/64, fixed/vlen strings |
| Attributes | Inline messages v1–v3 |
| Chunk fills | Default zeros and fixed-size numeric/string fills, including unallocated indexes |

Writing, filters (gzip/shuffle), modern chunk indexes, dense storage, external
links and compound/enum/reference types are outside the current scope.
Undefined/disabled chunk fill policies and custom vlen fills are rejected.
Version support does not imply support for every structure in that version.
Nonstandard numeric layouts and dense attributes are explicitly rejected.
Some v2 object headers have known parsing defects; unallocated contiguous
storage is also incomplete.
Inputs and decoded datasets reside in memory; this is not a streaming reader.

## Development

```sh
moon check
moon test
moon test --target native
moon info
moon fmt
```

CI is **manual only** through GitHub Actions → CI → Run workflow; it runs
`moon check` and default-target tests. Ordinary tests use committed fixtures
and do not need Python. See [fixture generation](testdata/README.md),
[design](docs/design.md) and [ecosystem comparison and sources](docs/prior-art.md).

Apache-2.0; see [LICENSE](LICENSE).
