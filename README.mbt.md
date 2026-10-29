# moon-hdf5

moon-hdf5 is a pure MoonBit implementation for reading HDF5 files and creating
new HDF5 files. It targets the HDF5 1.14.6 common-file profile, has no native
HDF5 dependency and supports both wasm and native targets.

The project is under active development and implements a documented subset of
HDF5. The v1.0 delivery profile in the
[compatibility matrix](docs/compatibility.md) is the release boundary and not
a claim of complete HDF5 support. See the
[audit report](docs/audit.md) for verification history.

## Read/write status

This is not a read-only library.

- `Hdf5File` reads existing files without modifying them.
- `Hdf5Writer` creates a complete new HDF5 file in memory and writes groups,
  datasets, attributes, soft and external links, fixed-size-value filters, fill
  values, legacy object references, contiguous external raw-data storage and one
  extensible dimension. Whole attributes and custom fill messages can
  optionally be shared through a file-wide SOHM table.
- In-place editing of an existing file is not implemented. `Hdf5File` parses
  external-link metadata without opening another file; `Hdf5Resolver` resolves
  those links and reads contiguous external raw-data storage when the target
  sources are registered.

The public reader and writer entry points are listed in
[`src/pkg.generated.mbti`](src/pkg.generated.mbti); see [Writing](#writing) for
an executable example.

## Features

- Reads groups, datasets, attributes, links and common HDF5 metadata versions.
- Enumerates every named hard-link path to a group while stopping recursive
  aliases at the current ancestor chain.
- Resolves shared object header messages (SOHM), including shared attributes,
  datatypes, dataspaces, fills and committed datatype targets. Inline, dense
  and source-backed attribute paths are supported.
- Decodes signed and unsigned 1–8-byte fixed-point values with declared byte
  order, bit offset, precision and padding flags; IEEE float16/32/64; normalized
  binary floats with implied-MSB layouts; string, compound, enum, array,
  bitfield, opaque, reference, variable-length sequence, time and complex values.
- Resolves legacy object references under 4- and 8-byte address widths to their
  target address, object kind and first hard-link path; resolves legacy
  dataset-region references to their target dataset, dataspace selection and
  selected values, including compact, contiguous or chunked target storage.
- Supports compact, contiguous and chunked layouts with the common HDF5 chunk
  indexes.
- Reads deflate, shuffle, Fletcher32, integer scale-offset and f32/f64 decimal
  floating-point scale-offset, including per-chunk masks and fill markers;
  writes shared fixed filter pipelines for fixed-size values.
- Provides metadata-only access, hyperslab selection and ordered point
  selection.
- Resolves chained soft and external links across registered sources, with
  cycle and indirection limits.
- Reads and writes a post-v1.0 contiguous external raw-data extension, including
  multiple external files, in-file offsets, full reads and selections.
- Reads datasets through a bounded-memory chunk iterator for edge-clipped tiles
  and early termination.
- Creates new files with groups, attributes, hard, soft and external links,
  compact or dense metadata, contiguous or chunked datasets, filters, fill
  values, object references, external raw-data slots and one extensible
  dimension. Large chunk indexes use fixed-array pages or extensible-array super
  blocks and data pages.

## Quick Start

Install the [MoonBit toolchain](https://www.moonbitlang.com/download). The
native target also requires a C compiler.

```sh
git clone https://github.com/Zz04321/moon-hdf5.git
cd moon-hdf5
moon update
moon run src/cmd/main --target native -- testdata/basic_v0.h5
```

The inspector writes diagnostics to stderr and returns status 0 for success, 1
for data errors and 2 for invalid arguments. A dataset-level error does not stop
inspection of other readable datasets.

## Reading

Import the library as shown in [src/cmd/main/moon.pkg](src/cmd/main/moon.pkg):

```moonbit nocheck
///|
import {
  "Zz04321/moon-hdf5" @lib,
  "moonbitlang/x/fs",
}

///|
let file = @lib.Hdf5File::parse(@fs.read_file_to_bytes("data.h5"))

///|
let metadata = file.read_dataset_metadata("/group/values")

///|
let dataset = file.read_dataset("/group/values")

///|
let selected = file.read_dataset_selection(
  "/matrix",
  @lib.Hyperslab([0UL, 0UL], [2UL, 2UL], [1UL, 2UL]),
)

///|
let attributes = file.attributes("/group/values")

///|
let nodes = file.walk()

///|
let resolver = @lib.Hdf5Resolver::new(
  @fs.read_file_to_bytes("data.h5"),
)
resolver.add_source(
  "other-file.h5",
  @fs.read_file_to_bytes("other-file.h5"),
)
let remote = resolver.read_dataset("/experiment/remote_counts")
let remote_chunks = resolver.chunk_iterator(
  "/experiment/remote_counts",
  [1UL, 2UL],
)

///|
let chunks = file.chunk_iterator("/matrix", [1UL, 2UL])
while chunks.next() is Some(chunk) {
  chunk.offset |> ignore
  chunk.values |> ignore
}
```

Simple values are returned as `Integers`, `UnsignedIntegers`, `Floats`,
`Strings` or `Enums`. Other datatype classes use recursive `Structured` values.
References are returned as encoded bytes. Legacy object-reference width follows
the file's address size (4 or 8 bytes); pass one to
`resolve_object_reference` to obtain its address, kind and first named path.
The all-zero null reference returns `None`; dangling addresses raise an error.
Pass a legacy dataset-region reference to `resolve_region_reference` to obtain
its dataset, path, dataspace and selection, or to `read_region_reference` to
read the selected values. Both return `None` for a null reference; region
targets must be datasets in the same file. Region-reference storage is
`offset_size + 4` bytes; selected-value reads reuse the normal selection path,
including filtered chunks. To resolve a reference read from a file reached
through `Hdf5Resolver`, open that file with `resolver.open(node.file)`.

`read_dataset_metadata` does not load dataset values. Selection reads support
compact, contiguous and chunked datasets; chunked reads decode only intersecting
allocated chunks.

`Hdf5Resolver` maintains a registry of named `Hdf5Source` values. The source passed
to `Hdf5Resolver::new` is named `<root>`; `add_source` registers external-link
targets. Path resolution follows hard, soft and external links, supports chained
links, rejects cycles, and reports an unregistered target explicitly. The
returned `Hdf5ResolvedNode` records the target source name, canonical path,
address and object kind.

`Hdf5Resolver` also reads datasets whose raw bytes use contiguous external
storage. Every referenced external file must be registered with `add_source`
using the name stored in the HDF5 file. The reader concatenates external slots
in declaration order and applies their file offsets, so full reads, hyperslabs
and point selections work across slot boundaries under 4- or 8-byte addresses
with 8-byte lengths. Slot boundaries need not align with element boundaries.
This is a post-v1.0 extension: `Hdf5File` rejects external raw data, and
filters, chunked external storage, and remote URLs are not supported for it. The
writer can create an unfiltered contiguous external dataset from caller-supplied
slots, but it does not create or copy the referenced external files.
`DatasetChunkIterator` reads edge-clipped tiles through the same selection
path. As in HDF5, missing tail bytes in a declared slot are read as zeroes while
the available prefix is returned unchanged.

`chunk_iterator` splits a dataset into caller-selected tiles. Each `next` call
returns the tile origin, its edge-clipped shape and decoded values. Memory is
bounded by one tile plus any intersecting storage chunk, and callers may stop
early by no longer calling `next`. This API does not change the selection or
storage formats supported by the reader. `Hdf5Resolver::chunk_iterator` applies
the same behavior after resolving an external-link target.

`Hdf5File::parse_source` also accepts an implementation of the random-access
`Hdf5Source` interface. Superblock parsing, object headers and group navigation
use bounded reads, as do dataset values, attributes and selections
from every supported storage layout. Global heap objects are read on demand,
and chunk selections read only their index blocks and intersecting chunks.
Source file addresses use 64-bit offsets and can exceed 2 GiB.

## Writing

`Hdf5Writer` builds a complete new file in memory. It supports signed and
unsigned 1–8-byte integers in either byte order, IEEE float32/64 in either byte
order, fixed or variable-length UTF-8 strings, and structured compound, enum,
fixed-array, time, bitfield, opaque and variable-length-sequence values.

```moonbit nocheck
///|
let writer = @lib.Hdf5Writer::new()

///|
writer.set_attribute_sharing(true)

///|
writer.set_fill_sharing(true)

///|
writer.add_dataset(
  "/experiment/counts",
  [2UL, 3UL],
  @lib.Int64Values([1L, 2L, 3L, 4L, 5L, 6L]),
)

///|
writer.add_soft_link("/experiment", "latest", "/experiment/counts")

///|
writer.add_external_link(
  "/experiment",
  "remote_counts",
  "other-file.h5",
  "/experiment/counts",
)

///|
writer.add_external_dataset(
  "/experiment/external_counts",
  [6UL],
  @lib.Int32Values([10, 20, 30, 40, 50, 60]),
  [
    {
      file_name: "external_counts_0.bin",
      file_offset: 4UL,
      data_size: 8UL,
    },
    {
      file_name: "external_counts_1.bin",
      file_offset: 0UL,
      data_size: 16UL,
    },
  ],
)

///|
writer.add_extensible_dataset(
  "/experiment/series",
  [0UL, 2UL],
  [None, Some(2UL)],
  [2U, 2U],
  @lib.Int64Values([]),
  fill_value=@lib.Int64Values([-1L]),
  filters=[@lib.Shuffle, @lib.Deflate(6), @lib.Fletcher32],
)

///|
writer.append_dataset(
  "/experiment/series",
  [2UL, 2UL],
  @lib.Int64Values([10L, 11L, 20L, 21L]),
)

///|
writer.add_chunked_dataset(
  "/experiment/codes",
  [6UL],
  [3U],
  @lib.Int32Values([1000, 1001, 1002, 2000, 2001, 2002]),
  fill_value=@lib.Int32Values([-1]),
  filters=[@lib.ScaleOffset(0), @lib.Deflate(6)],
)

///|
writer.add_attribute(
  "/experiment/counts",
  "unit",
  [],
  @lib.FixedStringValues(["items"], 8),
)

///|
let bytes = writer.encode()
```

Use `add_chunked_dataset` for a finite chunked dataset. Use
`add_extensible_dataset`, `resize_dataset` and `append_dataset` for a dataset
with one unlimited dimension. `add_soft_link` attaches a link to an absolute
target or a target relative to the containing group; dangling links are allowed.
`add_external_link` records a target file and object path. HDF5 readers resolve
them, and `Hdf5Resolver` resolves them when the target source is registered;
`Hdf5File` alone leaves the link unresolved. The `add_external_dataset` API
stores raw element bytes in caller-managed external
files. Its slots are concatenated in order and each slot applies its own byte
offset; their sizes must exactly match the dataset byte count. Only contiguous,
unfiltered fixed-size values are accepted, and the writer does not create the
external files themselves. The writer rejects link messages whose encoded
metadata exceeds HDF5's 16-bit message size. `ObjectReferenceValues` stores
top-level legacy object references as paths
to objects in the same file; `None` writes the null reference.
`Datatype::write_object_reference()` creates the matching 8-byte datatype, and
`ObjectReference(Some(path))` or `ObjectReference(None)` supplies a value nested
in a compound, fixed array or variable-length sequence. References may be stored
in datasets or attributes, and their targets may be added later because paths
are resolved when the file is encoded. `Datatype::write_region_reference()`
creates a 12-byte legacy dataset-region reference; `RegionReference(Some(path),
selection)` accepts `RegionEmpty`, `RegionAll`, `RegionPoints`,
`RegionHyperslab` and `RegionBlocks`, while `RegionReference(None, RegionEmpty)`
writes the null reference. Region targets must be datasets in the same file and
may be added after the reference. Region references can be stored directly or
in compound fields, fixed arrays and attributes. `Datatype::write_integer`,
`write_float`, `write_fixed_string`, `write_enum`, `write_array`,
`write_compound`, `write_object_reference`, `write_region_reference`,
`write_time`, `write_bitfield`, `write_opaque` and `write_vlen_sequence`
construct supported structured writer types. Use
`write_integer_with_order` and `write_float_with_order` when a
structured value needs big-endian storage; the standard constructors remain
little-endian. Variable-length sequences use a fixed-size base type, and empty,
repeated and nested-in-compound sequence values are supported. Strings and
sequence payloads share one global heap collection. Groups and objects with
attributes use dense storage when they contain 9–65,535 links or attributes and
every encoded heap object is at most 65,535 bytes. Dense groups use an FRHP
fractal heap with one FHDB direct block plus BTHD/BTLF name and creation-order indexes;
dense attributes use the same heap layout with both indexes. Link and attribute
creation order starts at zero within each group or object and survives HDF5
round-trips. The writer does not emit dense indirect or huge heap blocks or a
free-space manager.

`set_attribute_sharing(true)` places encoded whole-attribute messages of at
most 4 KiB in one file-wide SOHM heap. Compact attributes store a
shared-message wrapper in the object header; dense attribute B-tree records
store the SOHM heap ID directly, and the object's local attribute heap contains
only unshared messages. Larger attributes remain inline or dense but share
deduplicated datatype/dataspace messages, using attribute flags `0x01` and
`0x02` and version-3 heap-ID wrappers for those fields. The writer emits one
SMTB index carrying datatype, dataspace, fill and whole-attribute messages.
`set_fill_sharing(true)` places custom fill messages of at most 4 KiB in the
same SOHM heap and marks each object-header fill message as shared. Both sharing
options are disabled by default.
Metadata outside those bounds remains compact and is split across checksummed
OCHK continuation chunks. The writer demo includes a 2500-attribute object, a
group with both dense links and dense attributes including a variable-length
string attribute, a group whose non-alphabetical insertion order is checked for
both links and attributes, exactly eight oversized links and attributes that
exercise compact OCHK continuations, and dense collections with 60 KB managed
objects.

`Complex64Values` and `Complex128Values` write standard little-endian complex
arrays as HDF5's canonical `r`/`i` floating-point compound, which h5py exposes
as `complex64` and `complex128`. Complex values can also be stored in
attributes. The generic structured APIs remain available for custom compound
layouts and big-endian fields.

For integer values, `ScaleOffset(0)` selects automatic bit width and a positive
value fixes the width. For f32/f64 values, the parameter is the number of
decimal digits retained by HDF5's DSCALE mode. Scale-offset accepts fixed-size
numeric values, must be the first filter, and cannot be combined with Fletcher32
for floating-point data. Custom fixed-size and variable-length string fills are
supported; variable-length sequence fills are rejected. Call
`set_chunk_filter_mask(path, chunk_index, mask)` to skip pipeline entries for an
individual chunk using HDF5's bit ordering. Chunk indices follow the dataset's
finite or extensible chunk grid, and bits outside the pipeline are rejected.

Run the complete writer example and verify its output independently with h5py:

```sh
moon run src/cmd/write_demo --target native -- output.h5
python scripts/verify_writer_output.py output.h5
```

## Command Line

```sh
moon run src/cmd/main -- inspect testdata/basic_v0.h5
moon run src/cmd/main -- metadata testdata/basic_v0.h5 /matrix
moon run src/cmd/main -- stats testdata/basic_v0.h5 /matrix
moon run src/cmd/main -- read testdata/basic_v0.h5 /matrix 0,0 2,2 1,2
moon run src/cmd/main -- read testdata/basic_v0.h5 /matrix --points "1,2;0,1"
```

Coordinates are zero-based. Hyperslabs use comma-separated `start`, `count` and
`stride` vectors. Point selections use semicolon-separated coordinates and
preserve order and duplicates.

## Limitations

- Whole-dataset reads and single selection results remain memory resident even
  when the input uses random access.
- Use `chunk_iterator` when the complete dataset is too large for memory.
- The writer creates complete new files and does not edit existing files.
- Dense metadata writing is limited to 9–65,535 links or attributes with each
  encoded heap object at most 65,535 bytes. It uses one FHDB direct block and
  one BTHD/BTLF leaf name index plus a creation-order index per collection;
  dense objects above that size, indirect blocks and a free-space manager are
  not written. Attribute sets above the record or size limit fall back to
  compact OCHK continuation storage.
- External raw-data storage is a post-v1.0 extension. `Hdf5Resolver` reads it
  when every external file is registered as an `Hdf5Source`; `Hdf5File` rejects
  the dataset. The writer creates contiguous, unfiltered external datasets from
  caller-supplied slots. Remote URLs, filtered or chunked external layouts and
  automatic external-file creation are not supported. Tile iteration is
  supported for reads.
- Optional writer SOHM support covers whole attributes up to 4 KiB,
  deduplicated datatype/dataspace fields for larger attributes, and custom fill
  messages up to 4 KiB through one combined SMTB index and one managed-object
  fractal heap. It does not emit committed-datatype targets, and it does not
  deduplicate equal whole attributes.
- Numeric storage wider than 8 bytes and binary floating-point normalizations
  other than implied MSB are not supported.
- Floating-point scale-offset supports f32/f64 decimal scaling; other floating
  scale modes are rejected. Writer scale-offset must be the first filter and
  rejects automatic bit reduction when a chunk needs full precision; use
  `ScaleOffset(element_width * 8)` to store an integer chunk unfiltered. Writer
  floating-point data and fill values must be finite, and floating-point
  scale-offset cannot be combined with Fletcher32.
- Custom variable-length sequence fills and other writer filters are not
  supported. Complex output is limited to little-endian 64- and 128-bit values;
  variable-length sequence bases must be fixed-size, and a base that is or
  contains a region reference is rejected because HDF5 1.14.6 cannot safely
  consume that combination.
- Reference resolution covers legacy object and dataset-region references under
  4- and 8-byte address widths. Region targets must be datasets in the same
  file. Version-4 encoded references are not resolved. Region-reference writing
  covers the forms listed above; version-4 encoded references and region
  references inside variable-length sequences are not written.
- Variable-length writer datasets cannot use filters; HDF5 1.14.6 rejects this
  combination natively.

Version support does not imply support for every structure in that version.
Unsupported structures raise `UnsupportedFeature`; malformed data normally
raises `InvalidField` or `UnexpectedEof`. Invalid selections and writer inputs
raise `InvalidArgument`.

## Development

```sh
moon info
moon fmt
moon check
moon test --target wasm
moon test --target native
```

Run `scripts/release_check.sh` to include interface and formatting checks,
documentation generation, package contents, the writer demo and independent
h5py verification, including the hard-link and MATLAB 7.3-compatible fixtures,
the committed NetCDF-4 producer fixture, and the external raw-data and
shared-message fixtures. Set `H5PY_PYTHON` when h5py is installed in a specific
environment, or `SKIP_H5PY=1` to omit the Python interoperability checks.

The v1.0 release gate runs the writer demo and verifies its output with h5py
3.13.0/HDF5 1.14.6, then verifies the numeric-layout, hard-link and MATLAB
7.3-compatible fixtures. It also verifies the NetCDF-4 fixture produced by
netCDF4 1.7.5/libnetcdf 4.10.1/HDF5 2.2.0, the post-v1.0 external raw-data
fixture set, the writer's external slot declarations and values, and the
shared-message fixture. The HDF5 1.14.6 default and `latest`
datatype fixtures use the pinned environment documented in
[testdata/README.md](testdata/README.md); fixture tests compare decoded
semantics because binary hashes are not stable.

GitHub Actions CI is manual and runs checks, builds and wasm/native tests.
Fixture generation is documented in
[testdata/README.md](testdata/README.md). Architecture and implementation notes
are in [docs/design.md](docs/design.md); external sources and ecosystem projects
are listed in [docs/prior-art.md](docs/prior-art.md).

## License

Apache-2.0. Zlib decompression uses
[MoonZip](https://github.com/moonbitstack/moonzip), also under Apache-2.0.
