# Design notes

`moon-hdf5` is a read-only, dependency-light HDF5 reader. The implementation is
split into framing, metadata parsing, data decoding and a small high-level API.

## Reading pipeline

1. `superblock.mbt` locates the signature (including user blocks) and parses the
   address and length sizes.
2. `object_header.mbt` reads v1/v2 object headers and follows continuations.
3. Group metadata is decoded from symbol-table B-trees (`heap.mbt`,
   `symbol_table.mbt`) or from link messages (`link.mbt`).
4. Dataset metadata is decoded from dataspace, datatype and data-layout
   messages. `dataset.mbt` handles compact and contiguous storage;
   `chunked.mbt` reconstructs unfiltered chunked storage from the v1 chunk
   B-tree.
5. `global_heap.mbt` resolves variable-length strings, and `attribute.mbt`
   decodes attribute messages v1/v2/v3.
6. `hdf5_file.mbt` provides path lookup, soft-link following, tree walking and
   convenience methods for datasets and attributes.

## Error policy

Unsupported structures raise `UnsupportedFeature` rather than returning
plausible-looking but incorrect data. This applies to filtered chunks, layout
message v4, dense attribute storage, external links, unsupported datatype
classes and v2/fractal-heap chunk indexes.

All byte reads go through `reader.mbt`, which checks bounds before every access.
HDF5 addresses are converted to file offsets with the superblock base address,
so files with a user block work without special cases in the decoders.

## Testing strategy

The committed fixtures cover:

- legacy and latest superblocks, including a 512-byte user block;
- object header v1 and v2 with continuation blocks;
- symbol-table groups and link-message groups;
- hard links, soft links and external-link detection;
- signed/unsigned integers, floats, fixed and variable-length strings;
- scalar datasets, contiguous datasets and chunked datasets with edge chunks;
- attributes in legacy and latest formats.

Tests run on both `wasm` and `native` targets. CI executes `moon check` and
`moon test` on every push and pull request.
