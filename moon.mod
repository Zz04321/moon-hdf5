// Learn more about moon.mod configuration:
// https://docs.moonbitlang.com/en/latest/toolchain/moon/module.html
//
// To add a dependency, run this command in your terminal:
//   moon add moonbitlang/x
//
// Or manually declare it in `import`, for example:
// import {
//   "moonbitlang/x@0.4.6",
// }

name = "Zz04321/moon-hdf5"

source = "src"

version = "0.1.0"

readme = "README.mbt.md"

repository = "https://github.com/Zz04321/moon-hdf5"

license = "Apache-2.0"

keywords = [ "hdf5", "scientific-data", "file-format", "reader", "writer" ]

preferred_target = "wasm"

description = "A pure MoonBit HDF5 reader and new-file writer"

import {
  "moonbitlang/x@0.5.5",
  "moonbitstack/moonzip@0.3.0",
}
