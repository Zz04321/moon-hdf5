from pathlib import Path

import h5py
import numpy as np


def generate(directory: Path) -> None:
    with h5py.File(directory / "numeric_layouts.h5", "w") as handle:
        for name, precision, offset in (("offset", 16, 8), ("precision", 16, 0)):
            datatype = h5py.h5t.STD_I32LE.copy()
            datatype.set_precision(precision)
            datatype.set_offset(offset)
            dataset = h5py.h5d.create(
                handle.id, name.encode(), datatype, h5py.h5s.create_simple((1,))
            )
            dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([42], dtype="<i4"))
            attribute = h5py.h5a.create(dataset, b"custom", datatype, h5py.h5s.create_simple((1,)))
            attribute.write(np.array([42], dtype="<i4"))
        datatype = h5py.h5t.IEEE_F32LE.copy()
        datatype.set_ebias(126)
        dataset = h5py.h5d.create(
            handle.id, b"custom_float", datatype, h5py.h5s.create_simple((1,))
        )
        dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([1.5], dtype="<f4"))
        for endian in ("<", ">"):
            prefix = "le" if endian == "<" else "be"
            for kind in ("i", "u", "f"):
                for width in ((4, 8) if kind == "f" else (1, 2, 4, 8)):
                    handle.create_dataset(
                        f"{prefix}_{kind}{width}",
                        data=np.array([1, 42], dtype=f"{endian}{kind}{width}"),
                        track_times=False,
                    )
    with h5py.File(directory / "numeric_layouts.h5", "r") as handle:
        for name in handle:
            expected = [42] if name in ("offset", "precision") else (
                [1.5] if name == "custom_float" else [1, 42]
            )
            np.testing.assert_array_equal(handle[name][:], expected)
            if name in ("offset", "precision"):
                np.testing.assert_array_equal(handle[name].attrs["custom"], [42])
    with h5py.File(directory / "attribute_storage.h5", "w", libver=("v108", "v108")) as handle:
        for name, count, ordered in (
            ("compact", 2, False), ("compact_ordered", 2, True),
            ("dense", 12, False), ("dense_ordered", 12, True),
        ):
            dataset = handle.create_dataset(name, data=[42], track_order=ordered)
            for index in range(count):
                dataset.attrs[f"attr_{index:02d}"] = index
    with h5py.File(directory / "attribute_storage.h5", "r") as handle:
        for name in handle:
            count = 12 if name.startswith("dense") else 2
            assert dict(handle[name].attrs) == {f"attr_{index:02d}": index for index in range(count)}
    print(f"Verified numeric layouts and attribute storage with h5py {h5py.__version__}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "testdata")
