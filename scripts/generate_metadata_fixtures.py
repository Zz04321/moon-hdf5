from pathlib import Path

import h5py
import numpy as np


def custom_integer(base, precision: int, offset: int):
    datatype = base.copy()
    datatype.set_precision(precision)
    datatype.set_offset(offset)
    return datatype


def generate(directory: Path) -> None:
    with h5py.File(directory / "numeric_layouts.h5", "w") as handle:
        integer_cases = (
            (
                "offset",
                h5py.h5t.STD_I32LE,
                16,
                8,
                np.array([42, -2, -32768], dtype="<i4"),
            ),
            (
                "precision",
                h5py.h5t.STD_I32LE,
                16,
                0,
                np.array([42, -2, -32768], dtype="<i4"),
            ),
            (
                "unsigned",
                h5py.h5t.STD_U32LE,
                20,
                4,
                np.array([0, 1, 703710, 1048575], dtype="<u4"),
            ),
            (
                "big_endian",
                h5py.h5t.STD_I32BE,
                20,
                4,
                np.array([-2, 42, 524287], dtype=">i4"),
            ),
        )
        for name, base, precision, offset, values in integer_cases:
            datatype = custom_integer(base, precision, offset)
            dataset = h5py.h5d.create(
                handle.id,
                name.encode(),
                datatype,
                h5py.h5s.create_simple(values.shape),
            )
            memory_type = h5py.h5t.py_create(values.dtype)
            dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, values, mtype=memory_type)
            attribute = h5py.h5a.create(
                dataset,
                b"custom",
                datatype,
                h5py.h5s.create_simple(values.shape),
            )
            attribute.write(values, mtype=memory_type)

        fill_type = custom_integer(h5py.h5t.STD_I32LE, 16, 8)
        dataset_creation = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dataset_creation.set_fill_value(np.array([-2], dtype="<i4"))
        h5py.h5d.create(
            handle.id,
            b"fill",
            fill_type,
            h5py.h5s.create_simple((3,)),
            dcpl=dataset_creation,
        )
        datatype = h5py.h5t.IEEE_F32LE.copy()
        datatype.set_ebias(126)
        dataset = h5py.h5d.create(
            handle.id, b"custom_float", datatype, h5py.h5s.create_simple((1,))
        )
        dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([1.5], dtype="<f4"))
        for endian in ("<", ">"):
            prefix = "le" if endian == "<" else "be"
            for kind in ("i", "u", "f"):
                for width in ((2, 4, 8) if kind == "f" else (1, 2, 4, 8)):
                    handle.create_dataset(
                        f"{prefix}_{kind}{width}",
                        data=np.array([1, 42], dtype=f"{endian}{kind}{width}"),
                        track_times=False,
                    )
        half_dataset = handle.create_dataset(
            "half_special",
            data=np.array(
                [
                    1.0,
                    -2.5,
                    65504.0,
                    6.103515625e-05,
                    5.960464477539063e-08,
                    0.0,
                    -0.0,
                ],
                dtype="<f2",
            ),
            track_times=False,
        )
        half_dataset.attrs.create(
            "custom", np.array([1.0, -2.5, 65504.0], dtype="<f2")
        )
        handle.create_dataset(
            "half_fill", shape=(3,), dtype="<f2", fillvalue=-2.5,
            track_times=False,
        )
    with h5py.File(directory / "numeric_layouts.h5", "r") as handle:
        expected = {
            "offset": [42, -2, -32768],
            "precision": [42, -2, -32768],
            "unsigned": [0, 1, 703710, 1048575],
            "big_endian": [-2, 42, 524287],
            "fill": [-2, -2, -2],
            "custom_float": [1.5],
        }
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
        for name in ("offset", "precision", "unsigned", "big_endian"):
            np.testing.assert_array_equal(handle[name].attrs["custom"], expected[name])
        for endian in ("<", ">"):
            prefix = "le" if endian == "<" else "be"
            for kind in ("i", "u", "f"):
                for width in ((2, 4, 8) if kind == "f" else (1, 2, 4, 8)):
                    name = f"{prefix}_{kind}{width}"
                    np.testing.assert_array_equal(handle[name][:], [1, 42])
        np.testing.assert_array_equal(
            handle["half_special"][:],
            np.array(
                [
                    1.0,
                    -2.5,
                    65504.0,
                    6.103515625e-05,
                    5.960464477539063e-08,
                    0.0,
                    -0.0,
                ],
                dtype="<f2",
            ),
        )
        np.testing.assert_array_equal(
            handle["half_special"].attrs["custom"], [1.0, -2.5, 65504.0]
        )
        np.testing.assert_array_equal(handle["half_fill"][:], [-2.5, -2.5, -2.5])
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
