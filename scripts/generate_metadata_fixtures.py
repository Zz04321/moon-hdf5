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

        signed_20 = h5py.h5t.STD_I32LE.copy()
        signed_20.set_size(3)
        signed_20.set_precision(20)
        signed_20.set_offset(0)
        signed_20.set_pad(h5py.h5t.PAD_ZERO, h5py.h5t.PAD_ONE)
        signed_20_values = np.array([42, -2, -524288, 524287], dtype="<i4")
        signed_20_dataset = h5py.h5d.create(
            handle.id,
            b"signed_20",
            signed_20,
            h5py.h5s.create_simple(signed_20_values.shape),
        )
        signed_20_dataset.write(
            h5py.h5s.ALL,
            h5py.h5s.ALL,
            signed_20_values,
            mtype=h5py.h5t.STD_I32LE,
        )
        signed_20_attribute = h5py.h5a.create(
            signed_20_dataset,
            b"custom",
            signed_20,
            h5py.h5s.create_simple(signed_20_values.shape),
        )
        signed_20_attribute.write(signed_20_values, mtype=h5py.h5t.STD_I32LE)

        unsigned_40 = h5py.h5t.STD_U64BE.copy()
        unsigned_40.set_size(5)
        unsigned_40.set_precision(40)
        unsigned_40.set_offset(0)
        unsigned_40.set_order(h5py.h5t.ORDER_BE)
        unsigned_40.set_pad(h5py.h5t.PAD_ZERO, h5py.h5t.PAD_ZERO)
        unsigned_40_values = np.array([0, 1, 0xFFFFFFFFFF], dtype="<u8")
        unsigned_40_dataset = h5py.h5d.create(
            handle.id,
            b"unsigned_40_be",
            unsigned_40,
            h5py.h5s.create_simple(unsigned_40_values.shape),
        )
        unsigned_40_dataset.write(
            h5py.h5s.ALL,
            h5py.h5s.ALL,
            unsigned_40_values,
            mtype=h5py.h5t.STD_U64LE,
        )
        unsigned_40_attribute = h5py.h5a.create(
            unsigned_40_dataset,
            b"custom",
            unsigned_40,
            h5py.h5s.create_simple(unsigned_40_values.shape),
        )
        unsigned_40_attribute.write(
            unsigned_40_values, mtype=h5py.h5t.STD_U64LE
        )

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
        datatype = h5py.h5t.IEEE_F32LE.copy()
        datatype.set_fields(26, 20, 6, 0, 13)
        datatype.set_ebias(20)
        datatype.set_norm(h5py.h5t.NORM_IMPLIED)
        datatype.set_offset(0)
        custom_float_values = np.array([1.5, -2.25, 0.125], dtype="<f4")
        dataset = h5py.h5d.create(
            handle.id,
            b"custom_binary_float",
            datatype,
            h5py.h5s.create_simple(custom_float_values.shape),
        )
        dataset.write(
            h5py.h5s.ALL,
            h5py.h5s.ALL,
            custom_float_values,
            mtype=h5py.h5t.IEEE_F32LE,
        )
        attribute = h5py.h5a.create(
            dataset,
            b"custom",
            datatype,
            h5py.h5s.create_simple(custom_float_values.shape),
        )
        attribute.write(custom_float_values, mtype=h5py.h5t.IEEE_F32LE)
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
        custom_float = h5py.h5d.open(handle.id, b"custom_binary_float")
        custom_float_values = np.array([1.5, -2.25, 0.125], dtype="<f4")
        actual = np.empty((3,), dtype="<f4")
        custom_float.read(
            h5py.h5s.ALL,
            h5py.h5s.ALL,
            actual,
            mtype=h5py.h5t.IEEE_F32LE,
        )
        np.testing.assert_array_equal(actual, custom_float_values)
        custom_float_attribute = h5py.h5a.open(custom_float, b"custom")
        actual = np.empty((3,), dtype="<f4")
        custom_float_attribute.read(actual, mtype=h5py.h5t.IEEE_F32LE)
        np.testing.assert_array_equal(actual, custom_float_values)
        for name, memory_type, expected in (
            (
                b"signed_20",
                h5py.h5t.STD_I32LE,
                np.array([42, -2, -524288, 524287], dtype="<i4"),
            ),
            (
                b"unsigned_40_be",
                h5py.h5t.STD_U64LE,
                np.array([0, 1, 0xFFFFFFFFFF], dtype="<u8"),
            ),
        ):
            dataset = h5py.h5d.open(handle.id, name)
            actual = np.empty(
                dataset.get_space().get_simple_extent_dims(),
                dtype=expected.dtype,
            )
            dataset.read(h5py.h5s.ALL, h5py.h5s.ALL, actual, mtype=memory_type)
            np.testing.assert_array_equal(actual, expected)
            attribute = h5py.h5a.open(dataset, b"custom")
            actual = np.empty(
                attribute.get_space().get_simple_extent_dims(),
                dtype=expected.dtype,
            )
            attribute.read(actual, mtype=memory_type)
            np.testing.assert_array_equal(actual, expected)
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
