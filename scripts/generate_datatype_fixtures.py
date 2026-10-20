from pathlib import Path

import h5py
import numpy as np


def write_low_level_dataset(
    handle: h5py.File,
    name: bytes,
    file_type: h5py.h5t.TypeID,
    memory_type: h5py.h5t.TypeID,
    values: np.ndarray,
) -> None:
    space = h5py.h5s.create_simple(values.shape)
    dataset = h5py.h5d.create(handle.id, name, file_type, space)
    dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, values, mtype=memory_type)


def generate(destination: Path) -> None:
    inner = np.dtype([("left", "<i2"), ("right", "<i2")])
    compound = np.dtype(
        [
            ("id", "<i4"),
            ("weight", ">f8"),
            ("label", "S5"),
            ("pair", inner),
            ("matrix", ("<u2", (2, 2))),
        ]
    )
    compound_values = np.zeros(3, dtype=compound)
    compound_values["id"] = [7, -2, 99]
    compound_values["weight"] = [1.5, -2.25, 0.0]
    compound_values["label"] = [b"alpha", b"xy", b"z"]
    compound_values["pair"]["left"] = [1, 3, 5]
    compound_values["pair"]["right"] = [2, 4, 6]
    compound_values["matrix"] = [
        [[10, 11], [12, 13]],
        [[20, 21], [22, 23]],
        [[30, 31], [32, 33]],
    ]

    array_type = np.dtype((">i2", (2, 3)))
    array_values = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[-1, -2, -3], [-4, -5, -6]],
        ],
        dtype=">i2",
    )

    sequences = np.empty(3, dtype=object)
    sequences[0] = np.array([1, -2, 3], dtype="<i2")
    sequences[1] = np.array([], dtype="<i2")
    sequences[2] = np.array([300, 400], dtype="<i2")

    with h5py.File(destination, "w", libver=("v200", "v200")) as handle:
        handle.create_dataset(
            "compound",
            data=compound_values,
            track_times=False,
        )
        handle.create_dataset(
            "compound_fill",
            shape=(2,),
            dtype=compound,
            fillvalue=compound_values[1],
            track_times=False,
        )
        array_dataset = handle.create_dataset(
            "array", shape=(2,), dtype=array_type, track_times=False
        )
        array_dataset[...] = array_values
        handle.create_dataset(
            "sequences",
            data=sequences,
            dtype=h5py.vlen_dtype(np.dtype("<i2")),
            track_times=False,
        )
        handle.create_dataset(
            "utf8",
            data=np.array(["月亮", "HDF5", ""], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
            track_times=False,
        )
        opaque_type = h5py.opaque_dtype(np.dtype("V4"))
        handle.create_dataset(
            "opaque",
            data=np.array([b"\x00\x01\xfe\xff", b"MOON"], dtype=opaque_type),
            track_times=False,
        )
        write_low_level_dataset(
            handle,
            b"bitfield",
            h5py.h5t.STD_B16LE,
            h5py.h5t.NATIVE_UINT16,
            np.array([0x0001, 0x00F0, 0xA55A], dtype=np.uint16),
        )
        write_low_level_dataset(
            handle,
            b"time",
            h5py.h5t.UNIX_D64LE,
            h5py.h5t.UNIX_D64LE,
            np.array([0, 1_700_000_000], dtype="<u8"),
        )
        target = handle.create_dataset("target", data=np.arange(6, dtype="<i4"))
        object_refs = np.array([target.ref, h5py.Reference()], dtype=h5py.ref_dtype)
        handle.create_dataset("object_refs", data=object_refs, track_times=False)
        region = target.regionref[2:5]
        region_refs = np.array([region], dtype=h5py.regionref_dtype)
        handle.create_dataset("region_refs", data=region_refs, track_times=False)
        write_low_level_dataset(
            handle,
            b"complex64",
            h5py.h5t.COMPLEX_IEEE_F32LE,
            h5py.h5t.NATIVE_FLOAT_COMPLEX,
            np.array([1 + 2j, -3.5 + 0.25j], dtype=np.complex64),
        )
        write_low_level_dataset(
            handle,
            b"complex128_be",
            h5py.h5t.COMPLEX_IEEE_F64BE,
            h5py.h5t.NATIVE_DOUBLE_COMPLEX,
            np.array([2 - 4j], dtype=np.complex128),
        )
        handle["compound"].attrs.create("record", compound_values[1])
        handle["array"].attrs.create("grid", array_values[0], dtype=array_type)

    with h5py.File(destination, "r") as handle:
        np.testing.assert_array_equal(handle["compound"][:], compound_values)
        np.testing.assert_array_equal(
            handle["compound_fill"][:], np.repeat(compound_values[1:2], 2)
        )
        np.testing.assert_array_equal(handle["array"][:], array_values)
        for actual, expected in zip(handle["sequences"][:], sequences):
            np.testing.assert_array_equal(actual, expected)
        assert handle["utf8"].asstr()[:].tolist() == ["月亮", "HDF5", ""]
        assert handle["opaque"].dtype.metadata == {"h5py_opaque": True}
        np.testing.assert_array_equal(handle["bitfield"][:], [1, 240, 42330])
        assert bool(handle["object_refs"][0])
        assert not bool(handle["object_refs"][1])
        region_ref = handle["region_refs"][0]
        np.testing.assert_array_equal(handle[region_ref][region_ref], [2, 3, 4])
        np.testing.assert_array_equal(
            handle["complex64"][:], np.array([1 + 2j, -3.5 + 0.25j], dtype="<c8")
        )
        np.testing.assert_array_equal(
            handle["complex128_be"][:], np.array([2 - 4j], dtype=">c16")
        )
    print(
        f"Verified {destination.name} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "datatypes_v200.h5")
