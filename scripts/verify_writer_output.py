import sys
from pathlib import Path

import h5py
import numpy as np


def verify(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        assert list(handle) == ["experiment", "structured", "types"]
        assert sorted(handle["experiment"]) == [
            "chunked",
            "counts",
            "events",
            "filtered_names",
            "labels",
            "metrics",
            "names",
            "series",
        ]
        np.testing.assert_array_equal(
            handle["experiment/counts"][:], [[1, 2, 3], [4, 5, 6]]
        )
        assert handle["experiment/counts"].dtype == np.dtype("<i8")
        assert handle["experiment/labels"].asstr()[:].tolist() == [
            "control",
            "sample",
        ]
        np.testing.assert_array_equal(
            handle["experiment/metrics"][:], [1.5, -2.25]
        )
        assert handle["experiment/metrics"].dtype == np.dtype("<f8")
        assert handle["experiment"].attrs["revision"] == np.uint64(1)
        chunked = handle["experiment/chunked"]
        expected = np.full((5, 7), -7, dtype="<i8")
        expected[0, 0] = 42
        expected[2, 3] = 99
        expected[4, 6] = -2
        np.testing.assert_array_equal(chunked[:], expected)
        assert chunked.chunks == (2, 3)
        assert chunked.fillvalue == -7
        assert chunked.shuffle
        assert chunked.compression == "gzip"
        assert chunked.compression_opts == 6
        assert chunked.fletcher32
        series = handle["experiment/series"]
        np.testing.assert_array_equal(
            series[:], [[10, 11], [20, 21], [30, 31], [-1, -1]]
        )
        assert series.maxshape == (None, 2)
        assert series.chunks == (2, 2)
        assert series.fillvalue == -1
        assert series.shuffle
        assert series.compression == "gzip"
        assert series.compression_opts == 6
        assert series.fletcher32
        names = handle["experiment/names"]
        assert h5py.check_string_dtype(names.dtype).encoding == "utf-8"
        assert names.asstr()[:].tolist() == ["月亮", "HDF5", "", "月亮"]
        assert names.attrs["description"] == "UTF-8 标签"
        filtered_names = handle["experiment/filtered_names"]
        assert filtered_names.asstr()[:].tolist() == [
            "alpha",
            "",
            "数据",
            "alpha",
            "",
        ]
        assert filtered_names.chunks == (2,)
        assert filtered_names.compression == "gzip"
        assert filtered_names.fletcher32
        events = handle["experiment/events"]
        assert events.asstr()[:].tolist() == ["start", "结束", ""]
        assert events.maxshape == (None,)
        assert events.chunks == (2,)
        unit = handle["experiment/counts"].attrs["unit"]
        assert unit.decode("utf-8").rstrip("\0") == "items"
        cases = {
            "i8": ("<i1", [-128, 0, 127]),
            "i16": ("<i2", [-32768, 0, 32767]),
            "i32": ("<i4", [-2147483647, 0, 2147483647]),
            "u8": ("<u1", [0, 1, 255]),
            "u16": ("<u2", [0, 1, 65535]),
            "u32": ("<u4", [0, 1, 0xFFFFFFFF]),
            "f32": ("<f4", [-1.5, 0.0, 2.25]),
        }
        for name, (dtype, values) in cases.items():
            dataset = handle[f"types/{name}"]
            assert dataset.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(dataset[:], values)
        assert handle["types"].attrs["i8_marker"] == np.int8(-5)
        assert handle["types"].attrs["i8_marker"].dtype == np.dtype("<i1")
        assert handle["types"].attrs["f32_marker"] == np.float32(1.5)
        assert handle["types"].attrs["f32_marker"].dtype == np.dtype("<f4")
        chunked_i16 = handle["types/chunked_i16"]
        np.testing.assert_array_equal(chunked_i16[:], [-7, 2, -7, 4, -7])
        assert chunked_i16.dtype == np.dtype("<i2")
        assert chunked_i16.fillvalue == -7
        assert chunked_i16.shuffle
        assert chunked_i16.compression == "gzip"
        assert chunked_i16.fletcher32
        growing_f32 = handle["types/growing_f32"]
        np.testing.assert_array_equal(growing_f32[:], [1.25, 2.5, 3.75, -1.0])
        assert growing_f32.dtype == np.dtype("<f4")
        assert growing_f32.maxshape == (None,)
        assert growing_f32.shuffle
        assert growing_f32.compression == "gzip"
        assert growing_f32.fletcher32
        records = handle["structured/records"]
        assert records.dtype.names == ("id", "state", "samples", "label")
        assert records.dtype.itemsize == 19
        assert [records.dtype.fields[name][1] for name in records.dtype.names] == [
            0,
            4,
            5,
            13,
        ]
        assert records.dtype.fields["id"][0] == np.dtype("<i4")
        assert records.dtype.fields["state"][0].metadata["enum"] == {
            "UNKNOWN": 0,
            "OPEN": 1,
            "CLOSED": 2,
        }
        assert records.dtype.fields["samples"][0].subdtype == (
            np.dtype("<i2"),
            (2, 2),
        )
        assert records.dtype.fields["label"][0] == np.dtype("S6")
        assert records[0]["id"] == 7
        assert records[0]["state"] == 1
        np.testing.assert_array_equal(records[0]["samples"], [[1, 2], [3, 4]])
        assert records[0]["label"] == b"alpha"
        assert records[1]["state"] == 0
        assert records[2]["id"] == -2
        assert records.fillvalue["state"] == 0
        assert records.shuffle
        assert records.compression == "gzip"
        assert records.fletcher32
        assert records.attrs["prototype"]["id"] == 7
        vectors = handle["structured/vectors"]
        assert vectors.dtype.subdtype == (np.dtype("<f4"), (2,))
        np.testing.assert_array_equal(vectors[:], [[1.5, -2.0], [3.25, 4.5]])
        states = handle["structured/states"]
        assert states.dtype.metadata["enum"] == {
            "UNKNOWN": 0,
            "OPEN": 1,
            "CLOSED": 2,
        }
        assert states.maxshape == (None,)
        np.testing.assert_array_equal(states[:], [1, 2, 0, 0])
    with h5py.File(path, "r+") as handle:
        series = handle["experiment/series"]
        series.resize((5, 2))
        series[4] = [40, 41]
        np.testing.assert_array_equal(series[4], [40, 41])
        events = handle["experiment/events"]
        events.resize((4,))
        events[3] = "continued"
        assert events.asstr()[3] == "continued"
    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_writer_output.py <file.h5>")
    verify(Path(sys.argv[1]))
