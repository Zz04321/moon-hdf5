import sys
from pathlib import Path

import h5py
import numpy as np


def filter_records(dataset):
    plist = dataset.id.get_create_plist()
    return [plist.get_filter(index) for index in range(plist.get_nfilters())]


def verify(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        assert list(handle) == ["experiment", "metadata", "structured", "types"]
        assert sorted(handle["experiment"]) == [
            "bounded_names",
            "chunked",
            "chunked_names",
            "counts",
            "events",
            "external",
            "labels",
            "latest",
            "metrics",
            "names",
            "paged_series",
            "series",
        ]
        np.testing.assert_array_equal(
            handle["experiment/counts"][:], [[1, 2, 3], [4, 5, 6]]
        )
        assert handle["experiment/counts"].dtype == np.dtype("<i8")
        latest = handle.get("experiment/latest", getlink=True)
        assert isinstance(latest, h5py.SoftLink)
        assert latest.path == "/experiment/counts"
        np.testing.assert_array_equal(
            handle["experiment/latest"][:], handle["experiment/counts"][:]
        )
        external = handle.get("experiment/external", getlink=True)
        assert isinstance(external, h5py.ExternalLink)
        assert external.filename == "external-target.h5"
        assert external.path == "/experiment/counts"
        assert handle["experiment/labels"].asstr()[:].tolist() == [
            "control",
            "sample",
        ]
        np.testing.assert_array_equal(
            handle["experiment/metrics"][:], [1.5, -2.25]
        )
        assert handle["experiment/metrics"].dtype == np.dtype("<f8")
        assert handle["experiment"].attrs["revision"] == np.uint64(1)
        assert len(handle["experiment"].attrs) == 1
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
        chunked_names = handle["experiment/chunked_names"]
        assert chunked_names.asstr()[:].tolist() == [
            "alpha",
            "",
            "数据",
            "alpha",
            "",
        ]
        assert chunked_names.chunks == (2,)
        assert chunked_names.compression is None
        assert not chunked_names.fletcher32
        assert chunked_names.id.get_num_chunks() == 3
        bounded_names = handle["experiment/bounded_names"]
        assert bounded_names.shape == (1, 2)
        assert bounded_names.maxshape == (None, 4)
        assert bounded_names.chunks == (1, 2)
        assert bounded_names.id.get_num_chunks() == 1
        assert bounded_names.asstr()[:].tolist() == [["a", "b"]]
        events = handle["experiment/events"]
        assert events.asstr()[:].tolist() == ["start", "结束", ""]
        assert events.maxshape == (None,)
        assert events.chunks == (2,)
        assert events.id.get_num_chunks() == 2
        unit = handle["experiment/counts"].attrs["unit"]
        assert unit.decode("utf-8").rstrip("\0") == "items"
        assert len(handle["experiment/counts"].attrs) == 1
        cases = {
            "i8": ("<i1", [-128, 0, 127]),
            "i16": ("<i2", [-32768, 0, 32767]),
            "i32": ("<i4", [-2147483647, 0, 2147483647]),
            "u8": ("<u1", [0, 1, 255]),
            "u16": ("<u2", [0, 1, 65535]),
            "u32": ("<u4", [0, 1, 0xFFFFFFFF]),
            "f32": ("<f4", [-1.5, 0.0, 2.25]),
            "big_i16": (">i2", [-32768, -1, 12345]),
            "big_u64": (">u8", [0, 1 << 40, 0xFFFFFFFFFFFFFFFF]),
            "big_f32": (">f4", [-1.5, 0.0, 2.25]),
            "big_f64": (">f8", [-1.25, 0.0, 3.5]),
        }
        for name, (dtype, values) in cases.items():
            dataset = handle[f"types/{name}"]
            assert dataset.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(dataset[:], values)
        assert handle["types"].attrs["i8_marker"] == np.int8(-5)
        assert len(handle["types"].attrs) == 3
        assert handle["types"].attrs["i8_marker"].dtype == np.dtype("<i1")
        assert handle["types"].attrs["f32_marker"] == np.float32(1.5)
        assert handle["types"].attrs["f32_marker"].dtype == np.dtype("<f4")
        assert handle["types"].attrs["big_i32_marker"] == np.int32(-5)
        assert handle["types"].attrs.get_id("big_i32_marker").dtype == np.dtype(">i4")
        big_state = handle["types/big_state"]
        assert big_state.dtype == np.dtype(">i2")
        assert big_state.dtype.metadata["enum"] == {"LOW": -1, "HIGH": 300}
        np.testing.assert_array_equal(big_state[:], [300, -1])
        big_record = handle["types/big_record"]
        assert big_record.dtype.names == ("id", "weight", "count", "state", "samples")
        assert big_record.dtype.itemsize == 24
        assert [big_record.dtype.fields[name][1] for name in big_record.dtype.names] == [
            0,
            4,
            12,
            14,
            16,
        ]
        assert big_record.dtype.fields["id"][0] == np.dtype("<i4")
        assert big_record.dtype.fields["weight"][0] == np.dtype(">f8")
        assert big_record.dtype.fields["count"][0] == np.dtype(">u2")
        state_dtype = big_record.dtype.fields["state"][0]
        assert state_dtype == np.dtype(">i2")
        assert state_dtype.metadata["enum"] == {"LOW": -1, "HIGH": 300}
        assert big_record.dtype.fields["samples"][0].subdtype == (
            np.dtype(">u4"),
            (2,),
        )
        assert big_record[0]["id"] == 7
        assert big_record[0]["weight"] == -2.5
        assert big_record[0]["count"] == 513
        assert big_record[0]["state"] == 300
        np.testing.assert_array_equal(
            big_record[0]["samples"], [1, 0x10203040]
        )
        big_fill = handle["types/big_fill"]
        assert big_fill.dtype == np.dtype(">i2")
        assert big_fill.fillvalue == -999
        np.testing.assert_array_equal(big_fill[:], [7, -999, -999, -999])
        chunked_i16 = handle["types/chunked_i16"]
        np.testing.assert_array_equal(chunked_i16[:], [-7, 2, -7, 4, -7])
        assert chunked_i16.dtype == np.dtype("<i2")
        assert chunked_i16.fillvalue == -7
        assert chunked_i16.shuffle
        assert chunked_i16.compression == "gzip"
        assert chunked_i16.fletcher32
        scaleoffset_i32 = handle["types/scaleoffset_i32"]
        np.testing.assert_array_equal(
            scaleoffset_i32[:], [10, -7, 12, 13, 100, -7, 101, 102]
        )
        assert scaleoffset_i32.fillvalue == -7
        (filter_id, flags, parameters, name), = filter_records(scaleoffset_i32)
        assert (filter_id, flags, name) == (6, 1, b"scaleoffset")
        assert parameters[:8] == (2, 0, 4, 0, 4, 1, 0, 1)
        assert parameters[8] == 0xFFFFFFF9
        scaleoffset_pipeline = handle["types/scaleoffset_pipeline"]
        np.testing.assert_array_equal(
            scaleoffset_pipeline[:], [100, 101, 102, 200, 201, 202]
        )
        assert scaleoffset_pipeline.fillvalue == -1
        assert scaleoffset_pipeline.fletcher32
        records = filter_records(scaleoffset_pipeline)
        assert [(record[0], record[1], record[3]) for record in records] == [
            (6, 1, b"scaleoffset"),
            (1, 0, b"deflate"),
            (3, 0, b"fletcher32"),
        ]
        assert records[0][2][:8] == (2, 4, 3, 0, 2, 1, 0, 1)
        assert records[0][2][8] == 0xFFFF
        scaleoffset_u64 = handle["types/scaleoffset_u64"]
        np.testing.assert_array_equal(
            scaleoffset_u64[:], [1 << 40, (1 << 40) + 1, (1 << 40) + 2]
        )
        assert scaleoffset_u64.fillvalue == 7
        (filter_id, flags, parameters, name), = filter_records(scaleoffset_u64)
        assert (filter_id, flags, name) == (6, 1, b"scaleoffset")
        assert parameters[:8] == (2, 0, 3, 0, 8, 0, 0, 1)
        assert parameters[8:10] == (7, 0)
        scaleoffset_full = handle["types/scaleoffset_full"]
        np.testing.assert_array_equal(scaleoffset_full[:], [-5, -4, -3, -2])
        assert scaleoffset_full.fillvalue == 0
        (filter_id, flags, parameters, name), = filter_records(scaleoffset_full)
        assert (filter_id, flags, name) == (6, 1, b"scaleoffset")
        assert parameters[:8] == (2, 32, 2, 0, 4, 1, 0, 1)
        scaleoffset_f32 = handle["types/scaleoffset_f32"]
        np.testing.assert_array_equal(
            scaleoffset_f32[:], [1.25, 2.5, -1.75, 9.5, -9.5, 0.0]
        )
        assert scaleoffset_f32.dtype == np.dtype("<f4")
        assert scaleoffset_f32.fillvalue == np.float32(-9.5)
        (filter_id, flags, parameters, name), = filter_records(scaleoffset_f32)
        assert (filter_id, flags, name) == (6, 1, b"scaleoffset")
        assert parameters[:8] == (0, 2, 3, 1, 4, 0, 0, 1)
        assert parameters[8] == 0xC1180000
        scaleoffset_f64 = handle["types/scaleoffset_f64"]
        np.testing.assert_array_equal(
            scaleoffset_f64[:], [1.5, -2.5, 3.5, -9.5, 0.0]
        )
        assert scaleoffset_f64.dtype == np.dtype("<f8")
        assert scaleoffset_f64.fillvalue == -9.5
        (filter_id, flags, parameters, name), = filter_records(scaleoffset_f64)
        assert (filter_id, flags, name) == (6, 1, b"scaleoffset")
        assert parameters[:8] == (0, 1, 5, 1, 8, 0, 0, 1)
        assert parameters[8:10] == (0, 0xC0230000)
        scaleoffset_f32_pipeline = handle["types/scaleoffset_f32_pipeline"]
        np.testing.assert_array_equal(
            scaleoffset_f32_pipeline[:], [1.25, 2.5, -1.75, 9.5, -9.5, 0.0]
        )
        records = filter_records(scaleoffset_f32_pipeline)
        assert [(record[0], record[1], record[3]) for record in records] == [
            (6, 1, b"scaleoffset"),
            (1, 0, b"deflate"),
        ]
        assert records[0][2][:8] == (0, 2, 3, 1, 4, 0, 0, 1)
        growing_f32 = handle["types/growing_f32"]
        np.testing.assert_array_equal(growing_f32[:], [1.25, 2.5, 3.75, -1.0])
        assert growing_f32.dtype == np.dtype("<f4")
        assert growing_f32.maxshape == (None,)
        assert growing_f32.shuffle
        assert growing_f32.compression == "gzip"
        assert growing_f32.fletcher32
        dense_values = handle["metadata/values"]
        assert len(dense_values.attrs) == 2500
        for index in (0, 1250, 2499):
            assert dense_values.attrs[f"attr_{index}"] == np.uint64(index)
        dense = handle["metadata/dense"]
        assert len(dense) == 12
        assert len(dense.attrs) == 13
        for index in (0, 5, 11):
            assert dense[f"value_{index}"][()] == np.uint64(index)
            assert dense.attrs[f"dense_attr_{index}"] == np.uint64(index + 100)
        assert dense.attrs["dense_text"] == "密集属性"
        continued = handle["metadata/continued"]
        assert len(continued.attrs) == 8
        assert (
            continued.attrs["continued_7"].decode("utf-8").rstrip("\0")
            == "c" * 9000
        )
        continued_link = handle.get(
            "metadata/continued_links/link_7", getlink=True
        )
        assert isinstance(continued_link, h5py.SoftLink)
        assert continued_link.path == "/" + "t" * 9000
        multi = handle["metadata/continued_multi"]
        assert len(multi.attrs) == 20
        for index in (0, 10, 19):
            assert multi.attrs[f"multi_{index}"].decode("utf-8").rstrip("\0") == (
                "m" * 60000
            )
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
        paged_series = handle["experiment/paged_series"]
        assert paged_series.maxshape == (None,)
        assert paged_series.chunks == (1,)
        assert paged_series.compression == "gzip"
        assert paged_series.compression_opts == 6
        assert paged_series.fletcher32
        expected_paged = np.zeros(134200, dtype="u1")
        expected_paged[[0, 243, 244, 131059, 131060, 132100, 134199]] = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ]
        np.testing.assert_array_equal(paged_series[:], expected_paged)
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
