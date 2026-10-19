from pathlib import Path
import h5py
import numpy as np


def generate(destination: Path, bounds: tuple[str, str]) -> None:
    expected = {}
    with h5py.File(destination, "w", libver=bounds) as handle:
        for name, options in (
            ("gzip", dict(compression="gzip")),
            ("shuffle", dict(shuffle=True)),
            ("fletcher", dict(fletcher32=True)),
            ("combined", dict(compression="gzip", shuffle=True, fletcher32=True)),
        ):
            values = np.arange(35, dtype=">i4").reshape(5, 7)
            handle.create_dataset(name, data=values, chunks=(3, 4), track_times=False, **options)
            expected[name] = values
        group = handle.create_group("special")
        values = np.arange(9, dtype="u1")
        group.create_dataset("odd", data=values, chunks=(3,), fletcher32=True, track_times=False)
        expected["special/odd"] = values
        dataset = group.create_dataset("partial", shape=(6,), chunks=(2,), dtype="<i4", fillvalue=-7, compression="gzip", shuffle=True, track_times=False)
        dataset[0] = 42
        expected["special/partial"] = np.array([42, -7, -7, -7, -7, -7])
        dataset = group.create_dataset("skipped", shape=(4,), chunks=(4,), dtype="<i4", compression="gzip", shuffle=True, fletcher32=True, track_times=False)
        values = np.array([9, 8, 7, 6], dtype="<i4")
        dataset.id.write_direct_chunk((0,), values.tobytes(), filter_mask=7)
        expected["special/skipped"] = values
        dataset = group.create_dataset("skip_deflate", shape=(4,), chunks=(4,), dtype="<i4", compression="gzip", shuffle=True, track_times=False)
        shuffled = values.view("u1").reshape(4, 4).T.copy().tobytes()
        dataset.id.write_direct_chunk((0,), shuffled, filter_mask=2)
        expected["special/skip_deflate"] = values
        floats = np.array([1.5, -2.25, 0.0, 123.5, 7.25], dtype=">f8")
        group.create_dataset("floats", data=floats, chunks=(3,), compression="gzip", shuffle=True, fletcher32=True, track_times=False)
        expected["special/floats"] = floats
        dataset = group.create_dataset("corrupt", data=np.arange(4, dtype="<i4"), chunks=(4,), fletcher32=True, track_times=False)
        mask, raw = dataset.id.read_direct_chunk((0,))
        damaged = bytes([raw[0] ^ 1]) + raw[1:]
        dataset.id.write_direct_chunk((0,), damaged, filter_mask=mask)
    with h5py.File(destination, "r") as handle:
        for name, values in expected.items():
            np.testing.assert_array_equal(handle[name][:], values)
        try:
            handle["special/corrupt"][:]
        except OSError:
            pass
        else:
            raise AssertionError("h5py accepted a corrupted checksum")
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "testdata"
    generate(directory / "filters_v1.h5", ("earliest", "v108"))
    generate(directory / "filters_v2.h5", ("v108", "v108"))
