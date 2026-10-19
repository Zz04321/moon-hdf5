from pathlib import Path

import h5py


def generate(destination: Path) -> None:
    names = ["signed", "big_endian", "floating", "text", "unallocated", "default", "partial_default"]
    with h5py.File(destination, "w", libver=("v108", "v108")) as handle:
        for name in names:
            handle.create_dataset(name, data=[42], track_times=False)
    with h5py.File(destination, "r") as handle:
        assert sorted(handle) == sorted(names)
        for name in names:
            assert handle[name][:].tolist() == [42]
    print(f"Verified {destination} with h5py {h5py.__version__}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "testdata" / "header_continuations.h5")
