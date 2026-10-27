"""Verify the committed HDF5 1.14.6 object- and region-reference fixture."""

from pathlib import Path
import sys

import h5py
import numpy as np


def verify(path: Path) -> None:
    expected = {
        0: ("/target", "Dataset"),
        1: ("/group", "Group"),
        2: ("/group/child", "Dataset"),
        3: ("/", "Group"),
        4: ("/named_i4", "Datatype"),
    }
    with h5py.File(path, "r", libver="latest") as handle:
        references = handle["object_refs"][:]
        assert handle["object_refs"].id.get_type().get_size() == 8
        assert handle["region_refs"].id.get_type().get_size() == 12
        for index, (expected_path, expected_kind) in expected.items():
            resolved = handle[references[index]]
            assert resolved.name == expected_path
            assert type(resolved).__name__ == expected_kind
        assert not bool(references[5])

        child_reference = handle["target"].attrs["child_ref"]
        assert handle[child_reference[0]].name == "/group/child"

        dangling = handle["dangling_refs"][0]
        try:
            handle[dangling]
        except KeyError:
            pass
        else:
            raise AssertionError("dangling object reference unexpectedly resolved")

        region_expected = [
            ("/target", "Hyperslab", [1, 2]),
            ("/target", "Hyperslab", [0, 1, 3, 4]),
            ("/target", "Points", [4, 0, 3]),
            ("/target", "Hyperslab", [0, 1, 4]),
            ("/target", "All", [0, 1, 2, 3, 4, 5]),
            ("/target", "None", []),
            ("/scalar", "All", [7]),
            (
                "/matrix",
                "Hyperslab",
                [0, 3, 5, 8, 10, 13, 15, 18],
            ),
            ("/matrix", "Hyperslab", [0, 1, 2, 16, 17]),
        ]
        region_refs = handle["region_refs"][:]
        assert region_refs.shape == (11,)
        assert region_refs.dtype == h5py.regionref_dtype
        type_names = {
            h5py.h5s.SEL_NONE: "None",
            h5py.h5s.SEL_POINTS: "Points",
            h5py.h5s.SEL_HYPERSLABS: "Hyperslab",
            h5py.h5s.SEL_ALL: "All",
        }
        for index, (target_path, selection_type, values) in enumerate(
            region_expected
        ):
            reference = region_refs[index]
            resolved = handle[reference]
            assert resolved.name == target_path
            space = h5py.h5r.get_region(reference, handle.id)
            assert type_names[space.get_select_type()] == selection_type
            selected = np.asarray(resolved[reference]).reshape(-1)
            np.testing.assert_array_equal(selected, values)

        assert not bool(region_refs[9])
        try:
            handle[region_refs[10]]
        except KeyError:
            pass
        else:
            raise AssertionError("dangling region reference unexpectedly resolved")

    print(
        f"Verified {path} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_reference_fixture.py <file.h5>")
    verify(Path(sys.argv[1]))
