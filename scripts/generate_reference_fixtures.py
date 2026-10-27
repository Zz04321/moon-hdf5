"""Generate the HDF5 1.14.6 object-reference fixture."""

from pathlib import Path

import h5py
import numpy as np


def generate(destination: Path, libver: str | tuple[str, str] = "latest") -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "references.h5 requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    with h5py.File(destination, "w", libver=libver) as handle:
        target = handle.create_dataset(
            "target", data=np.arange(6, dtype="<i4"), track_times=False
        )
        group = handle.create_group("group")
        child = group.create_dataset(
            "child", data=np.array([4, 5], dtype="<i4"), track_times=False
        )
        datatype = h5py.h5t.py_create(np.dtype("<i4"), logical=True)
        datatype.commit(handle.id, b"named_i4")

        dangling = handle.create_dataset(
            "dangling_target", data=np.array([9], dtype="<i4"), track_times=False
        )
        handle.create_dataset(
            "object_refs",
            data=np.array(
                [
                    target.ref,
                    group.ref,
                    child.ref,
                    handle["/"].ref,
                    handle["named_i4"].ref,
                    h5py.Reference(),
                ],
                dtype=h5py.ref_dtype,
            ),
            track_times=False,
        )
        handle.create_dataset(
            "dangling_refs",
            data=np.array([dangling.ref], dtype=h5py.ref_dtype),
            track_times=False,
        )
        target.attrs.create(
            "child_ref", np.array([child.ref], dtype=h5py.ref_dtype)
        )
        handle.create_dataset(
            "region_refs",
            data=np.array([target.regionref[1:3]], dtype=h5py.regionref_dtype),
            track_times=False,
        )
        del handle["dangling_target"]

    verify(destination)
    print(
        f"Generated {destination} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


def verify(destination: Path) -> None:
    expected = {
        0: ("/target", "Dataset"),
        1: ("/group", "Group"),
        2: ("/group/child", "Dataset"),
        3: ("/", "Group"),
        4: ("/named_i4", "Datatype"),
    }
    with h5py.File(destination, "r", libver="latest") as handle:
        references = handle["object_refs"][:]
        assert handle["object_refs"].id.get_type().get_size() == 8
        assert handle["region_refs"].id.get_type().get_size() == 12
        for index, (path, kind) in expected.items():
            reference = references[index]
            resolved = handle[reference]
            assert resolved.name == path
            assert type(resolved).__name__ == kind
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

        region = handle["region_refs"][0]
        np.testing.assert_array_equal(handle[region][region], [1, 2])

        target_address = h5py.h5o.get_info(handle["target"].id).addr
        group_address = h5py.h5o.get_info(handle["group"].id).addr
        root_address = h5py.h5o.get_info(handle["/"].id).addr
        datatype_address = h5py.h5o.get_info(handle["named_i4"].id).addr
        assert len(
            {
                target_address,
                group_address,
                root_address,
                datatype_address,
            }
        ) == 4


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        generate(Path(sys.argv[1]))
    elif len(sys.argv) == 1:
        fixture_directory = Path(__file__).resolve().parents[1] / "testdata"
        generate(fixture_directory / "references.h5")
        generate(fixture_directory / "references_v108.h5", ("earliest", "v108"))
    else:
        raise SystemExit(
            "usage: generate_reference_fixtures.py [file.h5]"
        )
