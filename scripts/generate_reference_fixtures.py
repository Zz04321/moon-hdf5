"""Generate the HDF5 1.14.6 object- and region-reference fixtures."""

from pathlib import Path

import h5py
import numpy as np


def create_layout_dataset(handle, name: str, data, layout) -> None:
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_obj_track_times(False)
    dcpl.set_layout(layout)
    if layout == h5py.h5d.CHUNKED:
        dcpl.set_chunk((2,))
        dcpl.set_shuffle()
        dcpl.set_deflate(6)
    space = h5py.h5s.create_simple(data.shape)
    datatype = h5py.h5t.py_create(data.dtype)
    dataset = h5py.h5d.create(
        handle.id, name.encode(), datatype, space, dcpl=dcpl
    )
    dataset.write(h5py.h5s.ALL, h5py.h5s.ALL, data)
    dataset.close()


def region_reference(dataset, configure) -> h5py.RegionReference:
    space = dataset.id.get_space()
    configure(space)
    return h5py.h5r.create(
        dataset.id, b".", h5py.h5r.DATASET_REGION, space
    )


def region_reference_values():
    return [
        ("slice", "/target", "Hyperslab", [1, 2]),
        ("regular", "/target", "Hyperslab", [0, 1, 3, 4]),
        ("points", "/target", "Points", [4, 0, 3]),
        ("blocks", "/target", "Hyperslab", [0, 1, 4]),
        ("all", "/target", "All", [0, 1, 2, 3, 4, 5]),
        ("none", "/target", "None", []),
        ("scalar", "/scalar", "All", [7]),
        (
            "matrix_regular",
            "/matrix",
            "Hyperslab",
            [0, 3, 5, 8, 10, 13, 15, 18],
        ),
        (
            "matrix_blocks",
            "/matrix",
            "Hyperslab",
            [0, 1, 2, 16, 17],
        ),
    ]


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
        matrix = handle.create_dataset(
            "matrix",
            data=np.arange(30, dtype="<i4").reshape(6, 5),
            track_times=False,
        )
        scalar = handle.create_dataset(
            "scalar", data=np.array(7, dtype="<i4"), track_times=False
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
        region_refs = [
            target.regionref[1:3],
            region_reference(
                target,
                lambda space: space.select_hyperslab(
                    (0,), (2,), stride=(3,), block=(2,)
                ),
            ),
            region_reference(
                target,
                lambda space: space.select_elements([(4,), (0,), (3,)]),
            ),
            region_reference(
                target,
                lambda space: (
                    space.select_hyperslab((0,), (1,), block=(2,)),
                    space.select_hyperslab(
                        (4,), (1,), op=h5py.h5s.SELECT_OR
                    ),
                ),
            ),
            region_reference(target, lambda space: space.select_all()),
            region_reference(target, lambda space: space.select_none()),
            region_reference(scalar, lambda space: space.select_all()),
            region_reference(
                matrix,
                lambda space: space.select_hyperslab(
                    (0, 0), (2, 2), stride=(2, 3), block=(2, 1)
                ),
            ),
            region_reference(
                matrix,
                lambda space: (
                    space.select_hyperslab((0, 0), (1, 1), block=(1, 3)),
                    space.select_hyperslab(
                        (3, 1), (1, 1), block=(1, 2), op=h5py.h5s.SELECT_OR
                    ),
                ),
            ),
            h5py.RegionReference(),
        ]
        dangling_region_target = handle.create_dataset(
            "dangling_region_target",
            data=np.array([11, 12], dtype="<i4"),
            track_times=False,
        )
        region_refs.append(dangling_region_target.regionref[0:2])
        handle.create_dataset(
            "region_refs",
            data=np.array(region_refs, dtype=h5py.regionref_dtype),
            track_times=False,
        )
        del handle["dangling_target"]
        del handle["dangling_region_target"]

    verify(destination)
    print(
        f"Generated {destination} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


def generate_layouts_32(destination: Path) -> None:
    if h5py.version.hdf5_version != "1.14.6":
        raise SystemExit(
            "references_layouts_32.h5 requires HDF5 1.14.6, found "
            + h5py.version.hdf5_version
        )

    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fcpl.set_sizes(4, 8)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
    fid = h5py.h5f.create(
        bytes(destination), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl
    )
    handle = h5py.File(fid)
    try:
        data = np.arange(6, dtype="<i4")
        create_layout_dataset(handle, "compact", data, h5py.h5d.COMPACT)
        create_layout_dataset(handle, "contiguous", data, h5py.h5d.CONTIGUOUS)
        create_layout_dataset(handle, "chunked", data, h5py.h5d.CHUNKED)
        handle.create_group("group")
        for name, reference in (
            ("object_ref_compact", handle["compact"].ref),
            ("object_ref_group", handle["group"].ref),
            ("object_ref_chunked", handle["chunked"].ref),
            ("object_ref_null", h5py.Reference()),
        ):
            handle.create_dataset(
                name,
                data=np.array([reference], dtype=h5py.ref_dtype),
                track_times=False,
            )
        handle["compact"].attrs.create(
            "sibling_ref",
            np.array([handle["contiguous"].ref], dtype=h5py.ref_dtype),
        )
        references = (
            (
                "ref_compact",
                "compact",
                lambda space: space.select_hyperslab(
                    (1,), (2,), stride=(2,), block=(1,)
                ),
            ),
            (
                "ref_contiguous",
                "contiguous",
                lambda space: space.select_elements([(4,), (0,), (2,)]),
            ),
            (
                "ref_chunked",
                "chunked",
                lambda space: space.select_hyperslab(
                    (0,), (2,), stride=(3,), block=(2,)
                ),
            ),
        )
        for reference_name, target_name, configure in references:
            reference = region_reference(handle[target_name], configure)
            handle.create_dataset(
                reference_name,
                data=np.array([reference], dtype=h5py.regionref_dtype),
                track_times=False,
            )
    finally:
        handle.close()

    verify_layouts_32(destination)
    print(
        f"Generated {destination} with h5py {h5py.__version__} "
        f"and HDF5 {h5py.version.hdf5_version}"
    )


def verify_layouts_32(destination: Path) -> None:
    layouts = {
        h5py.h5d.COMPACT: "compact",
        h5py.h5d.CONTIGUOUS: "contiguous",
        h5py.h5d.CHUNKED: "chunked",
    }
    from verify_region_layout_fixture import verify_serialized_references

    with h5py.File(destination, "r", libver="latest") as handle:
        assert handle.id.get_create_plist().get_sizes() == (4, 8)
        for name, layout in (
            ("compact", "compact"),
            ("contiguous", "contiguous"),
            ("chunked", "chunked"),
        ):
            actual = layouts[handle[name].id.get_create_plist().get_layout()]
            assert actual == layout
        assert handle["chunked"].shuffle
        assert handle["chunked"].compression == "gzip"
        assert handle["chunked"].chunks == (2,)

        for name in ("compact", "contiguous", "chunked"):
            np.testing.assert_array_equal(
                handle[name][...], np.arange(6, dtype="<i4")
            )
        expected = {
            "object_ref_compact": ("/compact", "Dataset"),
            "object_ref_group": ("/group", "Group"),
            "object_ref_chunked": ("/chunked", "Dataset"),
        }
        for name, (path, kind) in expected.items():
            dataset = handle[name]
            assert dataset.id.get_storage_size() == 4
            resolved = handle[dataset[0]]
            assert resolved.name == path
            assert type(resolved).__name__ == kind
        null_dataset = handle["object_ref_null"]
        null_offset = null_dataset.id.get_offset()
        assert destination.read_bytes()[null_offset : null_offset + 4] == b"\0" * 4
        attribute_ref = handle["compact"].attrs["sibling_ref"][0]
        assert handle[attribute_ref].name == "/contiguous"

    verify_serialized_references(destination)


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

        region_refs = handle["region_refs"][:]
        assert region_refs.dtype == h5py.regionref_dtype
        assert region_refs.shape == (11,)

        for index, (_, path, selection_type, values) in enumerate(
            region_reference_values()
        ):
            reference = region_refs[index]
            resolved = handle[reference]
            assert resolved.name == path
            space = h5py.h5r.get_region(reference, handle.id)
            type_names = {
                h5py.h5s.SEL_NONE: "None",
                h5py.h5s.SEL_POINTS: "Points",
                h5py.h5s.SEL_HYPERSLABS: "Hyperslab",
                h5py.h5s.SEL_ALL: "All",
            }
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
        generate_layouts_32(fixture_directory / "references_layouts_32.h5")
    else:
        raise SystemExit(
            "usage: generate_reference_fixtures.py [file.h5]"
        )
