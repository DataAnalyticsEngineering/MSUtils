import h5py
import numpy as np

from MSUtils.general.grid import image_in_order, validate_order


def write_combo(combo, filename, group_name):
    """Write a ComBo microstructure to an HDF5 group."""
    with h5py.File(filename, "a") as file:
        if group_name in file:
            del file[group_name]
            print(f"Group {group_name} exists, overwriting it.")

        group = file.create_group(group_name)
        group.attrs["lengths"] = combo.lengths
        group.attrs["permute_order"] = "zyx"

        group.create_dataset(
            "coarse_image",
            data=np.ascontiguousarray(image_in_order(combo.coarse_image, "zyx")),
            compression="gzip",
        )

        coarse_normal = np.zeros((*combo.coarse_image.shape, 3), dtype=np.float32)
        coarse_volume_fraction_0 = (combo.coarse_image == 0).astype(np.float32)
        for voxel_info in combo.voxel_info_list:
            coarse_normal[voxel_info.coords] = voxel_info.normal
            coarse_volume_fraction_0[voxel_info.coords] = voxel_info.fraction_0

        group.create_dataset(
            "coarse_normal",
            data=np.ascontiguousarray(coarse_normal.transpose(2, 1, 0, 3)),
            compression="gzip",
        )
        group.create_dataset(
            "coarse_volume_fraction_0",
            data=np.ascontiguousarray(image_in_order(coarse_volume_fraction_0, "zyx")),
            compression="gzip",
        )


def read_combo(filename, group_name):
    """Read a ComBo microstructure from an HDF5 group."""
    from MSUtils.ComBo.ComBoMicrostructureImage import (
        ComBoMicrostructureImage,
        VoxelInfo,
    )

    with h5py.File(filename, "r") as file:
        if group_name not in file:
            raise ValueError(f"Group {group_name} not found in file {filename}")

        group = file[group_name]
        order = validate_order(group.attrs.get("permute_order", "xyz"))
        coarse_image = np.ascontiguousarray(
            image_in_order(group["coarse_image"][:], order)
        )
        if not np.all(np.isin(coarse_image, (0, 1, 2))):
            raise ValueError("coarse_image must contain only labels 0, 1, and 2.")
        lengths = tuple(group.attrs.get("lengths", (1.0, 1.0, 1.0)))

        composite_coords = np.argwhere(coarse_image == 2)
        voxel_info_list = []
        if len(composite_coords):
            if not {"coarse_normal", "coarse_volume_fraction_0"} <= set(group):
                raise ValueError("Composite boxel fields are missing from the file.")
            coarse_normal = group["coarse_normal"][:]
            coarse_volume_fraction_0 = group["coarse_volume_fraction_0"][:]
            if order == "zyx":
                coarse_normal = coarse_normal.transpose(2, 1, 0, 3)
                coarse_volume_fraction_0 = coarse_volume_fraction_0.transpose(2, 1, 0)
            for coords_array in composite_coords:
                coords = tuple(coords_array)
                fraction_0 = coarse_volume_fraction_0[coords]
                voxel_info_list.append(
                    VoxelInfo(
                        coords=coords,
                        fraction_0=fraction_0,
                        normal=tuple(coarse_normal[coords]),
                    )
                )

    return ComBoMicrostructureImage(coarse_image, voxel_info_list, lengths)
