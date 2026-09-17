import numpy as np

from MSUtils.ComBo.interface_normal import (
    c2c_normal,
    combo_normal,
    interface_laplacian,
)


def _validate_lengths(lengths):
    values = np.asarray(lengths, dtype=float)
    if values.shape != (3,) or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("lengths must contain three positive finite values.")
    return tuple(float(value) for value in values)


def _periodic_block(array, starts, widths):
    indices = (
        np.arange(start, start + width) % size
        for start, width, size in zip(starts, widths, array.shape, strict=True)
    )
    return array[np.ix_(*indices)]


class VoxelInfo:
    def __init__(self, coords, fraction_0, normal=None):
        self.coords = coords
        self.fraction_0 = fraction_0
        self.normal = normal


class ComBoMicrostructureImage:
    def __init__(
        self,
        coarse_image=None,
        voxel_info_list=None,
        lengths=(1.0, 1.0, 1.0),
    ):
        self.coarse_image = coarse_image
        self.voxel_info_list = voxel_info_list if voxel_info_list is not None else []
        self.lengths = _validate_lengths(lengths)

        self.volume_fractions = (
            None if coarse_image is None else self.compute_volume_fractions()
        )

    def compute_volume_fractions(self):
        """Compute the volume fraction of each phase."""
        fraction_0 = sum(voxel.fraction_0 for voxel in self.voxel_info_list)
        volumes = (
            np.count_nonzero(self.coarse_image == 0) + fraction_0,
            np.count_nonzero(self.coarse_image == 1)
            + len(self.voxel_info_list)
            - fraction_0,
        )
        return {
            phase: volume / self.coarse_image.size
            for phase, volume in enumerate(volumes)
            if volume
        }

    def downscale(
        self,
        data_array,
        Nx,
        Ny,
        Nz,
        min_vol_fraction=0.0,
        L=(1.0, 1.0, 1.0),
        pad_window=(0, 0, 0),
        normal_mode="combo",
    ):
        """Downscale a periodic binary image to an ``Nx x Ny x Nz`` ComBo image."""
        data_array = np.ascontiguousarray(data_array)
        if data_array.ndim != 3:
            raise ValueError("ComBo images must be three-dimensional.")
        if (
            np.count_nonzero(data_array == 0) + np.count_nonzero(data_array == 1)
            != data_array.size
        ):
            raise ValueError("ComBo images must use the two phase labels 0 and 1.")
        if not 0 <= min_vol_fraction <= 0.5:
            raise ValueError("min_vol_fraction must be between 0 and 0.5.")
        if any(
            not isinstance(size, (int, np.integer)) or size <= 0
            for size in (Nx, Ny, Nz)
        ):
            raise ValueError("The coarse resolution must contain positive integers.")
        lengths = _validate_lengths(L)
        nx, ny, nz = data_array.shape
        if nx % Nx != 0 or ny % Ny != 0 or nz % Nz != 0:
            raise ValueError(
                "The image shape must be divisible by the coarse resolution."
            )
        dx, dy, dz = nx // Nx, ny // Ny, nz // Nz
        pad_window = np.asarray(pad_window)
        if (
            pad_window.shape != (3,)
            or np.any(pad_window != np.floor(pad_window))
            or np.any(pad_window < 0)
        ):
            raise ValueError("pad_window must contain three nonnegative integers.")
        pad_window = pad_window.astype(int)
        has_padding = np.any(pad_window)

        if normal_mode not in {"c2c", "combo"}:
            raise ValueError("normal_mode must be 'c2c' or 'combo'.")

        fraction_1 = data_array.reshape(Nx, dx, Ny, dy, Nz, dz).mean(axis=(1, 3, 5))
        fraction_0 = 1.0 - fraction_1
        composite = (
            (fraction_0 > 0.0)
            & (fraction_1 > 0.0)
            & (fraction_0 >= min_vol_fraction)
            & (fraction_1 >= min_vol_fraction)
        )
        coarse_data = (fraction_1 > fraction_0).astype(np.uint8)
        coarse_data[composite] = 2

        voxel_sizes = np.asarray(lengths) / data_array.shape
        if normal_mode == "combo" and np.any(composite):
            interface, voxel_sizes = interface_laplacian(data_array, lengths)

        composite_voxels = []
        for i, j, k in np.argwhere(composite):
            block_slices = (
                slice(i * dx, (i + 1) * dx),
                slice(j * dy, (j + 1) * dy),
                slice(k * dz, (k + 1) * dz),
            )
            normal_block = data_array[block_slices]
            if normal_mode == "combo":
                interface_block = interface[block_slices]
                if has_padding:
                    starts = (
                        i * dx - pad_window[0],
                        j * dy - pad_window[1],
                        k * dz - pad_window[2],
                    )
                    widths = (
                        dx + 2 * pad_window[0],
                        dy + 2 * pad_window[1],
                        dz + 2 * pad_window[2],
                    )
                    normal_block = _periodic_block(data_array, starts, widths)
                    interface_block = _periodic_block(interface, starts, widths)

            normal = (
                c2c_normal(normal_block, voxel_sizes)
                if normal_mode == "c2c"
                else combo_normal(normal_block, interface_block, voxel_sizes)
            )
            composite_voxels.append(VoxelInfo((i, j, k), fraction_0[i, j, k], normal))

        self.coarse_image = coarse_data
        self.voxel_info_list = composite_voxels
        self.lengths = lengths
        self.volume_fractions = self.compute_volume_fractions()

    def write(self, filename, group_name):
        from MSUtils.ComBo.combo_io import write_combo

        if self.coarse_image is None:
            raise ValueError("The ComBo image has not been downscaled.")
        write_combo(self, filename, group_name)

    @staticmethod
    def read(filename, group_name):
        from MSUtils.ComBo.combo_io import read_combo

        return read_combo(filename, group_name)


def main():
    from MSUtils.ComBo.combo_mesh import write_combo_mesh
    from MSUtils.general.h52xdmf import write_xdmf
    from MSUtils.general.MicrostructureImage import MicrostructureImage

    image = MicrostructureImage(h5_filename="data/fibers1.h5", dset_name="/img").image

    result = ComBoMicrostructureImage()
    result.downscale(image, 80, 80, 30, normal_mode="combo")
    result.write("data/combo_normals.h5", "/combo_group")
    write_combo_mesh(result, "data/combo_mesh.xdmf")
    write_xdmf(
        h5_filepath="data/combo_normals.h5",
        xdmf_filepath="data/combo_normals.xdmf",
        verbose=True,
    )


if __name__ == "__main__":
    main()
