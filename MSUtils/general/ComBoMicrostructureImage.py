import h5py
import numpy as np
from numpy.fft import fftn, ifftn, irfftn, rfftn


class VoxelInfo:
    def __init__(
        self, coords, elem_xyz, materials, fractions, fine_scale_block, normal=None
    ):
        self.coords = coords
        self.elem_xyz = elem_xyz
        self.materials = materials
        self.fractions = fractions
        self.normal = normal
        self.fine_scale_block = fine_scale_block


class ComBoMicrostructureImage:
    def __init__(
        self, coarse_image=None, fine_image=None, voxel_info_list=None, iface=None
    ):
        self.coarse_image = coarse_image
        self.fine_image = fine_image
        self.voxel_info_list = voxel_info_list if voxel_info_list is not None else []
        self.iface = iface

        if coarse_image is not None and voxel_info_list is not None:
            self.volume_fractions = self.compute_volume_fractions()
        else:
            self.volume_fractions = None

    def compute_volume_fractions(self):
        """
        Compute the volume (in number of voxels) of each material in the 3D image.
        """
        data_array = self.coarse_image
        if self.voxel_info_list is not None:
            unique_labels = np.unique(data_array[data_array >= 0])
            volumes = {
                label: np.sum(data_array == label) / data_array.size
                for label in unique_labels
            }

            for voxel_info in self.voxel_info_list:
                for material, fraction in zip(
                    voxel_info.materials, voxel_info.fractions, strict=False
                ):
                    if material in volumes:
                        volumes[material] += fraction / data_array.size
        else:
            unique_labels = np.unique(data_array)
            volumes = {
                label: np.sum(data_array == label) / data_array.size
                for label in unique_labels
            }

        return volumes

    def get_image_laplacian(self, img, L=[1.0, 1.0, 1.0]):
        """
        Compute the periodic Laplacian of the input image.
        """
        s_img = np.array(img.shape, dtype=int)

        # Step 1: Apply periodic Laplace stencil to the input image
        img_stencil = np.zeros(img.shape)
        L = np.array(L, dtype=float)
        lz = L[0] / s_img[0]
        ly = L[1] / s_img[1]
        lx = L[2] / s_img[2]
        l_vx = np.array([lz, ly, lx])

        f = 3.0 / (lx * ly / lz + lz * ly / lx + lx * lz / ly)
        img_stencil[0, 0, 0] += 2.0 * f * (lx * ly / lz + lx * lz / ly + ly * lz / lx)
        img_stencil[(1, -1), 0, 0] += -f * lx * ly / lz
        img_stencil[0, (1, -1), 0] += -f * lx * lz / ly
        img_stencil[0, 0, (1, -1)] += -f * ly * lz / lx

        even = s_img[2] % 2 == 1
        if even:
            iface = np.abs(ifftn(fftn(img.astype(float)) * fftn(img_stencil)))
        else:
            iface = np.abs(irfftn(rfftn(img.astype(float)) * rfftn(img_stencil)))

        return iface, l_vx

    def supervoxel_normal(self, img, lap_img, l, vol_frac=None):
        """
        Get the normal vector from a supervoxel using the Laplacian on the image.
        """
        inline_norm = lambda x: np.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])

        assert (
            img.ndim == 3
        ), "error: expecting 3D ndarray (0--> phase 0; else-->phase 1)"
        N = np.array(img.shape)
        l_combi = l * np.array(img.shape)
        w = np.abs(lap_img)
        iface = np.where(w > 1e-5)
        w = w[iface[0], iface[1], iface[2]].flatten()
        w = w / w.sum()
        XX = l[:, None] * np.array(iface)
        Xbar = (XX * w).sum(axis=1)
        XX = XX - Xbar[:, None]
        Mmod = XX @ (XX * w).T
        eigval, evec = np.linalg.eigh(Mmod)
        EV = evec[:, np.argmin(eigval)]
        normal_xyz = EV[::-1] / np.sqrt(EV[0] * EV[0] + EV[1] * EV[1] + EV[2] * EV[2])

        p_sum = img.sum(axis=(1, 2))
        n_red = N[0] * N[1] * N[2]
        if vol_frac is None:
            c1 = p_sum.sum() / n_red
        else:
            c1 = vol_frac

        x = []
        for i in range(3):
            x.append(np.linspace(-0.5, 0.5, N[i] + 1)[:-1] + 0.5 / N[i])

        z1_c = (p_sum * x[0]).sum() / (n_red * c1) * l_combi[0]

        p_sum = img.sum(axis=0)
        y1_c = (p_sum.sum(axis=1) * x[1]).sum() / (n_red * c1) * l_combi[1]
        x1_c = (p_sum.sum(axis=0) * x[2]).sum() / (n_red * c1) * l_combi[2]

        if x1_c * normal_xyz[0] + y1_c * normal_xyz[1] + z1_c * normal_xyz[2] > 0.0:
            normal_xyz *= -1.0

        # normal_classic = np.array([x1_c, y1_c, z1_c])
        # normal_classic = -normal_classic / inline_norm(normal_classic)
        # return normal_xyz, normal_classic
        return normal_xyz

    def downscale(
        self,
        data_array,
        Nx,
        Ny,
        Nz,
        min_vol_fraction=0.0,
        L=[1.0, 1.0, 1.0],
        pad_window=[0, 0, 0],
    ):
        """
        Downscale the given 3D image data_array to the size Nx x Ny x Nz considering periodic boundaries.
        """
        coarse_data = np.zeros((Nx, Ny, Nz), dtype=int)
        nx, ny, nz = data_array.shape
        # Check divisibility
        if nx % Nx != 0 or ny % Ny != 0 or nz % Nz != 0:
            raise ValueError(
                "Non-integer downscale factors detected due to indivisible dimensions. Please ensure that the dimensions of the fine-scale image are divisible by the desired coarse-scale dimensions."
            )
        dx, dy, dz = nx // Nx, ny // Ny, nz // Nz

        iface, l_vx = self.get_image_laplacian(data_array, L)

        multi_material_voxels = []
        element_number = -1
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    # Using modulo arithmetic to handle the wraparound
                    block = data_array[
                        (i * dx) % nx : ((i + 1) * dx) % nx,
                        (j * dy) % ny : ((j + 1) * dy) % ny,
                        (k * dz) % nz : ((k + 1) * dz) % nz,
                    ]
                    unique, counts = np.unique(block, return_counts=True)

                    fractions = counts / counts.sum()
                    significant_materials = unique[fractions >= min_vol_fraction]
                    significant_fractions = fractions[fractions >= min_vol_fraction]
                    element_number += 1

                    normal = None
                    if len(significant_materials) == 2:
                        xb = (i * dx - pad_window[0]) % nx
                        xe = ((i + 1) * dx + pad_window[0]) % nx
                        yb = (j * dy - pad_window[1]) % ny
                        ye = ((j + 1) * dy + pad_window[1]) % ny
                        zb = (k * dz - pad_window[2]) % nz
                        ze = ((k + 1) * dz + pad_window[2]) % nz
                        block_pad = data_array[xb:xe, yb:ye, zb:ze]

                        if np.array_equal(np.unique(block), np.unique(block_pad)):
                            iface_block = iface[xb:xe, yb:ye, zb:ze]
                            supervoxel = np.where(
                                block_pad == significant_materials[0], 0, 1
                            )
                        else:
                            iface_block = iface[
                                (i * dx) % nx : ((i + 1) * dx) % nx,
                                (j * dy) % ny : ((j + 1) * dy) % ny,
                                (k * dz) % nz : ((k + 1) * dz) % nz,
                            ]
                            supervoxel = np.where(
                                block == significant_materials[0], 0, 1
                            )

                        normal = self.supervoxel_normal(
                            supervoxel,
                            lap_img=iface_block,
                            l=l_vx,
                            vol_frac=significant_fractions[1],
                        )

                    if len(significant_materials) == 1:
                        coarse_data[i, j, k] = significant_materials[0]
                    elif len(significant_materials) > 1:
                        coarse_data[i, j, k] = -len(significant_materials)
                        voxel_info = VoxelInfo(
                            coords=(i, j, k),
                            elem_xyz=element_number,
                            materials=significant_materials,
                            fractions=significant_fractions,
                            fine_scale_block=block,
                            normal=normal,
                        )
                        multi_material_voxels.append(voxel_info)

            self.coarse_image = coarse_data
            self.voxel_info_list = multi_material_voxels
            self.fine_image = data_array
            self.iface = iface
            self.volume_fractions = self.compute_volume_fractions()

    def write(self, filename, group_name):
        """
        Write the coarse data, fine data, and multi-material voxel information to an HDF5 file.
        """
        with h5py.File(filename, "a") as f:
            if group_name in f:
                del f[group_name]
                print(f"Group {group_name} exists, overwriting it.")

            grp = f.create_group(group_name)

            if self.coarse_image is not None:
                grp.create_dataset(
                    "coarse_image",
                    data=self.coarse_image,
                    dtype=int,
                    compression="gzip",
                    compression_opts=9,
                )

            if self.fine_image is not None:
                grp.create_dataset(
                    "fine_image",
                    data=self.fine_image,
                    dtype=np.uint8,
                    compression="gzip",
                    compression_opts=9,
                )

            # if self.iface is not None:
            #     grp.create_dataset("iface", data=self.iface, dtype=np.float16, compression="gzip", compression_opts=9)

            if self.voxel_info_list:
                if self.coarse_image is not None:
                    coarse_normal = np.zeros(
                        (
                            self.coarse_image.shape[0],
                            self.coarse_image.shape[1],
                            self.coarse_image.shape[2],
                            3,
                        )
                    )
                    for voxel_info in self.voxel_info_list:
                        i, j, k = voxel_info.coords
                        normal_val = (
                            voxel_info.normal
                            if voxel_info.normal is not None
                            else (0.0, 0.0, 0.0)
                        )
                        coarse_normal[i, j, k] = normal_val
                    grp.create_dataset(
                        "coarse_normal",
                        data=coarse_normal,
                        dtype="f4",
                        compression="gzip",
                        compression_opts=9,
                    )

                max_materials = max(
                    len(voxel_info.materials) for voxel_info in self.voxel_info_list
                )
                block_shape = self.voxel_info_list[0].fine_scale_block.shape

                dtype = np.dtype(
                    [
                        ("coords", "3i4"),
                        ("elem_xyz", "i4"),
                        ("num_materials", "i4"),
                        ("materials", f"{max_materials}i4"),
                        ("fractions", f"{max_materials}f4"),
                        ("normal", "3f4"),
                        (
                            "fine_scale_block",
                            f"{block_shape[0]},{block_shape[1]},{block_shape[2]}i4",
                        ),
                    ]
                )
                structured_array = np.zeros(len(self.voxel_info_list), dtype=dtype)

                for idx, voxel_info in enumerate(self.voxel_info_list):
                    materials = np.pad(
                        voxel_info.materials.astype(np.int32),
                        (0, max_materials - len(voxel_info.materials)),
                        constant_values=-1,
                    )
                    fractions = np.pad(
                        voxel_info.fractions,
                        (0, max_materials - len(voxel_info.fractions)),
                        constant_values=-1.0,
                    )
                    num_materials = len(voxel_info.materials)
                    normal_val = (
                        voxel_info.normal
                        if voxel_info.normal is not None
                        else (0.0, 0.0, 0.0)
                    )

                    structured_array[idx] = (
                        voxel_info.coords,
                        voxel_info.elem_xyz,
                        num_materials,
                        materials,
                        fractions,
                        normal_val,
                        voxel_info.fine_scale_block,
                    )

                grp.create_dataset(
                    "multi_material_voxels",
                    data=structured_array,
                    dtype=dtype,
                    compression="gzip",
                    compression_opts=9,
                )

    @staticmethod
    def read(filename, group_name):
        """
        Read a ComBoMicrostructureImage object from an HDF5 file.
        """
        with h5py.File(filename, "r") as f:
            if group_name not in f:
                raise ValueError(f"Group {group_name} not found in file {filename}")

            group = f[group_name]

            if "coarse_image" in group:
                coarse_image = group["coarse_image"][:]
            if "fine_image" in group:
                fine_image = group["fine_image"][:]
            if "iface" in group:
                iface = group["iface"][:]

            voxel_info_list = []
            if "multi_material_voxels" in group:
                voxel_data = group["multi_material_voxels"]
                for voxel in voxel_data:
                    coords = tuple(voxel["coords"])
                    elem_xyz = voxel["elem_xyz"]
                    materials = voxel["materials"][voxel["materials"] != -1]
                    fractions = voxel["fractions"][voxel["fractions"] != -1]
                    normal = tuple(voxel["normal"])
                    fine_scale_block = voxel["fine_scale_block"]
                    voxel_info = VoxelInfo(
                        coords, elem_xyz, materials, fractions, fine_scale_block, normal
                    )
                    voxel_info_list.append(voxel_info)

            return ComBoMicrostructureImage(
                coarse_image, fine_image, voxel_info_list, iface
            )


def main():
    from MSUtils.general.h52xdmf import write_xdmf
    from MSUtils.general.MicrostructureImage import MicrostructureImage
    from MSUtils.general.resize_image import resize_image

    ms = MicrostructureImage(h5_filename="data/fibers1.h5", dset_name="/img")
    ms = MicrostructureImage(
        image=resize_image(ms.image, target_resolution=[256, 256, 256])
    )

    combo_micro_img = ComBoMicrostructureImage()
    combo_micro_img.downscale(ms.image, 64, 64, 64, pad_window=[2, 2, 2])
    combo_micro_img.write("data/test_combo.h5", "/combo_group")

    write_xdmf(
        h5_filepath="data/test_combo.h5",
        xdmf_filepath="data/test_combo.xdmf",
        microstructure_length=[1, 1, 1],
        verbose=True,
    )


if __name__ == "__main__":
    main()
