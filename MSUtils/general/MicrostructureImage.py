import h5py
import numpy as np


class MicrostructureImage:
    def __init__(
        self,
        h5_filename=None,
        dset_name=None,
        image=None,
        resolution=None,
        L=[1, 1, 1],
        metadata=None,
    ):
        self.h5_filename = h5_filename
        self.dset_name = dset_name
        self.image = image
        self.resolution = resolution
        self.L = L
        self.metadata = metadata
        self.volume_fractions = None

        if self.h5_filename and self.dset_name:
            self.read()

        if self.image is not None and self.resolution is None:
            self.resolution = self.image.shape
            self.compute_volume_fractions()

    def read(self, h5_filename=None, dset_name=None):
        if h5_filename:
            self.h5_filename = h5_filename
        if dset_name:
            self.dset_name = dset_name

        if not self.h5_filename or not self.dset_name:
            raise ValueError("Both h5_filename and dset_name must be specified.")

        with h5py.File(self.h5_filename, "r") as f:
            if self.dset_name in f:
                dset = f[self.dset_name]
                self.image = np.array(dset)
                self.resolution = self.image.shape
                self.metadata = {key: value for key, value in dset.attrs.items()}
                self.compute_volume_fractions()
            else:
                raise ValueError(
                    f"No dataset with name {self.dset_name} found in {self.h5_filename}"
                )

    def write(self, h5_filename=None, dset_name=None, order="xyz"):
        if h5_filename:
            self.h5_filename = h5_filename
        if dset_name:
            self.dset_name = dset_name

        if not self.h5_filename or not self.dset_name:
            raise ValueError("Both h5_filename and dset_name must be specified.")

        if self.image is None:
            raise ValueError("No image to write to H5 file.")

        # Permute the image data based on the specified order
        if order == "xyz":
            permuted_image = self.image
        elif order == "zyx":
            permuted_image = np.transpose(
                self.image, (2, 1, 0)
            )  # Permute dimensions from xyz to zyx
        else:
            raise ValueError("Invalid order specified. Use 'xyz' or 'zyx'.")

        with h5py.File(self.h5_filename, "a") as f:
            if self.dset_name in f:
                del f[self.dset_name]
                print(f"Dataset {self.dset_name} exists in {self.h5_filename}, overwriting it.")
            dset = f.create_dataset(
                self.dset_name, data=permuted_image, compression="gzip", compression_opts=9
            )

            if self.metadata:
                for key, value in self.metadata.items():
                    dset.attrs[key] = value

            print(
                f"Image written to dataset {self.dset_name} in {self.h5_filename} with GZIP level 9 compression."
            )

    def compute_volume_fractions(self):
        unique_labels = np.unique(self.image)
        volumes = {label: np.sum(self.image == label) / self.image.size for label in unique_labels}

        self.volume_fractions = volumes
