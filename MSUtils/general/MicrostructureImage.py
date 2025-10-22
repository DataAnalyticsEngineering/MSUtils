from typing import Dict, List, Optional, Union

import h5py
import numpy as np


class MicrostructureImage:
    """
    A class for handling microstructure images stored in HDF5 format.

    This class provides methods for reading and writing 3D microstructure images
    from/to HDF5 files, handling associated metadata, and computing volume fractions
    of different phases present in the image.
    """

    def __init__(
        self,
        h5_filename: Optional[str] = None,
        dset_name: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        resolution: Optional[Union[List[int], tuple]] = None,
        L: Optional[Union[List[float], tuple]] = None,
        metadata: Optional[Dict[str, Union[str, int, float]]] = None,
    ):
        """
        Initializes the MicrostructureImage object.

        Args:
            h5_filename (Optional[str]): Path to the HDF5 file.
            dset_name (Optional[str]): Name of the dataset in the HDF5 file.
            image (Optional[np.ndarray]): Numpy array representing the image.
            resolution (Optional[Union[List[int], tuple]]): Number of voxels along each axis.
            L (Optional[Union[List[float], tuple]]): Physical dimensions of the image along each axis.
            metadata (Optional[Dict[str, Union[str, int, float]]]): Metadata associated with the image.

        Note:
            Either 'h5_filename' and 'dset_name' must be provided to read an image from a file,
            or 'image' must be provided to initialize the object with an image array.

            If 'image' is provided, 'h5_filename' and 'dset_name' are optional and can be used
            when writing the image to a file.

            If 'resolution' is not provided, it is set to the shape of the image array.

            If 'L' is not provided, default values are used.
        """
        self.h5_filename = h5_filename
        self.dset_name = dset_name
        self.image = image
        self.metadata = metadata if metadata is not None else {}
        self.volume_fractions = None

        self.shape = None
        self.resolution = resolution
        self.L = L

        if self.h5_filename and self.dset_name:
            self.read()

        if self.image is not None:
            self.shape = self.image.shape
            if self.resolution is None:
                self.resolution = self.shape
            if self.L is None:
                self.L = [1.0, 1.0, 1.0]  # Default physical dimensions
            self.compute_volume_fractions()

    def read(
        self, h5_filename: Optional[str] = None, dset_name: Optional[str] = None
    ) -> None:
        """
        Reads the image data and metadata from the specified HDF5 file and dataset.

        Args:
            h5_filename (Optional[str]): Path to the HDF5 file.
            dset_name (Optional[str]): Name of the dataset in the HDF5 file.

        Raises:
            ValueError: If both h5_filename and dset_name are not specified.
            ValueError: If the specified dataset is not found in the HDF5 file.
        """
        if h5_filename:
            self.h5_filename = h5_filename
        if dset_name:
            self.dset_name = dset_name

        if not self.h5_filename or not self.dset_name:
            raise ValueError("Both h5_filename and dset_name must be specified.")

        with h5py.File(self.h5_filename, "r") as f:
            if self.dset_name in f:
                dset = f[self.dset_name]
                self.image = dset[...]
                # Read metadata
                self.metadata = {key: value for key, value in dset.attrs.items()}

                # check for permute_order attribute
                if "permute_order" in dset.attrs:
                    permute_order = dset.attrs["permute_order"]
                else:
                    permute_order = "zyx"  # Default order

                if permute_order == "zyx":
                    self.image = self.image.transpose(2, 1, 0)
                elif permute_order != "xyz":
                    raise ValueError(
                        f"Invalid permute order {permute_order} in dataset {self.dset_name}"
                    )
                self.shape = self.image.shape
                self.resolution = self.shape
                # Read L from attributes if available
                if "L" in dset.attrs and isinstance(dset.attrs["L"], (list, tuple)):
                    self.L = dset.attrs["L"]
                else:
                    self.L = [1.0, 1.0, 1.0]  # Default physical dimensions
                self.compute_volume_fractions()
            else:
                raise ValueError(
                    f"No dataset with name {self.dset_name} found in {self.h5_filename}"
                )

    def write(
        self,
        h5_filename: Optional[str] = None,
        dset_name: Optional[str] = None,
        order: str = "zyx",
        compression_level: int = 6,
    ) -> None:
        """
        Writes the image data and metadata to the specified HDF5 file and dataset.

        Args:
            h5_filename (Optional[str]): Path to the HDF5 file.
            dset_name (Optional[str]): Name of the dataset in the HDF5 file.
            order (str): Order of the dimensions ('xyz' or 'zyx').
            compression_level (int): GZIP compression level (0-9).

        Raises:
            ValueError: If both h5_filename and dset_name are not specified.
            ValueError: If there is no image to write to the HDF5 file.
            ValueError: If an invalid order is specified.
            ValueError: If an invalid compression level is specified.

        Note:
            If a dataset with the same name already exists in the HDF5 file, it will be overwritten.
        """
        if h5_filename:
            self.h5_filename = h5_filename
        if dset_name:
            self.dset_name = dset_name

        if not self.h5_filename or not self.dset_name:
            raise ValueError("Both h5_filename and dset_name must be specified.")

        if self.image is None:
            raise ValueError("No image to write to H5 file.")

        if not (0 <= compression_level <= 9):
            raise ValueError("Invalid compression level. Must be between 0 and 9.")

        # Permute the image data based on the specified order
        if order == "xyz":
            permuted_image = self.image
        elif order == "zyx":
            permuted_image = self.image.transpose(2, 1, 0)
            self.L = self.L[::-1]
        else:
            raise ValueError("Invalid order specified. Use 'xyz' or 'zyx'.")

        with h5py.File(self.h5_filename, "a") as f:
            if self.dset_name in f:
                del f[self.dset_name]
                print(
                    f"Dataset {self.dset_name} exists in {self.h5_filename}, overwriting it."
                )
            dset = f.create_dataset(
                self.dset_name,
                data=permuted_image,
                compression="gzip",
                compression_opts=compression_level,
            )
            dset.attrs["permute_order"] = order

            # Write metadata
            if self.metadata:
                for key, value in self.metadata.items():
                    dset.attrs[key] = value
            # Write L as attribute
            if self.L is not None:
                dset.attrs["L"] = self.L

            print(
                f"Image written to dataset {self.dset_name} in {self.h5_filename} "
                f"with GZIP level {compression_level} compression."
            )

    def compute_volume_fractions(self) -> None:
        """
        Computes the volume fractions of unique labels in the image.

        The volume fractions are stored in the 'volume_fractions' attribute
        as a dictionary mapping labels to their respective volume fractions.
        """
        unique_labels = np.unique(self.image)
        self.volume_fractions = {
            label: np.sum(self.image == label) / self.image.size
            for label in unique_labels
        }


def main():
    ms = MicrostructureImage(
        h5_filename="data/sphere.h5", dset_name="/sphere03628/240x240x240/ms"
    )

    for key in sorted(ms.volume_fractions.keys()):
        value = ms.volume_fractions[key]
        print(f"Volume fraction of phase {key}: {value * 100:.6f}%")

    ms.write(
        h5_filename="data/test_MicrostructureImage.h5",
        dset_name="ms",
        order="zyx",
        compression_level=6,
    )


if __name__ == "__main__":
    main()
