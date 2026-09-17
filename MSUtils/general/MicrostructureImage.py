import h5py
import numpy as np

from MSUtils.general.grid import (
    GridSpec,
    image_in_order,
    validate_order,
)


class MicrostructureImage:
    """A three-dimensional microstructure image with HDF5 persistence."""

    def __init__(
        self,
        h5_filename: str | None = None,
        dset_name: str | None = None,
        image: np.ndarray | None = None,
        L: list[float] | tuple | None = None,
        metadata: dict[str, object] | None = None,
        grid: GridSpec | None = None,
    ):
        self.h5_filename = h5_filename
        self.dset_name = dset_name
        self.image = None
        self.grid = None
        self.metadata = dict(metadata or {})
        self.volume_fractions = None

        if h5_filename and dset_name:
            self.read()
        elif image is not None:
            image = np.ascontiguousarray(image)
            if grid is None:
                grid = GridSpec(
                    shape=image.shape,
                    lengths=(1.0, 1.0, 1.0) if L is None else tuple(L),
                )
            if not isinstance(grid, GridSpec):
                raise TypeError("grid must be a GridSpec instance.")
            if image.shape != grid.shape:
                raise ValueError("grid.shape must match image.shape.")
            if L is not None and not np.array_equal(L, grid.lengths):
                raise ValueError("L and grid.lengths must match.")
            self.image = image
            self.grid = grid
            self.compute_volume_fractions()

    @property
    def shape(self) -> tuple[int, int, int] | None:
        return None if self.image is None else self.image.shape

    @property
    def L(self) -> tuple[float, float, float] | None:
        return None if self.grid is None else self.grid.lengths

    def read(
        self, h5_filename: str | None = None, dset_name: str | None = None
    ) -> None:
        """Read an HDF5 dataset into canonical in-memory XYZ order."""
        self.h5_filename = h5_filename or self.h5_filename
        self.dset_name = dset_name or self.dset_name
        if not self.h5_filename or not self.dset_name:
            raise ValueError("Both h5_filename and dset_name must be specified.")

        with h5py.File(self.h5_filename, "r") as h5_file:
            if self.dset_name not in h5_file:
                raise ValueError(
                    f"No dataset with name {self.dset_name} found in {self.h5_filename}"
                )
            dataset = h5_file[self.dset_name]
            if dataset.ndim != 3:
                raise ValueError("Microstructure images must be three-dimensional.")

            order = validate_order(dataset.attrs.get("permute_order", "zyx"))
            self.image = np.ascontiguousarray(image_in_order(dataset[...], order))
            self.grid = GridSpec.from_h5_attributes(
                self.image.shape, dataset.attrs, order
            )
            self.metadata = dict(dataset.attrs)

        self.compute_volume_fractions()

    def write(
        self,
        h5_filename: str | None = None,
        dset_name: str | None = None,
        order: str = "zyx",
        compression_level: int = 6,
    ) -> None:
        """Write the image and grid, optionally storing axes in ZYX order."""
        self.h5_filename = h5_filename or self.h5_filename
        self.dset_name = dset_name or self.dset_name
        if not self.h5_filename or not self.dset_name:
            raise ValueError("Both h5_filename and dset_name must be specified.")
        if self.image is None or self.grid is None:
            raise ValueError("No image to write to H5 file.")
        if self.image.shape != self.grid.shape:
            raise ValueError("grid.shape must match image.shape.")
        if not 0 <= compression_level <= 9:
            raise ValueError("Invalid compression level. Must be between 0 and 9.")
        order = validate_order(order)

        with h5py.File(self.h5_filename, "a") as h5_file:
            if self.dset_name in h5_file:
                del h5_file[self.dset_name]
            dataset = h5_file.create_dataset(
                self.dset_name,
                data=np.ascontiguousarray(image_in_order(self.image, order)),
                compression="gzip",
                compression_opts=compression_level,
            )
            dataset.attrs.update(self.metadata)
            dataset.attrs.update(self.grid.to_h5_attributes(order))

    def compute_volume_fractions(self) -> None:
        """Compute the fraction occupied by each unique image label."""
        labels, counts = np.unique(self.image, return_counts=True)
        self.volume_fractions = {
            label: count / self.image.size for label, count in zip(labels, counts)
        }
