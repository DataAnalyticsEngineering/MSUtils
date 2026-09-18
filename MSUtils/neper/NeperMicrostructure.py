import mmap
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from MSUtils.general.grid import GridSpec
from MSUtils.general.MicrostructureImage import MicrostructureImage

_FORMAT_VERSION = "2.2"
_ORIENTATION_SIZES = {
    "rodrigues": 3,
    "euler-bunge": 3,
    "euler-kocks": 3,
    "euler-roe": 3,
    "rotmat": 9,
    "axis-angle": 4,
    "quaternion": 4,
}
_DATA_DTYPES = {
    "ascii": np.dtype(np.int64),
    "binary8": np.dtype("u1"),
    "binary16": np.dtype("<u2"),
    "binary16_big": np.dtype(">u2"),
    "binary32": np.dtype("<u4"),
    "binary32_big": np.dtype(">u4"),
}
_CANONICAL_CONVENTION = "Q_crystal_to_sample: v_sample = Q @ v_crystal"


class _Reader:
    def __init__(self, data: bytes, filename: Path):
        self.data = data
        self.filename = filename
        self.position = 0

    def line(self) -> bytes | None:
        if self.position >= len(self.data):
            return None
        end = self.data.find(b"\n", self.position)
        if end < 0:
            end = len(self.data)
        line = self.data[self.position : end].strip()
        self.position = min(end + 1, len(self.data))
        return line.rstrip(b"\r")

    def nonempty_line(self) -> bytes | None:
        line = self.line()
        while line is not None and not line:
            line = self.line()
        return line

    def token(self) -> bytes:
        while self.position < len(self.data) and self.data[self.position] in b" \t\r\n":
            self.position += 1
        if self.position >= len(self.data):
            raise ValueError(f"Unexpected end of {self.filename}.")
        start = self.position
        while (
            self.position < len(self.data)
            and self.data[self.position] not in b" \t\r\n"
        ):
            self.position += 1
        return self.data[start : self.position]

    def raw(self, dtype: np.dtype, count: int) -> np.ndarray:
        end = self.position + count * dtype.itemsize
        if end > len(self.data):
            raise ValueError(f"Unexpected end of {self.filename}.")
        values = np.frombuffer(
            self.data, dtype=dtype, count=count, offset=self.position
        ).copy()
        self.position = end
        return values


def _text(value: bytes | None, filename: Path) -> str:
    if value is None:
        raise ValueError(f"Unexpected end of {filename}.")
    try:
        return value.decode()
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid text in {filename}.") from error


def _values(reader: _Reader, count: int, converter, name: str) -> tuple:
    values = _text(reader.nonempty_line(), reader.filename).split()
    if len(values) != count:
        raise ValueError(f"{name} must contain {count} values in {reader.filename}.")
    try:
        return tuple(converter(value) for value in values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"Invalid {name} in {reader.filename}.") from error


def _ascii(reader: _Reader, count: int, dtype: np.dtype, name: str) -> np.ndarray:
    converter = float if dtype.kind == "f" else int
    try:
        values = [
            converter(_text(reader.token(), reader.filename)) for _ in range(count)
        ]
        return np.asarray(values, dtype=dtype)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"Invalid {name} in {reader.filename}.") from error


def _orientation_description(value: str) -> tuple[str, str]:
    descriptor, separator, convention = value.partition(":")
    if descriptor not in _ORIENTATION_SIZES or ":" in convention:
        raise ValueError(f"Unsupported Neper orientation descriptor {value!r}.")
    convention = convention if separator else "passive"
    if convention not in {"active", "passive"}:
        raise ValueError(f"Invalid Neper orientation convention {convention!r}.")
    return descriptor, convention


def _orientation_matrices(
    values: np.ndarray, descriptor: str, convention: str
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != _ORIENTATION_SIZES[descriptor]:
        raise ValueError(f"Invalid {descriptor} orientation array shape.")
    if np.any(~np.isfinite(values)):
        raise ValueError("Neper orientations must contain only finite values.")

    if descriptor == "rotmat":
        matrices = values.reshape(-1, 3, 3)
        if not np.allclose(
            matrices @ matrices.transpose(0, 2, 1), np.eye(3), atol=1e-6
        ):
            raise ValueError("Neper rotation matrices must be orthogonal.")
        if not np.allclose(np.linalg.det(matrices), 1.0, atol=1e-6):
            raise ValueError("Neper rotation matrices must have determinant one.")
        matrices = matrices.transpose(0, 2, 1)
    elif descriptor == "quaternion":
        matrices = Rotation.from_quat(values[:, (1, 2, 3, 0)]).as_matrix()
    elif descriptor == "rodrigues":
        matrices = Rotation.from_quat(
            np.column_stack((values, np.ones(len(values))))
        ).as_matrix()
    elif descriptor == "axis-angle":
        axes = values[:, :3]
        norms = np.linalg.norm(axes, axis=1)
        if np.any((norms == 0) & ~np.isclose(values[:, 3], 0.0)):
            raise ValueError("A nonzero axis-angle rotation requires a nonzero axis.")
        axes = axes / np.where(norms == 0, 1.0, norms)[:, None]
        matrices = Rotation.from_rotvec(
            axes * np.deg2rad(values[:, 3, None])
        ).as_matrix()
    else:
        angles = values.copy()
        if descriptor == "euler-kocks":
            angles[:, 0] += 90.0
            angles[:, 2] = 90.0 - angles[:, 2]
        elif descriptor == "euler-roe":
            angles[:, 0] += 90.0
            angles[:, 2] -= 90.0
        matrices = Rotation.from_euler("ZXZ", angles, degrees=True).as_matrix()

    if convention == "active":
        matrices = matrices.transpose(0, 2, 1)
    return np.ascontiguousarray(matrices)


def _external_file(reader: _Reader) -> str | None:
    position = reader.position
    line = reader.nonempty_line()
    if line is None or not line.startswith(b"*file"):
        reader.position = position
        return None
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Missing external filename in {reader.filename}.")
    return _text(parts[1], reader.filename).strip("\"'")


def _payload(reader: _Reader, encoding: str, count: int) -> np.ndarray:
    dtype = _DATA_DTYPES.get(encoding)
    if dtype is None:
        raise ValueError(f"Unsupported Neper data format {encoding!r}.")

    filename = _external_file(reader)
    if filename is None:
        values = (
            _ascii(reader, count, dtype, "voxel data")
            if encoding == "ascii"
            else reader.raw(dtype, count)
        )
    else:
        path = Path(filename)
        if not path.is_absolute():
            path = reader.filename.parent / path
        try:
            values = np.fromfile(
                path, dtype=dtype, sep=" " if encoding == "ascii" else ""
            )
        except OSError as error:
            raise ValueError(f"Cannot read external Neper data file {path}.") from error
    if values.size != count:
        raise ValueError(f"Neper data contain {values.size} values; expected {count}.")
    if encoding != "ascii" and dtype.byteorder in {"<", ">"}:
        values = values.astype(dtype.newbyteorder("="), copy=False)
    return values


class NeperMicrostructure(MicrostructureImage):
    """A Neper 5 ``.tesr`` in MSUtils' in-memory XYZ order."""

    def __init__(self, tesr_filename: str | Path):
        self.tesr_filename = Path(tesr_filename)
        self.canoncial_convention_grains = _CANONICAL_CONVENTION
        try:
            with self.tesr_filename.open("rb") as file:
                with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    self._parse(_Reader(data, self.tesr_filename))
        except OSError as error:
            raise ValueError(f"Cannot read Neper file {self.tesr_filename}.") from error

        lengths = tuple(
            size * spacing for size, spacing in zip(self.resolution, self.voxel_size)
        )
        super().__init__(
            image=self.image,
            grid=GridSpec(shape=self.resolution, lengths=lengths),
        )

    def _parse(self, reader: _Reader) -> None:
        if (
            reader.nonempty_line() != b"***tesr"
            or reader.nonempty_line() != b"**format"
        ):
            raise ValueError(f"{self.tesr_filename} is not a Neper .tesr file.")
        self.format_version = _text(reader.nonempty_line(), reader.filename)
        if self.format_version != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Neper .tesr format {self.format_version!r}; only {_FORMAT_VERSION} is supported."
            )
        if reader.nonempty_line() != b"**general":
            raise ValueError("Missing **general section in Neper .tesr file.")

        self.dimension = _values(reader, 1, int, "dimension")[0]
        if self.dimension != 3:
            raise ValueError("Only three-dimensional Neper rasters are supported.")
        self.resolution = _values(reader, 3, int, "resolution")
        self.voxel_size = _values(reader, 3, float, "voxel size")
        if any(value <= 0 for value in self.resolution):
            raise ValueError("Neper raster dimensions must be positive.")
        if any(not np.isfinite(value) or value <= 0 for value in self.voxel_size):
            raise ValueError("Neper voxel dimensions must be positive and finite.")

        self.origin = (0.0, 0.0, 0.0)
        for name in (
            "grain_count",
            "grain_ids",
            "crystal_symmetry",
            "rotation_matrices",
            "image",
        ):
            setattr(self, name, None)

        marker = reader.nonempty_line()
        while (
            marker is not None
            and marker.startswith(b"*")
            and not marker.startswith(b"**")
        ):
            if marker == b"*origin":
                self.origin = _values(reader, 3, float, "origin")
                if any(not np.isfinite(value) for value in self.origin):
                    raise ValueError("Neper origin must contain only finite values.")
            marker = self._skip(reader, b"*")

        voxel_count = int(np.prod(self.resolution))
        while marker is not None and marker != b"***end":
            if marker == b"**cell":
                marker = self._read_cells(reader)
            elif marker == b"**data":
                encoding = _text(reader.nonempty_line(), reader.filename)
                values = _payload(reader, encoding, voxel_count)
                if np.any(values < 0):
                    raise ValueError("Neper voxel grain IDs cannot be negative.")
                self.image = np.ascontiguousarray(
                    values.reshape(self.resolution, order="F")
                )
                marker = reader.nonempty_line()
            elif marker == b"**oridata":
                raise NotImplementedError(
                    "Voxel-wise Neper orientations are not supported yet."
                )
            elif marker == b"**oridef":
                raise NotImplementedError(
                    "Voxel-wise Neper orientations are not supported yet."
                )
            elif marker.startswith(b"**"):
                marker = self._skip(reader, b"**")
            else:
                raise ValueError(
                    f"Unexpected .tesr field {_text(marker, reader.filename)!r}."
                )

        if marker != b"***end" or self.image is None:
            raise ValueError(
                "Missing **data section or ***end marker in Neper .tesr file."
            )
        self._finish_grains()

    def _read_cells(self, reader: _Reader) -> bytes | None:
        marker = reader.nonempty_line()
        if marker is not None and not marker.startswith(b"*"):
            try:
                self.grain_count = int(_text(marker, reader.filename))
            except ValueError as error:
                raise ValueError(
                    f"Invalid Neper grain count {_text(marker, reader.filename)!r}."
                ) from error
            if self.grain_count < 0:
                raise ValueError("Neper grain count cannot be negative.")
            marker = reader.nonempty_line()

        while marker is not None and not marker.startswith(b"**"):
            if marker == b"*id":
                if self.grain_count is None:
                    raise ValueError("Neper *id requires a grain count.")
                self.grain_ids = _ascii(
                    reader, self.grain_count, np.dtype(np.int64), "grain IDs"
                )
            elif marker == b"*ori":
                if self.grain_count is None:
                    raise ValueError("Neper *ori requires a grain count.")
                description = _text(reader.nonempty_line(), reader.filename)
                descriptor, convention = _orientation_description(description)
                values = _ascii(
                    reader,
                    self.grain_count * _ORIENTATION_SIZES[descriptor],
                    np.dtype(np.float64),
                    "grain orientations",
                ).reshape(self.grain_count, _ORIENTATION_SIZES[descriptor])
                self.rotation_matrices = _orientation_matrices(
                    values, descriptor, convention
                )
            elif marker == b"*crysym":
                self.crystal_symmetry = _text(reader.nonempty_line(), reader.filename)
            elif marker.startswith(b"*"):
                marker = self._skip(reader, b"*")
                continue
            marker = reader.nonempty_line()
        return marker

    @staticmethod
    def _skip(reader: _Reader, prefix: bytes) -> bytes | None:
        marker = reader.nonempty_line()
        while marker is not None and not marker.startswith(prefix):
            marker = reader.nonempty_line()
        return marker

    def _finish_grains(self) -> None:
        if self.grain_count is None:
            values, inverse = np.unique(self.image, return_inverse=True)
            self.void_present = bool(values[0] == 0)
            self.grain_ids = np.ascontiguousarray(
                values[int(self.void_present) :], dtype=np.int64
            )
            self.grain_count = len(self.grain_ids)
            self.image = inverse.reshape(self.image.shape)
            return

        if self.grain_ids is None:
            self.grain_ids = np.arange(1, self.grain_count + 1, dtype=np.int64)
        else:
            self.grain_ids = np.asarray(self.grain_ids, dtype=np.int64)

        if (
            len(self.grain_ids) != self.grain_count
            or len(np.unique(self.grain_ids)) != self.grain_count
        ):
            raise ValueError(
                "Neper grain IDs must be unique and match the grain count."
            )
        if np.any(self.grain_ids <= 0):
            raise ValueError(
                "Neper grain IDs must be positive; zero is reserved for voids."
            )
        if np.any(self.image > self.grain_count):
            invalid = np.unique(self.image[self.image > self.grain_count])
            raise ValueError(
                f"Voxel data contain invalid internal grain IDs {invalid.tolist()}."
            )

        self.void_present = bool(np.any(self.image == 0))
        first_grain_id = int(self.void_present)
        self.image = np.ascontiguousarray(self.image, dtype=np.int64)
        self.image += first_grain_id - 1
        self.grain_ids = np.ascontiguousarray(self.grain_ids)

    def write(
        self,
        h5_filename: str | Path | None = None,
        dset_name: str | None = None,
        order: str = "zyx",
        compression_level: int = 6,
    ) -> None:
        """Write the microstructure and its rotation matrices."""
        dataset_name = "/" + (dset_name or self.dset_name or "").strip("/")
        rotation_name = dataset_name.rsplit("/", 1)[0] + "/rotation_matrices"
        if dataset_name == rotation_name:
            raise ValueError("The image dataset cannot be named 'rotation_matrices'.")

        super().write(h5_filename, dset_name, order, compression_level)

        with h5py.File(self.h5_filename, "a") as h5_file:
            if rotation_name in h5_file:
                del h5_file[rotation_name]
            if self.rotation_matrices is not None:
                rotation_matrices = self.rotation_matrices
                if self.void_present:
                    rotation_matrices = np.concatenate(
                        (np.eye(3)[None], rotation_matrices)
                    )
                dataset = h5_file.create_dataset(rotation_name, data=rotation_matrices)
                dataset.attrs["canonical_rotation_convention"] = _CANONICAL_CONVENTION
