import numpy as np
from typing import Iterable, Optional

from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.general.h52xdmf import write_xdmf


def _to_angle(x, L):
    return (x / L) * 2.0 * np.pi


class TPMS:
    """
    TPMS generator.

    Parameters
    ----------
    tpms_type : str
        Type of TPMS surface (e.g., 'gyroid', 'schwarz_p', 'diamond', 'neovius', 'iwp', 'lidinoid').
    resolution : tuple of int
        Number of voxels in each direction (Nx, Ny, Nz).
    L : tuple of float
        Physical size in each direction (Lx, Ly, Lz).
    threshold : float
        Level-set value at which to threshold the implicit function.
    unitcell_frequency : tuple of int
        Number of unit cell repeats in each direction.
    invert : bool
        If True, swap the solid/void assignment (invert phases).
    mode : str
        'solid' (default) for classic TPMS, 'shell' for a shell of finite thickness.
    shell_thickness : float
        If mode='shell', the thickness of the shell (in field units, not physical units).
    """

    def __init__(
        self,
        tpms_type,
        resolution: Optional[Iterable[int]] = (128, 128, 128),
        L: Optional[Iterable[float]] = (1.0, 1.0, 1.0),
        threshold: Optional[float] = 0.5,
        unitcell_frequency: Optional[Iterable[int]] = (1, 1, 1),
        invert: bool = False,
        mode: str = "solid",
        shell_thickness: float = 0.1,
    ):
        self.kind = tpms_type.lower()
        self.resolution = tuple(int(v) for v in resolution)
        self.L = tuple(float(v) for v in L)
        if isinstance(unitcell_frequency, int):
            unitcell_frequency = (
                unitcell_frequency,
                unitcell_frequency,
                unitcell_frequency,
            )
        self.frequency = tuple(int(v) for v in unitcell_frequency)
        self.threshold = threshold
        self.invert = invert
        self.mode = mode
        self.shell_thickness = shell_thickness

        self._field = None  # cache for field
        self.image = self.generate()

    def implicit_function(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        # Map by frequency: scale coordinates before mapping to angle
        kx, ky, kz = self.frequency
        X = _to_angle(x * kx, self.L[0])
        Y = _to_angle(y * ky, self.L[1])
        Z = _to_angle(z * kz, self.L[2])

        kind = self.kind
        # Standard references: https://minimalsurfaces.blog/home/repository/triply-periodic/
        #                      https://kenbrakke.com/evolver/examples/periodic/periodic.html
        if kind in ("gyroid",):
            # Gyroid: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)
            return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
        if kind in ("schwarz_p", "p"):
            # Schwarz Primitive: cos(x) + cos(y) + cos(z)
            return np.cos(X) + np.cos(Y) + np.cos(Z)
        if kind in ("schwarz_d", "diamond", "d"):
            # Diamond: sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z) + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z)
            return (
                np.sin(X) * np.sin(Y) * np.sin(Z)
                + np.sin(X) * np.cos(Y) * np.cos(Z)
                + np.cos(X) * np.sin(Y) * np.cos(Z)
                + np.cos(X) * np.cos(Y) * np.sin(Z)
            )
        if kind in ("neovius",):
            # Neovius: 3 * (cos(x) + cos(y) + cos(z)) + 4 * cos(x)*cos(y)*cos(z)
            return 3 * (np.cos(X) + np.cos(Y) + np.cos(Z)) + 4 * np.cos(X) * np.cos(
                Y
            ) * np.cos(Z)
        if kind in ("iwp"):
            # I-WP : 2 * (cos(x)cos(y) + cos(y)cos(z) + cos(z)cos(x)) - (cos(2x) + cos(2y) + cos(2z))
            return 2 * (
                np.cos(X) * np.cos(Y) + np.cos(Y) * np.cos(Z) + np.cos(Z) * np.cos(X)
            ) - (np.cos(2 * X) + np.cos(2 * Y) + np.cos(2 * Z))
        if kind in ("lidinoid",):
            # Lidinoid: 0.5 * (sin(2x)cos(y)sin(z) + sin(2y)cos(z)sin(x) + sin(2z)cos(x)sin(y)) - 0.5 * (cos(2x)cos(2y) + cos(2y)cos(2z) + cos(2z)cos(2x)) + 0.15
            return (
                0.5
                * (
                    np.sin(2 * X) * np.cos(Y) * np.sin(Z)
                    + np.sin(2 * Y) * np.cos(Z) * np.sin(X)
                    + np.sin(2 * Z) * np.cos(X) * np.sin(Y)
                )
                - 0.5
                * (
                    np.cos(2 * X) * np.cos(2 * Y)
                    + np.cos(2 * Y) * np.cos(2 * Z)
                    + np.cos(2 * Z) * np.cos(2 * X)
                )
                + 0.15
            )
        raise ValueError(f"Unknown or unsupported TPMS kind: {self.kind}")

    def _compute_field(self):
        # Compute and cache the field
        Nx, Ny, Nz = self.resolution
        Lx, Ly, Lz = self.L
        xs = np.linspace(0.0, Lx, Nx, endpoint=False)
        ys = np.linspace(0.0, Ly, Ny, endpoint=False)
        zs = np.linspace(0.0, Lz, Nz, endpoint=False)
        X = xs[:, None, None]
        Y = ys[None, :, None]
        Z = zs[None, None, :]
        self._field = self.implicit_function(X, Y, Z)
        # range normalize to [0, 1]
        self._field = (self._field - np.min(self._field)) / (
            np.max(self._field) - np.min(self._field)
        )
        return self._field

    def generate(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Generate the binary microstructure.
        Returns a 3D numpy array (1=solid, 0=void). If invert=True, phases are swapped.
        If mode='shell', produces a shell of given thickness (in field units).
        """
        if self._field is None:
            field = self._compute_field()
        else:
            field = self._field
        if threshold is None:
            threshold = self.threshold
        if self.mode == "solid":
            img = (field > threshold).astype(np.uint8)
        elif self.mode == "shell":
            t = abs(self.shell_thickness)
            img = (np.abs(field - threshold) < t).astype(np.uint8)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.invert:
            img = 1 - img
        return img

    def find_threshold_for_volume_fraction(
        self,
        target_vf: float,
        tol: float = 1e-3,
        max_iter: int = 30,
        n_thresh: int = 50,
        optimize: str = "both",
    ) -> tuple:
        """
        Find threshold (and shell thickness if mode='shell') for target volume fraction.
        Parameters:
            target_vf: target volume fraction (fraction of solid voxels)
            tol: tolerance for volume fraction
            max_iter: max iterations for bisection
            n_thresh: number of threshold samples (for shell mode, if optimizing threshold)
            optimize: 'threshold', 'shell_thickness', or 'both' (shell mode only)
                - 'threshold': optimize threshold, keep shell_thickness fixed
                - 'shell_thickness': optimize shell_thickness, keep threshold fixed
                - 'both': jointly optimize both (default)
        Returns:
            - solid mode: (threshold, None)
            - shell mode: (threshold, shell_thickness)
        """
        if self._field is None:
            field = self._compute_field()
        else:
            field = self._field
        flat = field.ravel()
        n_vox = flat.size
        if self.mode == "solid":
            # For solid: threshold at quantile
            k = int(np.round((1 - target_vf) * n_vox))
            sorted_field = np.partition(flat, k)
            thr = sorted_field[k]
            self.threshold = thr
            return thr, None
        elif self.mode == "shell":
            minf, maxf = float(np.min(flat)), float(np.max(flat))
            if optimize == "shell_thickness":
                # Only optimize shell_thickness, keep threshold fixed
                thr = self.threshold
                lo, hi = 0.0, max(maxf - thr, thr - minf)
                for _ in range(max_iter):
                    mid = 0.5 * (lo + hi)
                    vf = np.mean(np.abs(flat - thr) < mid)
                    err = abs(vf - target_vf)
                    if err < tol:
                        break
                    if vf > target_vf:
                        hi = mid
                    else:
                        lo = mid
                self.shell_thickness = mid
                return thr, mid
            elif optimize == "threshold":
                # Only optimize threshold, keep shell_thickness fixed
                t = abs(self.shell_thickness)
                best_err = float("inf")
                best_thr = None
                for thr in np.linspace(minf, maxf, n_thresh):
                    vf = np.mean(np.abs(flat - thr) < t)
                    err = abs(vf - target_vf)
                    if err < best_err:
                        best_err = err
                        best_thr = thr
                        if best_err <= tol:
                            break
                self.threshold = best_thr
                return best_thr, t
            elif optimize == "both":
                # Jointly optimize threshold and shell_thickness
                best_err = float("inf")
                best_thr = None
                best_t = None
                for thr in np.linspace(minf, maxf, n_thresh):
                    lo, hi = 0.0, max(maxf - thr, thr - minf)
                    for _ in range(max_iter):
                        mid = 0.5 * (lo + hi)
                        vf = np.mean(np.abs(flat - thr) < mid)
                        err = abs(vf - target_vf)
                        if err < tol:
                            break
                        if vf > target_vf:
                            hi = mid
                        else:
                            lo = mid
                    vf = np.mean(np.abs(flat - thr) < mid)
                    err = abs(vf - target_vf)
                    if err < best_err:
                        best_err = err
                        best_thr = thr
                        best_t = mid
                        if best_err <= tol:
                            break
                self.threshold = best_thr
                self.shell_thickness = best_t
                return best_thr, best_t
            else:
                raise ValueError(f"Unknown optimize mode: {optimize}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def main():
    N = 512, 256, 128
    L = 4.0, 2.0, 1.0
    tpms_types = ["gyroid", "schwarz_p", "diamond", "neovius", "iwp", "lidinoid"]
    h5_filename = "data/tpms.h5"
    unitcell_frequency = (4, 2, 1)
    invert = True

    for tpms_type in tpms_types:
        tpms = TPMS(
            tpms_type=tpms_type,
            resolution=N,
            L=L,
            unitcell_frequency=unitcell_frequency,
            invert=invert,
            mode="solid",
        )
        MS = MicrostructureImage(image=tpms.image)
        MS.write(
            h5_filename=h5_filename,
            dset_name=tpms_type,
            order="zyx",
            compression_level=9,
        )

    write_xdmf(
        h5_filepath=h5_filename,
        xdmf_filepath="data/tpms.xdmf",
        microstructure_length=L[::-1],
        time_series=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
