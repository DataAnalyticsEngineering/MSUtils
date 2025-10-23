import os
import sys
import h5py
import numpy as np
import pyvista as pv
import argparse
from typing import Sequence, Optional


def _init_target(dtype: np.dtype, shape_3d, n_comp: int):
    """
    Create target array with sensible fill values: -1 for integer, 0 for float.
    Shape: (nx,ny,nz) or (nx,ny,nz,n_comp)
    """
    is_int = np.issubdtype(dtype, np.integer)
    fill = -1 if is_int else 0
    if n_comp == 1:
        tgt = np.full(
            shape_3d, fill, dtype=dtype if is_int else np.asarray(0, dtype).dtype
        )
    else:
        tgt = np.full(
            shape_3d + (n_comp,),
            fill,
            dtype=dtype if is_int else np.asarray(0.0, dtype).dtype,
        )
    return tgt


def _grid_from_centers(mesh, dec: int = 9):
    centers = mesh.cell_centers().points
    cx = np.round(centers[:, 0], dec)
    cy = np.round(centers[:, 1], dec)
    cz = np.round(centers[:, 2], dec)

    xu = np.unique(cx)
    yu = np.unique(cy)
    zu = np.unique(cz)
    nx, ny, nz = len(xu), len(yu), len(zu)
    if nx * ny * nz != mesh.n_cells:
        raise ValueError(
            f"Inferred grid {nx} x {ny} x {nz}={nx*ny*nz} != n_cells={mesh.n_cells}. "
            "Check dec (rounding) or input mesh."
        )

    # mean spacing from centers → robust to tiny roundoff
    dx = float(np.diff(xu).mean()) if nx > 1 else 0.0
    dy = float(np.diff(yu).mean()) if ny > 1 else 0.0
    dz = float(np.diff(zu).mean()) if nz > 1 else 0.0
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz

    ix = np.searchsorted(xu, cx)
    iy = np.searchsorted(yu, cy)
    iz = np.searchsorted(zu, cz)

    return (nx, ny, nz, ix, iy, iz, (Lx, Ly, Lz))


def vtk2h5(
    vtk_files: Sequence[str],
    h5_file_path: str,
    grp_name: str = "images",
    data_fields: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    dec: int = 9,
):
    mode = "a" if os.path.exists(h5_file_path) else "w"
    with h5py.File(h5_file_path, mode) as h5:
        root = h5[grp_name] if grp_name in h5 else h5.create_group(grp_name)

        for vf in vtk_files:
            stem = os.path.splitext(os.path.basename(vf))[0]
            grp = root[stem] if stem in root else root.create_group(stem)

            try:
                mesh = pv.read(vf)
            except Exception as e:
                print(f"[skip] {vf}: read error -> {e}")
                continue

            cd = mesh.cell_data
            if not cd:
                print(f"[skip] {vf}: no cell data present.")
                continue

            fields = (
                list(cd.keys())
                if data_fields is None
                else [f for f in data_fields if f in cd]
            )
            if not fields:
                print(f"[skip] {vf}: none of requested fields {data_fields} found.")
                continue

            try:
                nx, ny, nz, ix, iy, iz, domL = _grid_from_centers(mesh, dec=dec)
            except Exception as e:
                print(f"[warn] {vf}: grid inference failed ({e}); writing flat arrays.")
                for field in fields:
                    arr = np.asarray(cd[field])
                    if field in grp and not overwrite:
                        print(f"[skip] {vf}:{field} exists (overwrite=False).")
                        continue
                    if field in grp:
                        del grp[field]
                    ds = grp.create_dataset(
                        field,
                        data=arr,
                        compression="gzip",
                        compression_opts=9,
                        chunks=True,
                        shuffle=True,
                    )
                    # still record something minimal on fallback
                    ds.attrs["grid_shape"] = (mesh.n_cells,)
                grp.attrs["grid_type"] = mesh.__class__.__name__
                grp.attrs["n_cells"] = int(mesh.n_cells)
                continue

            for field in fields:
                arr = np.asarray(cd[field])
                n_comp = 1 if arr.ndim == 1 else arr.shape[1]
                tgt = _init_target(arr.dtype, (nx, ny, nz), n_comp)

                if n_comp == 1:
                    tgt[ix, iy, iz] = arr.astype(tgt.dtype, copy=False)
                else:
                    # handle vectors/tensors generically; for 3-vectors keep your VTK->xyz flip
                    if n_comp == 3:
                        tgt[ix, iy, iz, :] = arr.astype(tgt.dtype, copy=False)[
                            :, [2, 1, 0]
                        ]
                    else:
                        tgt[ix, iy, iz, :] = arr.astype(tgt.dtype, copy=False)

                if field in grp and not overwrite:
                    print(f"[skip] {vf}:{field} exists (overwrite=False).")
                    continue
                if field in grp:
                    del grp[field]

                chunks = tuple(min(64, s) for s in tgt.shape)
                dset = grp.create_dataset(
                    field,
                    data=tgt,
                    compression="gzip",
                    compression_opts=9,
                    chunks=chunks,
                    shuffle=True,
                )

                # --- Only the attributes you asked for ---
                dset.attrs["domain_lengths"] = domL  # (Lx, Ly, Lz)
                dset.attrs["grid_shape"] = (nx, ny, nz)

            # mirror the same minimal info at the group level
            grp.attrs["grid_shape"] = (nx, ny, nz)
            grp.attrs["domain_lengths"] = domL
            grp.attrs["grid_type"] = mesh.__class__.__name__
            grp.attrs["n_cells"] = int(mesh.n_cells)

        h5.attrs["num_files"] = len(root)

    print(f"Done → {h5_file_path}")
