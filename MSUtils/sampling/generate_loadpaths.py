import numpy as np
from pathlib import Path
from pyrecest.sampling.hyperspherical_sampler import LeopardiSampler


def generate_linear_load_paths(grid: np.ndarray,
                               num_steps: int = 10,
                               dev_max: float = 0.03,  
                               vol_max: float = 0.005
                               ) -> np.ndarray:
    """
    Parameters
    ----------
    grid      : (N, 6) array
        Unit (or arbitrary-magnitude) direction vectors in Mandel ordering.
    num_steps : int
        Number of scalar increments along each path (excludes the zero state).
    dev_max   : float
        Maximum allowed ||deviatoric part|| (Euclidean norm) in Mandel coords.
    vol_max   : float
        Maximum allowed volumetric strain |trace(ε)|.

    Returns
    -------
    paths : (N, num_steps, 6) array
        Strain paths satisfying the constraints.
    """

    I_mandel = np.array([1., 1., 1., 0., 0., 0.])          # identity tensor in Mandel form
    n_dir, dim = grid.shape
    assert dim == 6, "Expected Mandel ordering with 6 components"

    # scalar multipliers for successive load steps (linear ramp 1/num_steps … 1)
    scalars = np.linspace(1/num_steps, 1.0, num_steps)

    paths = np.empty((n_dir, num_steps, dim), dtype=float)

    for k, v in enumerate(grid):
        tr      = v[:3].sum()                              # trace(ε)
        v_dev   = v - (tr / 3.0) * I_mandel                # deviatoric part
        dev_nrm = np.linalg.norm(v_dev)                    # ||dev||
        
        # factors that would exactly hit the limits
        fac_dev = np.inf if np.isclose(dev_nrm, 0.0) else dev_max / dev_nrm
        fac_vol = np.inf if np.isclose(tr,      0.0) else vol_max / abs(tr)

        s       = min(fac_dev, fac_vol)                    # scale so BOTH constraints hold
        v_lim   = s * v                                    # final (step-10) vector

        # build the path ⟨step 1 … step num_steps⟩
        for j, a in enumerate(scalars):
            paths[k, j] = a * v_lim

    # check the constraints
    traces = paths[..., :3].sum(axis=-1)               
    devs   = paths - traces[..., None]/3.0 * I_mandel         
    assert np.all(np.abs(traces) <= max_volumetric_strain + 1e-12)
    assert np.all(np.linalg.norm(devs, axis=-1) <= max_deviatoric_strain + 1e-12)

    return paths

def dump_load_paths_to_json(
        paths: np.ndarray,
        filename: str = "macroscale_loading.json",
        include_zero_step: bool = False,
        decimals: int = 16
) -> Path:
    """
    Writes a JSON file with the strain paths in Mandel ordering.
    The JSON file is formatted as follows:
    {
      "macroscale_loading": [
        [ [e11, e22, e33, √2 e12, √2 e13, √2 e23], ... ],
        ...
      ]
    }

    where each strain path is a list of 6-component vectors in Mandel ordering.

    Parameters
    ----------
    paths : (N, S, 6) array
        The N×S×6 strain paths.
    filename : str
        Output file name.
    include_zero_step : bool
        Prepends a [0,0,0,0,0,0] state to every path if True.
    decimals : int
        Number of decimal places to round to.
    """
    # prepend zero step if requested
    if include_zero_step:
        zeros = np.zeros((paths.shape[0], 1, 6), dtype=paths.dtype)
        data_paths = np.concatenate([zeros, paths], axis=1)
    else:
        data_paths = paths

    # round once, to avoid cumulative formatting errors
    data_paths = np.round(data_paths, decimals)

    # ---------- build a pretty JSON dump ----------
    def vec2str(v):
        return "[" + ", ".join(f"{x:g}" for x in v) + "]"

    lines = []
    lines.append('{\n  "macroscale_loading": [')
    for p_idx, path in enumerate(data_paths):
        lines.append("    [")  # open one path
        for s_idx, vec in enumerate(path):
            comma = "," if s_idx < len(path) - 1 else ""
            lines.append(f"        {vec2str(vec)}{comma}")
        comma = "," if p_idx < len(data_paths) - 1 else ""
        lines.append(f"    ]{comma}")              # close path
    lines.append("  ]\n}")

    out_text = "\n".join(lines)

    # ---------- write to disk ----------
    out_path = Path(filename).expanduser().resolve()
    out_path.write_text(out_text, encoding="utf-8")
    print(f"✔ JSON written to: {out_path}")
    return out_path


if __name__ == "__main__":
    
    num_load_paths = 32 # number of load paths to generate
    num_time_steps = 10 # number of time steps per load path
    dim = 6 
    
    max_deviatoric_strain = 0.03 
    max_volumetric_strain = 0.005
    
    
    # Generate the Leopardi equal area grid 
    sampler = LeopardiSampler(original_code_column_order=True)
    grid, description = sampler.get_grid(num_load_paths, dim - 1)

    
    paths = generate_linear_load_paths(grid, 
                                       num_steps=num_time_steps,
                                       dev_max=max_deviatoric_strain,
                                       vol_max=max_volumetric_strain)
    
    dump_load_paths_to_json(paths,
                            filename="data/macroscale_loading.json",
                            include_zero_step=False)

