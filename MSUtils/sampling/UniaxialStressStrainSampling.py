import numpy as np

import MSUtils.sampling.generate_loadpaths as generate_loadpaths


def UniaxialStressStrainSampling(
    plane: str = "xy", n_phi: int = 500, exploit_symmetry: bool = True
):
    """Sample uniaxial directions and resulting tensors

    Parameters
    ----------
    plane : str, optional
        Plane on which to sample ('xy', 'yx', 'xz', 'zx', 'yz', 'zy'), by default 'xy'
    n_phi : int, optional
        Number of points used for sampling, by default 500
    exploit_symmetry : bool, optional
        Increases the effective resolution by a factor 2, but the vectors
        will only cover half a circle, by default True

    Returns
    -------
    nd.array
        directions in Mandel notation (ordering: xx, yy, zz, xy, xz, yz)
        in the n_phi rows
    nd.array
        directions of the vectors used to construct the tensor
        in the rows

    Raises
    ------
    ValueError
        If the parameter plane is inadmissible (i.e. none of 'xy', 'yx', 'xz', 'zx', 'yz', 'zy'), an error is raised.
    """
    directions, vectors = np.zeros((n_phi, 6)), np.zeros((n_phi, 3))
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    if exploit_symmetry:
        phi = np.linspace(0, np.pi, n_phi, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    if plane in ("xy", "yx"):
        idx_vec, idx_mandel = [0, 1], [0, 1, 3]
    elif plane in ("xz", "zx"):
        idx_vec, idx_mandel = [0, 2], [0, 2, 4]
    elif plane in ("yz", "zy"):
        idx_vec, idx_mandel = [1, 2], [1, 2, 5]
    else:
        raise ValueError(
            f'The "plane" parameter accepts "xy, yx, xz, zx, yz, zy", but received "{plane}".'
        )
    directions[:, idx_mandel] = np.vstack((c * c, s * s, np.sqrt(2) * c * s)).T
    vectors[:, idx_vec] = np.vstack((c, s)).T
    return directions, vectors


if __name__ == "__main__":
    num_load_paths = 128  # number of load paths to generate
    num_time_steps = 10  # number of time steps per load path
    dim = 6

    max_deviatoric_strain = 0.03
    max_volumetric_strain = 0.005

    grid, _ = UniaxialStressStrainSampling(
        plane="xy", n_phi=num_load_paths, exploit_symmetry=True
    )

    paths = generate_loadpaths.generate_linear_load_paths(
        grid,
        num_steps=num_time_steps,
        dev_max=max_deviatoric_strain,
        vol_max=max_volumetric_strain,
    )

    generate_loadpaths.dump_load_paths_to_json(
        paths, filename="data/macroscale_loading.json", include_zero_step=False
    )
