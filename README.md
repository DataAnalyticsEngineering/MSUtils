# MSUtils - Microstructure Utilities

<!-- Status & links -->
[![License](https://img.shields.io/badge/license-LGPL--3.0-blue)](LICENSE)[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)


MSUtils is a collection of utilities and scripts for creating, processing and exporting voxelized microstructure data which can be readily used in FFT-based solvers such as [FANS](https://github.com/DataAnalyticsEngineering/FANS).

## Installation

We use [Pixi](https://pixi.sh/latest/) for package management. If you don't have Pixi installed, install pixi via:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Clone the repository and install the required packages:

```bash
git clone https://github.com/DataAnalyticsEngineering/MSUtils.git
cd MSUtils/
pixi shell
```

## Microstructure generation

These modules produce periodic voxelized microstructures (numpy arrays / HDF5 datasets) of heterogeneous materials which can be readily used in FFT-based solvers such as [FANS](https://github.com/DataAnalyticsEngineering/FANS).

- Voronoi based microstructures: ([MSUtils/voronoi/](MSUtils/voronoi/))
  - [VoronoiSeeds.py](MSUtils/voronoi/VoronoiSeeds.py): Sampling strategies for Voronoi seed placement (sobol, halton, lhs-lloyd, lattice/honeycomb) and lattice vectors.
  - [VoronoiTessellation.py](MSUtils/voronoi/VoronoiTessellation.py): Build periodic Voronoi tessellations, compute analytical crystal volumes, interface areas, structure tensors and export to VTU.
  - [voronoi_foam.py](MSUtils/voronoi/voronoi_foam.py): Rasterize Voronoi edges into strut-based foam microstructures.
  - [VoronoiImage.py](MSUtils/voronoi/VoronoiImage.py): Rasterize seeds into a labelled voxel image using a periodic KDTree (nearest-seed labelling).
  - [VoronoiGBErosion.py](MSUtils/voronoi/VoronoiGBErosion.py): Erode Voronoi images to identify grain-boundary voxels, tag grain-boundary IDs and save grain-boundary metadata.

- Triply periodic minimal surface (TPMS) based microstructures: ([MSUtils/TPMS/](MSUtils/TPMS/))
  - [tpms.py](MSUtils/TPMS/tpms.py): Generate TPMS (Gyroid, Schwarz P, Diamond, Neovius, IWP, Lidinoid, etc.) based microstructures. Supports binarization modes (`solid`/`shell`) and threshold-finding for target volume fractions.
  - [tpms_functions.py](MSUtils/TPMS/tpms_functions.py): The raw implicit functions used by the TPMS generator.

- Spinodal microstructures: ([MSUtils/spinodoids/](MSUtils/spinodoids/))
  - [generate_spinodal_microstructure.py](MSUtils/spinodoids/generate_spinodal_microstructure.py): Spectral filtering method to synthesize spinodal-like microstructures with control over volume fraction, feature size and anisotropy.

- Lattice based microstructures: ([MSUtils/lattices/](MSUtils/lattices/))
  - [lattice_definitions.py](MSUtils/lattices/lattice_definitions.py): Unit-cell vertex + edge definitions for many lattices (BCC, FCC, octet, auxetic, etc.).
  - [lattice_image.py](MSUtils/lattices/lattice_image.py): Draw struts for lattice unit-cells onto a voxel grid.

## Microstructure utilities

Utilities for file I/O, conversions, and practical helpers.

- [MSUtils/general/](MSUtils/general/)
  - [MicrostructureImage.py](MSUtils/general/MicrostructureImage.py) - Core class for microstructure data: read/write HDF5 datasets with permute-order handling, metadata, and volume fraction calculation.
  - [ComBoMicrostructureImage.py](MSUtils/general/ComBoMicrostructureImage.py) - Morphologically sound coarse-graining via composite boxels (ComBo) as described in our [paper](https://doi.org/10.1007/s00466-022-02232-4).
  - [resize_image.py](MSUtils/general/resize_image.py) - Resize and smooth 3D labelled voxelized microstructure images to any target image resolution.
  - [vtk2h5.py](MSUtils/general/vtk2h5.py) - Convert VTI/VTU cell-centered meshes into HDF5 datasets inferred on a regular cell-center grid.
  - [h52xdmf.py](MSUtils/general/h52xdmf.py) - Convert HDF5 datasets into XDMF XML for ParaView. Supports scalar/vector/tensor attributes and time-series handling.
  - [merge_h5_files.py](MSUtils/general/merge_h5_files.py) - Merge multiple HDF5 files into one by recursively copying groups/datasets.

- [MSUtils/sampling/](MSUtils/sampling/)
  - [generate_loadpaths.py](MSUtils/sampling/generate_loadpaths.py) - Samples quasi-uniform equal-area directions on the unit hypersphere using the [`LeopardiSampler`](https://github.com/FlorianPfaff/pyRecEst/blob/main/pyrecest/sampling/leopardi_sampler.py) (See [paper](https://ftp.gwdg.de/pub/EMIS/journals/ETNA/vol.25.2006/pp309-327.dir/pp309-327.pdf) for further details). Using the sampled directions, produce linear strain ramps to user limits on deviatoric and volumetric strain magnitude and exporting the load paths to JSON that can be used in the input file for [FANS](https://github.com/DataAnalyticsEngineering/FANS).

## Acknowledgements

Funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC 2075 – 390740016. Contributions by Felix Fritzen are funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) within the Heisenberg program - DFG-FR2702/8 - 406068690; DFG-FR2702/10 - 517847245 and through NFDI-MatWerk - NFDI 38/1 - 460247524. We acknowledge the support of the Stuttgart Center for Simulation Science ([SimTech](https://www.simtech.uni-stuttgart.de/)).


## Contact

If you have questions or need support, please open an [issue](https://github.com/DataAnalyticsEngineering/MSUtils/issues).
You can contact [Sanath Keshav](https://github.com/sanathkeshav) (keshav.@mib.uni-stuttgart.de) with any other inquiries.

---
