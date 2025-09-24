from TexGen.Core import *

NumBinderLayers = 2
NumXYarns = 3
NumYYarns = 4
XSpacing = 1.0
YSpacing = 1.0
XHeight = 0.2
YHeight = 0.2
weave = CTextileLayerToLayer(
    NumXYarns, NumYYarns, XSpacing, YSpacing, XHeight, YHeight, NumBinderLayers
)

# set number of binder / warp yarns
NumBinderYarns = 3
NumWarpYarns = NumXYarns - NumBinderYarns
weave.SetWarpRatio(NumWarpYarns)
weave.SetBinderRatio(NumBinderYarns)

# setup layers: 3 warp, 4 weft
weave.SetupLayers(3, 4, NumBinderLayers)

# set yarn dimensions: widths / heights
weave.SetYYarnWidths(0.8)
weave.SetYYarnWidths(0.8)
weave.SetBinderYarnWidths(0.4)
weave.SetBinderYarnHeights(0.1)

# define offsets for the binder yarns (should match NumBinderYarns)
P = [[0, 1, 3, 0], [3, 0, 0, 3], [1, 2, 2, 1]]

# assign the z-positions to the binder yarns
for y in range(NumWarpYarns, NumXYarns):  # loop through number of binder yarns
    offset = 0
    for x in range(NumYYarns):  # loop through the node positions
        weave.SetBinderPosition(x, y, P[y - NumWarpYarns][offset])
        offset += 1

weave.AssignDefaultDomain()


###################################################
# Export to voxel mesh and convert to h5 and xdmf
###################################################

from MSUtils.general.vtk2h5 import vtk2h5
from MSUtils.general.h52xdmf import write_xdmf
import os

# choose resolution
nx, ny, nz = 128, 128, 128
vm = CRectangularVoxelMesh()

vm.SaveVoxelMesh(
    weave,
    "data/custom_weave1.vtu",
    nx,
    ny,
    nz,
    False,
    True,
    NO_BOUNDARY_CONDITIONS,
    0,
    VTU_EXPORT,
)

vtk2h5(
    vtk_files=["data/custom_weave1.vtu"],
    h5_file_path="data/TexGen_custom_weave1.h5",
    grp_name="/",
    overwrite=True,
    data_fields=["YarnIndex", "Orientation"],
)
os.remove("data/custom_weave1.vtu")
write_xdmf(
    h5_filepath="data/TexGen_custom_weave1.h5",
    xdmf_filepath="data/TexGen_custom_weave1.xdmf",
    microstructure_length=[4, 3, 1.43],
    time_series=False,
    verbose=True,
)
