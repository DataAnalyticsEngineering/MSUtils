from TexGen.Core import *

# Create a plain weave textile
t = 0.1  # layer thickness
weave = CTextileWeave2D(2, 2, 0.8, t, True)
weave.SwapPosition(0, 0)
weave.SwapPosition(1, 1)
weave.SetGapSize(0.01)  # set a gap between yarns

# Create the layered textile
LTextile = CTextileLayered()

# Add a plain weave layer with no offsets
LTextile.AddLayer(weave, XYZ(0, 0, 0))

# Add a second plain weave layer, offset in the z-direction
# by the textile thickness and by 0.5 and 0.5 in the x and y directions
LTextile.AddLayer(weave, XYZ(0.5, 0.5, t))

# Get the default domain of the plain weave and its min and max coordinates
Domain = weave.GetDefaultDomain()
Min = XYZ()
Max = XYZ()
Domain.GetBoxLimits(Min, Max)

# Get the domain upper surface
Plane = PLANE()
index = Domain.GetPlane(XYZ(0, 0, -1), Plane)
# Offset the top surface of the domain by the depth of the plain weave domain
Plane.d -= Max.z - Min.z
Domain.SetPlane(index, Plane)

# Assign the extended domain to the layered textile
LTextile.AssignDomain(Domain)

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
    LTextile,
    "data/custom_weave2.vtu",
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
    vtk_files=["data/custom_weave2.vtu"],
    h5_file_path="data/TexGen_custom_weave2.h5",
    grp_name="/",
    overwrite=True,
    data_fields=["YarnIndex", "Orientation"],
)
os.remove("data/custom_weave2.vtu")
write_xdmf(
    h5_filepath="data/TexGen_custom_weave2.h5",
    xdmf_filepath="data/TexGen_custom_weave2.xdmf",
    microstructure_length=[1.6, 1.6, 0.22],
    time_series=False,
    verbose=True,
)
