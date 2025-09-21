# =============================================================================
# TexGen: Geometric textile modeller.
# Copyright (C) 2015 Louise Brown

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# =============================================================================

# Python 3 version used runpy module to execute scripts from TexGen GUI which requires import of library
from TexGen.Core import *

# Create a 4x4 2d woven textile with yarn spacing of 5 and thickness 2
# The fifth parameter indicates whether to refine the textile to avoid intersections
Textile = CTextileWeave2D(4, 4, 5, 2, False)

# Set the weave pattern
Textile.SwapPosition(3, 0)
Textile.SwapPosition(2, 1)
Textile.SwapPosition(1, 2)
Textile.SwapPosition(0, 3)

# Adjust the yarn width and height
Textile.SetYarnWidths(4)
Textile.SetYarnHeights(0.8)

# Setup a domain
Textile.AssignDefaultDomain()

# Add the textile
AddTextile(Textile)

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
    Textile,
    "data/2dweave.vtu",
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
    vtk_files=["data/2dweave.vtu"],
    h5_file_path="data/TexGen_2dweave.h5",
    grp_name="/",
    overwrite=True,
)
os.remove("data/2dweave.vtu")
write_xdmf(
    h5_filepath="data/TexGen_2dweave.h5",
    xdmf_filepath="data/TexGen_2dweave.xdmf",
    microstructure_length=[20, 20, 2.2],
    time_series=False,
    verbose=True,
)
