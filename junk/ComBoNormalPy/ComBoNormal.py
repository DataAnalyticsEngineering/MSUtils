import numpy as np
import h5py
from numpy.fft import fftn,ifftn,rfftn,irfftn

class VoxelInfo:
    def __init__(self, coords, elem_xyz, materials, fractions, fine_scale_block, normal=None):
        self.coords = coords
        self.elem_xyz = elem_xyz
        self.materials = materials
        self.fractions = fractions
        self.normal = normal
        self.fine_scale_block = fine_scale_block

def compute_material_volumes_in_image(data_array, multi_material_voxels=None, verbosity=False):
    """
    Compute the volume (in number of voxels) of each material in the 3D image.

    Parameters:
    - data_array (numpy.ndarray): 3D numpy array containing labeled data.
    - multi_material_voxels (list): List of VoxelInfo objects representing voxels with multiple materials.
                                   Each VoxelInfo object should have a 'materials' attribute containing
                                   the list of materials present in the voxel, and a 'fractions' attribute
                                   containing the corresponding fractions of each material.
    - verbosity (bool): Flag indicating whether to print the volume fractions.

    Returns:
    - dict: Dictionary with material labels as keys and their respective volumes as values.
    """

    if multi_material_voxels is not None:
        unique_labels = np.unique(data_array[data_array >= 0])
        volumes = {label: np.sum(data_array == label)/data_array.size for label in unique_labels}
        
        for voxel_info in multi_material_voxels:
            for material, fraction in zip(voxel_info.materials, voxel_info.fractions):
                if material in volumes:
                    volumes[material] += fraction/data_array.size
    else:
        unique_labels = np.unique(data_array)
        volumes = {label: np.sum(data_array == label)/data_array.size for label in unique_labels}

    if verbosity:
        print("Volume Fractions:")
        for material, volume in volumes.items():
            print(f"Material {material}: {volume*100:.6f} %")

    return volumes

def get_image_laplacian(img, L = [1.,1.,1.]):

    s_img       = np.array(img.shape,dtype=int)

    # step 1: apply periodic Laplace stencil to the input image
    img_stencil = np.zeros(img.shape)
    L           = np.array(L, dtype=float)
    # dimensions of the finescale voxels
    lz      = L[0]/s_img[0]
    ly      = L[1]/s_img[1]
    lx      = L[2]/s_img[2]
    l_vx    = np.array([lz,ly,lx])

    # this stencil accounts for heterogeneous grid size along x, y, z
    f   = 3./(lx*ly/lz+lz*ly/lx+lx*lz/ly)
    img_stencil[0,0,0]      += 2.*f*(lx*ly/lz + lx*lz/ly + ly*lz/lx)
    img_stencil[(1,-1),0,0] += - f*lx*ly /lz
    img_stencil[0,(1,-1),0] += - f*lx*lz /ly
    img_stencil[0,0,(1,-1)] += - f*ly*lz /lx

    #----------------------------------------------------------------------
    # In case of an odd number of voxels the stencil can be computed
    # from a real-valued FFT (advantage: less memory effort, ...)
    even = (s_img[2] % 2 == 1)
    if( even ):
        iface= np.abs(ifftn( fftn(img.astype(float))*fftn(img_stencil) ))
    else:
        iface= np.abs(irfftn( rfftn(img.astype(float))*rfftn(img_stencil) ))

    return iface, l_vx

def supervoxel_normal(img, lap_img, l, vol_frac=None):
    """Get the normal vector from a supervoxel (given by img) and using the
    Laplacian on the image (lap_img). The Laplacian can be larger in size in order
    to get more accurate surface information (only for smooth surfaces),
    particularly if the supervoxels are small (resolution <= 4 for any axis).
    Then a least squares problem is solved to find the optimal normal vector.

    The normal is returned in (x,y,z) convention. The established normal cf.
    the approach of Kabel et al. (2015) is returned (ATTENTION: characteristic
    length is not adjusted for that).

    The normal points from the inclusion phase (1) into the matrix phase (0).

    :param img:     input pixels of the supervoxel under consideration (binary)
    :type img:      ndarray
    :param lap_img: like img, but the Laplacian of the img and (optional) of a different size
    :type lap_img:  ndarray
    :param l:       edge length of the fine scale voxels (lz, ly, lx), dtype=float
    :type l:        ndarray
    :param vol_frac: volume fraction of phase 1 (computed if None),
                    range: 0.0-1.0, defaults to None.
    :type vol_frac: float, optional
    """
    inline_norm = lambda x: np.sqrt( x[0]*x[0] +x[1]*x[1]+x[2]*x[2])

    assert(img.ndim == 3),'error: expecting 3D ndarray (0--> phase 0; else-->phase 1)'
    N       = np.array(img.shape)      # size of the supervoxel
    l_combi = l*np.array(img.shape)    # edge lengths of the supervoxel
    # assemble and solve the least squares problem
    w       = np.abs(lap_img)          # weights
    iface   = np.where(w > 1e-5)       # find the interface voxels
    w       = w[iface[0],iface[1],iface[2]].flatten()
    w       = w/w.sum()                # renormalize
    # coordinates of the 'on interface' voxels
    XX      = l[:,None]*np.array(iface)
    Xbar    = (XX*w).sum(axis=1)
    XX      = XX - Xbar[:,None]
    Mmod    = XX@(XX*w).T
    eigval, evec   = np.linalg.eigh(Mmod)
    EV       = evec[:,np.argmin(eigval)]
    # reorder: normal will have ordering (x, y, z) after the following line
    normal_xyz  = EV[::-1] / np.sqrt(EV[0]*EV[0]+EV[1]*EV[1]+EV[2]*EV[2])

    # concentration of phase '1' and its barycenter
    p_sum   = img.sum(axis=(1,2)) # x-y-slice sum
    n_red   = N[0]*N[1]*N[2]      # number of voxels in the supervoxel
    if( vol_frac is None ):
        c1  = p_sum.sum()/n_red
    else:
        c1  = vol_frac

    # a fast implementation that avoids redundant sub-summation
    # x1_c/y1_c/z1_c: barycenter of phase 1
    x       = []
    for i in range(3):
        x.append( np.linspace(-.5, .5, N[i]+1 )[:-1] + 0.5/ N[i]  )

    z1_c    = (p_sum*x[0]).sum()/(n_red*c1) * l_combi[0]

    p_sum   = img.sum(axis=0) # z-sum

    y1_c    = (p_sum.sum(axis=1)*x[1]).sum()/(n_red*c1) * l_combi[1] # x-sum
    x1_c    = (p_sum.sum(axis=0)*x[2]).sum()/(n_red*c1) * l_combi[2] # y-sum

    # flip direction if the normal points into phase 1:
    if( x1_c*normal_xyz[0] + y1_c*normal_xyz[1] + z1_c*normal_xyz[2] > 0. ):
        normal_xyz  *= -1.

    normal_classic = np.array([x1_c, y1_c, z1_c])
    normal_classic = -normal_classic / inline_norm(normal_classic)
    return normal_xyz, normal_classic

def downscale_image_periodic(data_array, Nx, Ny, Nz, min_vol_fraction=0.0, L=[1.,1.,1.], pad_window=[0,0,0]):
    """
    Downscale the given 3D image data_array to the size Nx x Ny x Nz considering periodic boundaries.
    
    Parameters:
    - data_array (numpy.ndarray): Fine-scale 3D numpy array.
    - Nx, Ny, Nz (int): Desired dimensions of the coarse-scale image.
    - min_vol_fraction (float): Minimum volume fraction to consider a material significant in a voxel.

    Returns:
    - numpy.ndarray: Coarse-scale 3D numpy array.
    - List[VoxelInfo]: Information about multi-material voxels.
    """
    coarse_data = np.zeros((Nx, Ny, Nz), dtype=int)
    nx, ny, nz = data_array.shape
    # Check divisibility
    if nx % Nx != 0 or ny % Ny != 0 or nz % Nz != 0:
        raise ValueError("Non-integer downscale factors detected due to indivisible dimensions. Please ensure that the dimensions of the fine-scale image are divisible by the desired coarse-scale dimensions.")
    dx, dy, dz = nx // Nx, ny // Ny, nz // Nz

    iface, l_vx = get_image_laplacian(data_array, L)
    
    multi_material_voxels = []
    element_number = -1
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Using modulo arithmetic to handle the wraparound
                block = data_array[(i*dx)%nx:((i+1)*dx)%nx, (j*dy)%ny:((j+1)*dy)%ny, (k*dz)%nz:((k+1)*dz)%nz]
                unique, counts = np.unique(block, return_counts=True)
                
                fractions = counts / counts.sum()
                significant_materials = unique[fractions >= min_vol_fraction]
                significant_fractions = fractions[fractions >= min_vol_fraction]
                element_number += 1

                normal = None
                if len(significant_materials) == 2:
                    xb = (i*dx - pad_window[0])%nx                 
                    xe = ((i+1)*dx + pad_window[0])%nx
                    yb = (j*dy - pad_window[1])%ny
                    ye = ((j+1)*dy + pad_window[1])%ny
                    zb = (k*dz - pad_window[2])%nz
                    ze = ((k+1)*dz + pad_window[2])%nz
                    block_pad = data_array[xb:xe, yb:ye, zb:ze]
                    
                    if np.array_equal(np.unique(block), np.unique(block_pad)):
                        iface_block = iface[xb:xe, yb:ye, zb:ze]
                        supervoxel = np.where(block_pad == significant_materials[0], 0, 1)
                    else:
                        iface_block = iface[(i*dx)%nx:((i+1)*dx)%nx, (j*dy)%ny:((j+1)*dy)%ny, (k*dz)%nz:((k+1)*dz)%nz]
                        supervoxel = np.where(block == significant_materials[0], 0, 1)

                    normal, N_classic =  supervoxel_normal( supervoxel, lap_img = iface_block, l = l_vx, vol_frac = significant_fractions[1])
                
                if len(significant_materials) == 1:
                    coarse_data[i, j, k] = significant_materials[0]
                elif len(significant_materials) > 1:
                    coarse_data[i, j, k] = -len(significant_materials)
                    voxel_info = VoxelInfo(coords=(i, j, k), elem_xyz=element_number, materials=significant_materials, fractions=significant_fractions, fine_scale_block=block, normal=normal)
                    multi_material_voxels.append(voxel_info)

    return coarse_data, multi_material_voxels, iface

def write_to_h5(filename, coarse_data = None, multi_material_voxels = None, fine_data = None, iface = None):
    """
    Write the coarse data, fine data, and multi-material voxel information to an HDF5 file.
    
    Parameters:
    - filename (str): Name of the HDF5 file to save to.
    - coarse_data (numpy.ndarray): Coarse-scale 3D numpy array.
    - multi_material_voxels (List[VoxelInfo]): List of multi-material voxel information.
    - fine_data (numpy.ndarray): Fine-scale 3D numpy array.
    """
    with h5py.File(filename, 'a') as f:
        # Saving the coarse_data if it is passed
        if coarse_data is not None:
            dset = f.create_dataset("coarse_data", data=coarse_data, dtype=int, compression="gzip", compression_opts=9)
        
        # Saving the fine_data if it is passed
        if fine_data is not None:
            dset_fine = f.create_dataset("fine_data", data=fine_data, dtype=np.uint8, compression="gzip", compression_opts=9)

        # Save laplacian of fine_data if it is passed
        if iface is not None:
            dset_iface = f.create_dataset("iface", data=iface, dtype=np.float16, compression="gzip", compression_opts=9)
        
        if multi_material_voxels:
            # Creating the coarse_normal dataset filled with zeros initially
            if coarse_data is not None:
                coarse_normal = np.zeros((coarse_data.shape[0], coarse_data.shape[1], coarse_data.shape[2], 3))
                for voxel_info in multi_material_voxels:
                    i, j, k = voxel_info.coords
                    normal_val = voxel_info.normal if voxel_info.normal is not None else (0.0, 0.0, 0.0)
                    coarse_normal[i, j, k] = normal_val
                dset_normal = f.create_dataset("coarse_normal", data=coarse_normal, dtype='f4', compression="gzip", compression_opts=9)
                
                # Saving the multi-material voxel information
                # Determine the maximum number of materials any voxel contains
                max_materials = max(len(voxel_info.materials) for voxel_info in multi_material_voxels)
                
                block_shape = multi_material_voxels[0].fine_scale_block.shape
            
            # Define structured numpy array dtype
            dtype = np.dtype([
                ('coords', '3i4'),
                ('elem_xyz', 'i4'),
                ('num_materials', 'i4'),
                ('materials', f'{max_materials}i4'),
                ('fractions', f'{max_materials}f4'),
                ('normal', '3f4'),
                ('fine_scale_block', f'{block_shape[0]},{block_shape[1]},{block_shape[2]}i4')
            ])
            structured_array = np.zeros(len(multi_material_voxels), dtype=dtype)
            
            for idx, voxel_info in enumerate(multi_material_voxels):
                materials = np.pad(voxel_info.materials.astype(np.int32), (0, max_materials - len(voxel_info.materials)), constant_values=-1)
                fractions = np.pad(voxel_info.fractions, (0, max_materials - len(voxel_info.fractions)), constant_values=-1.0)
                num_materials = len(voxel_info.materials)
                normal_val = voxel_info.normal if voxel_info.normal is not None else (0.0, 0.0, 0.0)
                
                structured_array[idx] = (voxel_info.coords, voxel_info.elem_xyz, num_materials, materials, fractions, normal_val, voxel_info.fine_scale_block)
            
            # Create the dataset for multi-material voxel information
            mm_dset = f.create_dataset("multi_material_voxels", data=structured_array, dtype=dtype, compression="gzip", compression_opts=9)