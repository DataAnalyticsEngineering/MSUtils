import sys, os, h5py, numpy as np
from ComBoNormal import *
sys.path.append(os.path.join('general/'))
from resize_image import resize_image
from MicrostructureImage import MicrostructureImage
from ComBoMicrostructureImage import ComBoMicrostructureImage


if __name__ == '__main__':

    ms = MicrostructureImage(h5_filename='data/sphere.h5', dset_name='/sphere/256x256x256/ms')
    ms = MicrostructureImage(image=resize_image(ms.image, target_resolution=[256,256,256]))
    print(ms.volume_fractions)

    # Nx, Ny, Nz = 64, 64, 64
    # min_vol_fraction = 0.00
    # L = [1,1,1]
    # downscaled, voxel_infos, iface = downscale_image_periodic(ms.image, Nx, Ny, Nz, min_vol_fraction, L, pad_window=[2,2,2])
    # write_to_h5('data/test2.h5', fine_data=ms.image, coarse_data=downscaled, multi_material_voxels=voxel_infos)
    # print(compute_material_volumes_in_image(downscaled, voxel_infos))

    combo_micro_img = ComBoMicrostructureImage()
    combo_micro_img.downscale(ms.image, 64, 64, 64, pad_window=[2,2,2])
    combo_micro_img.write('data/test.h5', '/combo_grp')
    print(combo_micro_img.volume_fractions)    

    combo_micro_img.read('data/test.h5', '/combo_grp')


































    # # Extract the combined smoothed surface mesh from the data array for all materials
    # surface_mesh = extract_surfaces_from_array(data_array)
    # smoothed_mesh = smooth_mesh(surface_mesh, iterations=40, pass_band=0.05)
    # # Save the smoothed surface mesh to a VTK file
    # vtk_filename = 'surface_mesh.vtk'  # Output VTK file path
    # save_polydata_to_vtk(smoothed_mesh, vtk_filename)
    # print(f"Saved surface mesh to {vtk_filename}. You can now visualize it using ParaView.")
