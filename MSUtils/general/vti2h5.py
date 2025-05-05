import pyvista as pv
import h5py
import os
import sys
import argparse

def multi_vti_to_h5(vti_files, h5_file_path, grp_name="images", data_field=None, overwrite=False):
    """
    Convert multiple VTI files to a single H5 file
    
    Parameters:
    -----------
    vti_files : list
        List of paths to the input VTI files
    h5_file_path : str
        Path to the output H5 file
    grp_name : str, optional
        Name of the group to store the datasets (default: "images")
    data_field : str, optional
        Name of the data field to extract. If None, will use the first available point data
    overwrite : bool, optional
        If True, will overwrite existing datasets. If False, will abort writing existing datasets.
    """
        
    # Determine mode based on whether file exists
    mode = 'a' if os.path.exists(h5_file_path) else 'w'
    
    with h5py.File(h5_file_path, mode) as h5file:
        # Create or get existing group
        if grp_name in h5file:
            images_group = h5file[grp_name]
            print(f"Using existing group: {grp_name}")
        else:
            images_group = h5file.create_group(grp_name)
            print(f"Created new group: {grp_name}")
        
        # Process each VTI file
        for vti_file in vti_files:
            # Extract base filename and remove .vti extension
            dset_name = os.path.splitext(os.path.basename(vti_file))[0]
            print(f"Reading VTI file: {vti_file}")
            
            # Check if dataset already exists
            if dset_name in images_group:
                if overwrite:
                    print(f"Dataset {dset_name} already exists. Overwriting.")
                    del images_group[dset_name]
                else:
                    print(f"Dataset {dset_name} already exists and overwrite=False. Skipping.")
                    continue
            
            try:
                mesh = pv.read(vti_file)
                
                # Try point data first, then cell data
                data_dict = mesh.point_data or mesh.cell_data
                
                if not data_dict:
                    print(f"Error: No data found in {vti_file}. Skipping.")
                    continue
                
                # Select data field
                field_name = data_field if data_field in data_dict else list(data_dict.keys())[0]
                print(f"Using data field: {field_name}")
                
                # Reshape data to grid dimensions
                meshdims = [dim-1 for dim in mesh.dimensions]
                data_array = data_dict[field_name].reshape(meshdims)
                
                # Create dataset
                images_group.create_dataset(
                    dset_name, 
                    data=data_array,
                    compression='gzip',
                    compression_opts=9,
                    chunks=tuple(meshdims)
                )
                
                # Store metadata
                images_group[dset_name].attrs.update({
                    'mesh_dimensions': mesh.dimensions,
                    'mesh_spacing': mesh.spacing,
                    'mesh_origin': mesh.origin,
                    'source_file': os.path.basename(vti_file),
                    'data_field': field_name
                })
                
            except Exception as e:
                print(f"Error processing {vti_file}: {str(e)}")
                continue
        
        # Update global attributes
        h5file.attrs['num_images'] = len(images_group)
    
    print(f"Conversion complete: processed {len(vti_files)} files → {h5_file_path}")

def main():
    # Example usage
    vti_files = [
        "data/spinodoid_network_np_isotropic_x192_30pi_dens50_0001.vti", 
        "data/spinodoid_network_np_isotropic_x192_30pi_dens50_0002.vti",
        "data/spinodoid_network_np_isotropic_x192_30pi_dens50_0003.vti"
    ]
    h5_file_path = "data/spinodoid_collection.h5"
    multi_vti_to_h5(vti_files=vti_files, h5_file_path=h5_file_path, grp_name="/spinodoids/isotropic", overwrite=True)
    

if __name__ == "__main__":
    
    # If no arguments provided, run the example in main()
    if len(sys.argv) == 1:
        main()
    else:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Convert VTI files to HDF5 format")
        parser.add_argument("--vti-files", nargs="+", required=True, help="List of VTI files to convert")
        parser.add_argument("--output", "-o", required=True, help="Output HDF5 file path")
        parser.add_argument("--group", "-g", default="images", help="Group name within H5 file (default: images)")
        parser.add_argument("--data-field", "-d", help="Specific data field to extract (optional)")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite existing datasets")
        
        args = parser.parse_args()
        
        multi_vti_to_h5(
            vti_files=args.vti_files, 
            h5_file_path=args.output, 
            grp_name=args.group, 
            data_field=args.data_field,
            overwrite=args.overwrite
        )
        
        # CLI examples:
        
        # Convert multiple VTI files to
        # python vti2h5.py --vti-files data/file1.vti data/file2.vti data/file3.vti --output output.h5 --group /spinodoids/isotropic --overwrite

        # Extract a specific data field
        # python vti2h5.py --vti-files data/*.vti --output collection.h5 --group images --data-field density

        # Basic usage with default group name
        # python vti2h5.py --vti-files data/spinodoid_*.vti --output spinodoids.h5
