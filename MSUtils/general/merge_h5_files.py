import os
import h5py
import argparse

def copy_contents(src_group, dst_group, verbose=False, overwrite=True):
    """
    Recursively copy all datasets and groups from src_group into dst_group.
    
    If a dataset already exists at the same path in dst_group:
    - If overwrite=True, the dataset is overwritten.
    - If overwrite=False, the dataset is skipped.

    Parameters
    ----------
    src_group : h5py.Group
        The source HDF5 group to copy from.
    dst_group : h5py.Group
        The destination HDF5 group to copy into.
    verbose : bool
        If True, print detailed information about operations.
    overwrite : bool
        If True, overwrite existing datasets. If False, skip them.
    """
    for key, item in src_group.items():
        if isinstance(item, h5py.Group):
            # Create or use existing group
            if key not in dst_group:
                if verbose:
                    print(f"Creating group: {dst_group.name}/{key}")
                new_group = dst_group.create_group(key)
            else:
                new_group = dst_group[key]

            # Recursively copy the contents
            copy_contents(item, new_group, verbose=verbose, overwrite=overwrite)

        elif isinstance(item, h5py.Dataset):
            # If dataset doesn't exist in the destination, copy it
            if key not in dst_group:
                if verbose:
                    print(f"Copying new dataset: {dst_group.name}/{key}")
                dst_group.copy(item, key)
            else:
                # Dataset exists
                if overwrite:
                    # Overwrite it
                    if verbose:
                        print(f"Overwriting dataset: {dst_group.name}/{key}")
                    del dst_group[key]
                    dst_group.copy(item, key)
                else:
                    # Skip overwriting
                    if verbose:
                        print(f"Skipping dataset (already exists): {dst_group.name}/{key}")

def merge_h5_files(output_file, input_files, verbose=False, overwrite=True):
    """
    Merge multiple HDF5 files into a single output HDF5 file.

    The entire hierarchy (groups/datasets) from each input file will be merged
    into the output file. If any dataset path already exists in the output file:
    - If overwrite=True, it will be overwritten with the dataset from the current input file.
    - If overwrite=False, it will be skipped.

    Parameters
    ----------
    output_file : str
        Path to the output HDF5 file.
    input_files : list of str
        Paths to the input HDF5 files to be merged.
    verbose : bool, optional
        If True, print detailed information about the merging process.
    overwrite : bool, optional
        If True (default), existing datasets are overwritten. If False, they are skipped.
    """
    mode = 'a' if os.path.exists(output_file) else 'w'
    with h5py.File(output_file, mode) as h5out:
        for infile in input_files:
            if verbose:
                print(f"Merging file: {infile}")
            with h5py.File(infile, 'r') as h5in:
                copy_contents(h5in, h5out, verbose=verbose, overwrite=overwrite)

    if verbose:
        print(f"Merging complete. Output saved at {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 files into a single output HDF5 file.")
    parser.add_argument('-o', '--output', required=True,
                        help='Path to the output HDF5 file.')
    parser.add_argument('-i', '--inputs', nargs='+', required=True,
                        help='Input HDF5 files to merge.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information about the merging process.')
    parser.add_argument('--no-overwrite', action='store_false', dest='overwrite', default=True,
                        help='If set, existing datasets are not overwritten, but skipped.')

    args = parser.parse_args()
    merge_h5_files(args.output, args.inputs, verbose=args.verbose, overwrite=args.overwrite)

if __name__ == "__main__":
    # main()

    # Example usage from Python
    output_file = 'path/to/aggregated_h5_file.h5'
    input_files = [f'path/to/individual/h5_file_{k}.h5' for k in range(0, 20)]
    verbose = False
    overwrite = True
    merge_h5_files(output_file, input_files, verbose=verbose, overwrite=overwrite)
    
    # # Example usage from command line
    # # nohup /usr/bin/time -v python merge_h5_files.py> nohup_test.txt 2>&1 &