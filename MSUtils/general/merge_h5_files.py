import argparse
from pathlib import Path

import h5py


def copy_contents(src_group, dst_group, verbose=False, overwrite=True):
    """Recursively merge an HDF5 group into another group."""
    for key, value in src_group.attrs.items():
        if overwrite or key not in dst_group.attrs:
            dst_group.attrs[key] = value

    for key, item in src_group.items():
        path = f"{dst_group.name.rstrip('/')}/{key}"
        if key not in dst_group:
            if verbose:
                print(f"Copying: {path}")
            src_group.copy(key, dst_group)
            continue

        destination = dst_group[key]
        if isinstance(item, h5py.Group) and isinstance(destination, h5py.Group):
            copy_contents(item, destination, verbose=verbose, overwrite=overwrite)
        elif overwrite:
            if verbose:
                print(f"Overwriting: {path}")
            del dst_group[key]
            src_group.copy(key, dst_group)
        elif verbose:
            print(f"Skipping existing object: {path}")


def merge_h5_files(output_file, input_files, verbose=False, overwrite=True):
    """Merge HDF5 files in order into one output file."""
    input_files = tuple(input_files)
    output_path = Path(output_file).resolve()
    if any(Path(input_file).resolve() == output_path for input_file in input_files):
        raise ValueError(
            "The output file is updated in place; omit it from the input files."
        )

    with h5py.File(output_file, "a") as h5out:
        for infile in input_files:
            if verbose:
                print(f"Merging file: {infile}")
            with h5py.File(infile, "r") as h5in:
                copy_contents(h5in, h5out, verbose=verbose, overwrite=overwrite)

    if verbose:
        print(f"Merging complete. Output saved at {output_file}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 files into a single output HDF5 file."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output HDF5 file."
    )
    parser.add_argument(
        "-i", "--inputs", nargs="+", required=True, help="Input HDF5 files to merge."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed information about the merging process.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        default=True,
        help="Keep existing objects and attributes.",
    )

    args = parser.parse_args(argv)
    merge_h5_files(
        args.output, args.inputs, verbose=args.verbose, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
