#!/usr/bin/env python3
"""
inspect_h5.py

Prints out the structure and metadata of an HDF5 (.h5/.hdf5) file.
"""

import os
import argparse
import h5py

def print_attrs(obj, indent=0):
    """Print all attributes of an HDF5 object."""
    prefix = ' ' * indent
    if obj.attrs:
        print(f"{prefix}Attributes:")
        for key, val in obj.attrs.items():
            print(f"{prefix}  • {key}: {val}")
    else:
        print(f"{prefix}(no attributes)")

def inspect_file(fname):
    # File size on disk
    size_bytes = os.path.getsize(fname)
    print(f"File: {fname}")
    print(f"Size on disk: {size_bytes} bytes\n")

    with h5py.File(fname, 'r') as f:
        # Root‐level attributes
        print("Root-level attributes:")
        print_attrs(f, indent=2)
        print()

        # Walk through every object in the file
        def visitor(name, obj):
            indent_level = name.count('/') * 2
            if isinstance(obj, h5py.Group):
                print(f"{' ' * indent_level}Group: /{name}")
                print_attrs(obj, indent=indent_level + 2)
            elif isinstance(obj, h5py.Dataset):
                print(f"{' ' * indent_level}Dataset: /{name}")
                print(f"{' ' * (indent_level + 2)}shape: {obj.shape}")
                print(f"{' ' * (indent_level + 2)}dtype: {obj.dtype}")
                print(f"{' ' * (indent_level + 2)}compression: {obj.compression}")
                print(f"{' ' * (indent_level + 2)}compression_opts: {obj.compression_opts}")
                print(f"{' ' * (indent_level + 2)}chunks: {obj.chunks}")
                print(f"{' ' * (indent_level + 2)}fillvalue: {obj.fillvalue}")
                print_attrs(obj, indent=indent_level + 2)
            else:
                print(f"{' ' * indent_level}Unknown object: /{name}")
            print()  # blank line for readability

        f.visititems(visitor)

def main():
    parser = argparse.ArgumentParser(
        description="Inspect an HDF5 file and list its groups, datasets, and attributes."
    )
    parser.add_argument("file", help="Path to the HDF5 file (.h5, .hdf5)")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}")
        return

    inspect_file(args.file)

if __name__ == "__main__":
    main()
