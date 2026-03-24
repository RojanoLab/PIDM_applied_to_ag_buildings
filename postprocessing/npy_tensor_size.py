import argparse
from pathlib import Path

import numpy as np


def format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print tensor size information from a .npy file."
    )
    parser.add_argument("input_file", type=Path, help="Path to the .npy file")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"File not found: {args.input_file}")

    array = np.load(args.input_file)

    print(f"File: {args.input_file}")
    print(f"Tensor size (shape): {array.shape}")
    print(f"Dimensions (ndim): {array.ndim}")
    print(f"Data type (dtype): {array.dtype}")
    print(f"Total elements: {array.size}")
    print(f"Bytes per element: {array.itemsize}")
    print(f"Total memory: {array.nbytes} bytes ({format_bytes(array.nbytes)})")


if __name__ == "__main__":
    main()