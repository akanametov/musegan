"""Load and save an array to shared memory."""
# ----------------------------------------------------------------------------------
# Modified from: https://github.com/salu133445/musegan/blob/main/src/process_data.py
# ----------------------------------------------------------------------------------

import argparse
import os.path
import sys

import numpy as np
import SharedArray as sa


def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to the data file.")
    parser.add_argument(
        "--name",
        default="train_x_lpd_5",
        help="File name to save in SharedArray. Defaults to the original file name.",
    )
    parser.add_argument(
        "--prefix",
        default=".npz",
        help="Prefix to the file name to save in SharedArray. Only effective when "
        "`name` is not given.",
    )
    parser.add_argument(
        "--dtype", default="bool", help="Datatype of the array. Defaults to bool."
    )
    args = parser.parse_args()
    return args.filepath, args.name, args.prefix, args.dtype


def create_shared_array(name, shape, dtype):
    """Create shared array. Prompt if a file with the same name existed."""
    try:
        return sa.create(name, shape, dtype)
    except FileExistsError:
        response = ""
        while response.lower() not in ["y", "n", "yes", "no"]:
            response = input(
                "Existing array (also named " + name + ") was found. Replace it? (y/n) "
            )
        if response.lower() in ("n", "no"):
            sys.exit(0)
        sa.delete(name)
        return sa.create(name, shape, dtype)


def main():
    """Load and save an array to shared memory."""
    filepath, name, prefix, dtype = parse_arguments()

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]
        if prefix is not None:
            name = prefix + "_" + name

    print("Loading data from '{}'.".format(filepath))
    if filepath.endswith(".npy"):
        data = np.load(filepath)
        data = data.astype(dtype)
        sa_array = create_shared_array(name, data.shape, data.dtype)
        print("Saving data to shared memory...")
        np.copyto(sa_array, data)
    else:
        with np.load(filepath) as loaded:
            sa_array = create_shared_array(name, loaded["shape"], dtype)
            print("Saving data to shared memory...")
            sa_array[[x for x in loaded["nonzero"]]] = 1

    print(
        "Successfully saved: (name='{}', shape={}, dtype={})".format(
            name, sa_array.shape, sa_array.dtype
        )
    )
    # (102378, 4, 48, 84, 5) -> (102378, 4, 5, 48, 84)
    sa_array = sa_array.transpose(0, 1, 4, 2, 3)
    np.savez(name + ".npz", sa_array)
    sa.delete(name)


if __name__ == "__main__":
    main()
