import argparse
import h5py
import numpy as np
from scipy.interpolate import griddata
import os

def load_data(filepath):
    with h5py.File(filepath, "r") as f:
        x = f["xcoordinates"][:]
        y = f["ycoordinates"][:]
        z = f["train_loss"][:]
    return x, y, z

def interpolate_data(x, y, z, resolution=100):
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = z.ravel()

    x_fine = np.linspace(x.min(), x.max(), resolution)
    y_fine = np.linspace(y.min(), y.max(), resolution)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    Z_fine = griddata(points, values, (X_fine, Y_fine), method='cubic')
    return x_fine, y_fine, Z_fine

def save_interpolated_data(output_dir, original_filename, x_fine, y_fine, z_fine):
    base = os.path.splitext(os.path.basename(original_filename))[0]
    outpath = os.path.join(output_dir, f"{base}_interpolated.h5")
    with h5py.File(outpath, "w") as f:
        f.create_dataset("xcoordinates", data=x_fine)
        f.create_dataset("ycoordinates", data=y_fine)
        f.create_dataset("train_loss", data=z_fine)
    print(f"Interpolated file saved to: {outpath}")

def main():
    parser = argparse.ArgumentParser(description="Interpolate 3D surface data from an HDF5 file.")
    parser.add_argument("filename", type=str, help="Path to the input .h5 file")
    parser.add_argument("output", type=str, help="Directory to save the interpolated .h5 file")
    parser.add_argument("--resolution", type=int, default=100, help="Resolution for interpolation grid")
    args = parser.parse_args()

    x, y, z = load_data(args.filename)
    x_fine, y_fine, z_fine = interpolate_data(x, y, z, args.resolution)
    save_interpolated_data(args.output, args.filename, x_fine, y_fine, z_fine)

if __name__ == "__main__":
    main()
