import h5py
import numpy as np
import argparse
import os
import datetimes


def load_surface(h5_path, dataset_name='loss'):
    """
    Load loss values (and coordinates if present) from an H5 file.
    """
    with h5py.File(h5_path, 'r') as f:
        if dataset_name in f:
            data = f[dataset_name][()]
        else:
            raise KeyError(f"Dataset '{dataset_name}' not found. Available keys: {list(f.keys())}")
        n = data.shape[0]
        if 'x' in f and 'y' in f:
            x = f['x'][()]
            y = f['y'][()]
        else:
            x = np.linspace(-1.0, 1.0, n)
            y = np.linspace(-1.0, 1.0, n)
    return x, y, data


def compute_metrics(x, y, loss, epsilon=0.1):
    """
    Compute basic metrics: min, mean, location of min, Hessian trace at the minimum,
    sharpness and normalized sharpness within radius epsilon.
    """
    # minimum and mean values
    min_val = np.min(loss)
    mean_val = np.mean(loss)
    x_0 = np.unravel_index(np.argmin(loss), loss.shape)
    min_point = (x[x_0[1]], y[x_0[0]])

    # approximate hessian finite differences, 2nd derivatives, compute trace
    d2_dx2 = np.gradient(np.gradient(loss, x, axis=1), x, axis=1)
    d2_dy2 = np.gradient(np.gradient(loss, y, axis=0), y, axis=0)
    trace_h = d2_dx2 + d2_dy2
    trace_at_min = trace_h[x_0]
    # epsilon sharpness
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt((X - min_point[0])**2 + (Y - min_point[1])**2)
    mask = dist <= epsilon
    local_losses = loss[mask]
    sharpness = np.max(local_losses) - min_val

    # Normalized sharpness
    norm_sharpness = sharpness / (1.0 + min_val)

    return {
        'min_val': min_val,
        'mean_val': mean_val,
        'min_point': min_point,
        'trace_at_min': trace_at_min,
        'sharpness': sharpness,
        'normalized_sharpness': norm_sharpness
    }


def count_extrema(loss):
    """
    Count local minima and maxima in a 2D grid by comparing each cell to its 8 neighbors.
    """
    minima = 0
    maxima = 0
    rows, cols = loss.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            window = loss[i-1:i+2, j-1:j+2]
            center = loss[i, j]
            neighbors = np.delete(window.flatten(), 4)
            if center < neighbors.min():
                minima += 1
            if center > neighbors.max():
                maxima += 1
    return minima, maxima


def main():
    parser = argparse.ArgumentParser(description='Analyze loss landscape from H5 files.')
    parser.add_argument('h5_paths', nargs='+', help='Paths to one or more H5 files containing loss surfaces.')
    parser.add_argument('--dataset', default='loss', help='Name of the dataset inside H5 files.')
    parser.add_argument('--save', default=False, help='Save plots.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Radius around minimum for sharpness metric')
    args = parser.parse_args()

    for path in args.h5_paths:
        try:
            x, y, loss = load_surface(path, args.dataset)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        metrics = compute_metrics(x, y, loss, args.epsilon)
        minima_count, maxima_count = count_extrema(loss)

        lines = [
            f"File: {path}",
            f"Minimum loss: {metrics['min_val']:.4f} at {metrics['min_point']}",
            f"Mean loss: {metrics['mean_val']:.4f}",
            f"Hessian trace at minimum: {metrics['trace_at_min']:.2f}",
            f"Sharpness (ε={args.epsilon}): {metrics['sharpness']:.4f}",
            f"Normalized sharpness: {metrics['normalized_sharpness']:.4f}",
            f"Local minima count: {minima_count}",
            f"Local maxima count: {maxima_count}"
        ]
        
        print("\n" + "\n".join(lines))

        # Save to file
        if args.save:
            base = os.path.splitext(os.path.basename(path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"{base}_{timestamp}.txt"
            with open(out_name, 'w') as f:
                f.write("\n".join(lines))

        if args.plot:
            plot_surface(x, y, loss, title=path)


if __name__ == '__main__':
    main()
