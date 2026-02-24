import h5py
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plot_1D

def plot_surface_from_h5(
    surf_file,
    surf_name='train_loss',
    elev=30,
    azim=45,
    zscale=1.0,
    zmin=None,
    zmax=None,
    vmin=None,
    vmax=None,
    cmap='coolwarm',
    title=None,
    show=False
):
    """
    Load a 2D loss surface from an existing .h5 file and create a customizable 3D surface plot.

    Parameters:
    - surf_file: path to the .h5 file containing 'xcoordinates', 'ycoordinates', and surface data
    - surf_name: the key in the .h5 file to plot (e.g., 'train_loss', 'test_loss', 'train_err')
    - elev, azim: elevation and azimuth angles for view (degrees)
    - zscale: scale factor to apply to the Z axis
    - zmin, zmax: absolute limits for the Z axis
    - vmin, vmax: absolute colorbar limits for Z values (ensures consistent coloring across plots)
    - cmap: matplotlib colormap for the surface
    - title: optional plot title
    - show: whether to call plt.show() at the end
    """
    with h5py.File(surf_file, 'r') as f:
        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])
        X, Y = np.meshgrid(x, y)

        # handle error metrics
        if surf_name in f:
            Z = np.array(f[surf_name][:])
        elif surf_name.endswith('_err'):
            Z = 100 - np.array(f[surf_name.replace('_err', '_acc')][:])
        else:
            raise KeyError(f"{surf_name} not found in {surf_file}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Z_plot = Z * zscale

    # Plot surface
    surf = ax.plot_surface(
        X, Y, Z_plot,
        cmap=cmap,
        linewidth=0,
        antialiased=True
    )

    # enforce consistent color scaling if provided
    if vmin is not None or vmax is not None:
        surf.set_clim(vmin, vmax)

    # customize view, labels, and title
    ax.view_init(elev=elev, azim=azim)
    #ax.set_xlabel('X coordinate')
    #ax.set_ylabel('Y coordinate')
    ax.set_zlabel(surf_name, labelpad=10)
    if title:
        ax.set_title(title)

    # apply absolute z-axis limits if provided
    if zmin is not None or zmax is not None:
        current_min, current_max = ax.get_zlim()
        low = zmin if zmin is not None else current_min
        high = zmax if zmax is not None else current_max
        ax.set_zlim(low, high)

    # Colorbar updates automatically
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()
    outname = f"{surf_file}_{surf_name}_3dsurface_custom.pdf"
    fig.savefig(outname, dpi=300, bbox_inches='tight')
    print(f"Saved 3D surface to {outname}")
    if show:
        plt.show()


def plot_accuracy_from_h5(
    surf_file,
    xmin=None,
    xmax=None,
    loss_max=5,
    log=False,
    title=None,
    show=False
):
    """
    Load 1D training loss and accuracy from an existing .h5 file and reproduce the accuracy plot.

    Parameters mirror plot_1d_loss_err:
    - surf_file: path to the .h5 file containing 'xcoordinates', 'train_loss', 'train_acc', etc.
    - xmin, xmax: bounds on the x-axis (defaults to file values)
    - loss_max: max loss for y-axis
    - log: log-scale for loss
    - title: optional plot title (applied via the delegated function if supported)
    - show: whether to call plt.show()
    """
    plot_1D.plot_1d_loss_err(
        surf_file,
        xmin=xmin if xmin is not None else -1.0,
        xmax=xmax if xmax is not None else 1.0,
        loss_max=loss_max,
        log=log,
        title=title,
        show=show
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Custom plotting for existing loss surface .h5 files')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # 3D surface
    p3 = sub.add_parser('surface', help='Create a 3D surface plot')
    p3.add_argument('-f', '--surf_file', required=True)
    p3.add_argument('-s', '--surf_name', default='train_loss')
    p3.add_argument('--elev', type=float, default=30)
    p3.add_argument('--azim', type=float, default=45)
    p3.add_argument('--zscale', type=float, default=1.0)
    p3.add_argument('--zmin', type=float, help='Minimum z-axis value')
    p3.add_argument('--zmax', type=float, help='Maximum z-axis value')
    p3.add_argument('--vmin', type=float, help='Minimum colorbar limit for Z')
    p3.add_argument('--vmax', type=float, help='Maximum colorbar limit for Z')
    p3.add_argument('--cmap', default='coolwarm')
    p3.add_argument('--title', type=str, help='Title for the plot')
    p3.add_argument('--show', action='store_true')

    # accuracy plot
    p1 = sub.add_parser('accuracy', help='Create 1D loss and accuracy plots')
    p1.add_argument('-f', '--surf_file', required=True)
    p1.add_argument('--xmin', type=float)
    p1.add_argument('--xmax', type=float)
    p1.add_argument('--loss_max', type=float, default=5)
    p1.add_argument('--log', action='store_true')
    p1.add_argument('--title', type=str, help='Title for the plot')
    p1.add_argument('--show', action='store_true')

    args = parser.parse_args()
    if args.cmd == 'surface':
        plot_surface_from_h5(
            args.surf_file,
            surf_name=args.surf_name,
            elev=args.elev,
            azim=args.azim,
            zscale=args.zscale,
            zmin=args.zmin,
            zmax=args.zmax,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            title=args.title,
            show=args.show
        )
    elif args.cmd == 'accuracy':
        plot_accuracy_from_h5(
            args.surf_file,
            xmin=args.xmin,
            xmax=args.xmax,
            loss_max=args.loss_max,
            log=args.log,
            title=args.title,
            show=args.show
        )
