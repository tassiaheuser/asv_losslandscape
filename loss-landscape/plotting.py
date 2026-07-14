import argparse
import h5py
import numpy as np
import datetime
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LogNorm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns


def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, zmax=0, show=False, detail_surf_file=None, out_dir="Output/new_figures"):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])

    print(f.keys())

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    # if a detailed surface file is provided, add its x, y coordinates and z values to the plot
    if detail_surf_file is not None:
        if not exists(detail_surf_file):
            print(f"Detail surface file {detail_surf_file} does not exist.")
            return
        detail_f = h5py.File(detail_surf_file, 'r')
        detail_x = np.array(detail_f['xcoordinates'][:])
        detail_y = np.array(detail_f['ycoordinates'][:])
        if surf_name in detail_f.keys():
            detail_Z = np.array(detail_f[surf_name][:])
        elif surf_name == 'train_err' or surf_name == 'test_err':
            detail_Z = 100 - np.array(detail_f[surf_name][:])
        else:
            print ('%s is not found in %s' % (surf_name, detail_surf_file))
            return

        # Combine the coordinates and values
        x_pointer_large = 0
        x_pointer_detail = 0
        y_pointer_large = 0
        y_pointer_detail = 0
        x_pointer = np.min((x,detail_x))
        y_pointer = np.min((y,detail_y))

        new_x = []
        new_y = []
        new_Z = []
        while y_pointer_large < len(y) and y_pointer_detail < len(detail_y):
            if y[y_pointer_large] < detail_y[y_pointer_detail]:
                x_pointer_large = 0
                while x_pointer_large < len(x):
                    new_y.append(y[y_pointer_large])
                    new_x.append(x[x_pointer_large])
                    new_Z.append(Z[y_pointer_large, x_pointer_large])
                    x_pointer_large += 1
                y_pointer_large += 1
                continue
            elif y[y_pointer_large] > detail_y[y_pointer_detail]:
                x_pointer_detail = 0
                while x_pointer_detail < len(detail_x):
                    new_y.append(detail_y[y_pointer_detail])
                    new_x.append(detail_x[x_pointer_detail])
                    new_Z.append(detail_Z[y_pointer_detail, x_pointer_detail])
                    x_pointer_detail += 1
                y_pointer_detail += 1
                continue
            elif y[y_pointer_large] == detail_y[y_pointer_detail]:
                while x_pointer_large < len(x) and x_pointer_detail < len(detail_x):
                    new_y.append(detail_y[y_pointer_detail])
                    if x[x_pointer_large] < detail_x[x_pointer_detail]:
                        new_x.append(x[x_pointer_large])
                        new_Z.append(Z[y_pointer_large, x_pointer_large])
                        x_pointer_large += 1
                    elif x[x_pointer_large] > detail_x[x_pointer_detail]:
                        new_x.append(detail_x[x_pointer_detail])
                        new_Z.append(detail_Z[y_pointer_detail, x_pointer_detail])
                        x_pointer_detail += 1
                    else:
                        new_x.append(x[x_pointer_large])
                        new_Z.append(Z[y_pointer_large, x_pointer_large])
                        x_pointer_large += 1
                        x_pointer_detail += 1
                y_pointer_large += 1
                y_pointer_detail += 1
                continue

        ax = plt.figure().add_subplot(projection='3d')
        surf = ax.plot_trisurf(new_x,new_y,new_Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.savefig(os.path.join(out_dir,".".join(surf_file.split(".")[:-1]) + '_' + surf_name + '_3dsurface_triangular.pdf'), dpi=300,
                    bbox_inches='tight', format='pdf')
        return

    X, Y = np.meshgrid(x, y)


            #         new_x.append(detail_x[x_pointer_detail])
            #     y_pointer_large += 1
            #     y_pointer_detail += 1
            #     y_list =
            # new_Z.append([])
            # while x_pointer_large < len(x) and x_pointer_detail < len(detail_x):
            #     if x[x_pointer_large] < x[x_pointer_detail]:
            #         new_x.append(x[x_pointer_large])
            #         x_pointer_large += 1
            #     elif x[x_pointer_large] > x[x_pointer_detail]:
            #         new_x.append(x[x_pointer_detail])
            #         x_pointer_detail += 1
            #     if x[x_pointer_large] == x[x_pointer_detail]:
            #         new_x.append(x[x_pointer_large])
            #         x_pointer_large += 1
            #         x_pointer_detail += 1
            #     new_Z[-1].append(Z[y_pointer_large, x_pointer_large])

    print('\n------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    surf_file_no_fileending = ".".join(surf_file.split(".")[:-1])
    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(os.path.join(out_dir,surf_file_no_fileending + '_' + surf_name + '_2dcontour' + '.pdf'), dpi=300,
                bbox_inches='tight', format='pdf')

    fig = plt.figure()
    print(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf')
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(os.path.join(out_dir,surf_file_no_fileending + '_' + surf_name + '_2dcontourf' + '.pdf'), dpi=300,
                bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    fig = plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(os.path.join(out_dir,surf_file_no_fileending + '_' + surf_name + '_2dheat.pdf'),
                                  dpi=300, bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    print("-------------------")
    print("-------------------")
    print("-------------------")
    print("-------------------")
    print("-------------------")
    print("")
    print("")
    print(zmax)
    print("")
    print("")
    print("-------------------")
    print("-------------------")
    print("-------------------")
    print("-------------------")
    print("-------------------")
    if zmax != 0:
        ax.set_zlim([0,zmax])
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(os.path.join(out_dir,".".join(surf_file.split(".")[:-1]) + '_' + surf_name + '_3dsurface.pdf'), dpi=300,
                bbox_inches='tight', format='pdf')

    if show: plt.show()
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recreate 2D contour plots from an H5 file')

    # Required arguments
    parser.add_argument('--surf_file', required=True, type=str, help='Path to the H5 surface file')
    parser.add_argument('--detail_surf_file', required=False, type=str, default=None, help='Path to a second H5 surface file')

    # Plot settings
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value for contour mapping')
    parser.add_argument('--vmin', default=0.1, type=float, help='Minimum value for contour mapping')
    parser.add_argument('--vlevel', default=0.5, type=float, help='Contour levels interval')
    parser.add_argument('--show', action='store_true', default=False, help='Show plotted figures')
    parser.add_argument('--zmax', default=0, type=float, help='Set Z direction (optional)')
    parser.add_argument('--out_dir', default='"Output/new_figures"', type=str, help='output directory for the figures')

    args = parser.parse_args()

    # Timestamp for logging
    now = datetime.datetime.now()
    args.timestamp = now.strftime("%d-%mT%H-%M-%S")

    time_dir = True
    if time_dir:
        args.out_dir = os.path.join(args.out_dir, args.timestamp)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir,exist_ok=True)

    # Generate 2D contour plots
    print(f"Generating 2D contour plots from {args.surf_file}...")
    plot_2d_contour(args.surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.zmax, args.show, detail_surf_file=args.detail_surf_file,out_dir=args.out_dir)
    plot_2d_contour(args.surf_file, 'train_acc', 0, 100, args.vlevel, args.zmax, args.show, detail_surf_file=args.detail_surf_file,out_dir=args.out_dir)

    print("Plotting complete. Files saved successfully.")

    # python plotting.py --surf_file "Output/res2net8s/26-05T12-22-55/surface_[-1.0,1.0,21]x[-1.0,1.0,21].h5" --detail_surf_file "Output/res2net8s/22-05T15-33-44/surface_[-0.1,0.1,21]x[-0.1,0.1,21].h5" --show --out_dir Output/new_figures/test
