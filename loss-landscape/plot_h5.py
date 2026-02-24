from plot_surface import crunch
import plot_1D
import plot_2D
import argparse
import matplotlib.pyplot as plt
from utils import load_config_from_yaml
        

#crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', comm, rank, args)


def plot_surface(surf_file, output_dir, use2D, xmin, xmax, loss_max, vmin, vmax, vlevel, proj_file=None, log=True, show=False):
    """
    Plot the surface from the given HDF5 file.
    """
    # Load the surface data
    #surf_data = crunch(surf_file)
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 7))
    
    # Plot the surface
    if use2D and proj_file is not None:
        # Load the projection data
        proj_data = crunch(proj_file)
        plot_2D.plot_2d_contour(proj_data, 'train_loss', vmin, vmax, vlevel, loss_max, show)
        plot_2D.plot_2d_contour(proj_data, 'train_acc', 0, 100, vlevel, loss_max, show)
    elif use2D:
        plot_2D.plot_2d_contour(surf_file, 'train_loss', vmin, vmax, vlevel, loss_max, show)
        plot_2D.plot_2d_contour(surf_file, 'train_acc', 0, 100, vlevel, loss_max, show)
    else:
        plot_1D.plot_1d_loss_err(surf_file, xmin, xmax, loss_max, log, show)

    # Save the figure
    fig.savefig(f"{output_dir}/surface_plot.png")
    plt.close(fig)
    
def main():
    parser = argparse.ArgumentParser(description="Plot surface from HDF5 file.")
    parser.add_argument("--surf_file", type=str, help="Path to the HDF5 surface file")
    parser.add_argument("--output_dir", type=str, help="Directory to save the plot")
    parser.add_argument("--use2D", action="store_true", help="Use 2D contour plot")
    parser.add_argument("--proj_file", type=str, help="Path to the projection file")
    parser.add_argument("--xmin", type=float, default=-1.0, help="Minimum x value")
    parser.add_argument("--xmax", type=float, default=1.0, help="Maximum x value")
    parser.add_argument("--loss_max", type=float, default=2000.0, help="Maximum loss value")
    parser.add_argument("--vmin", type=float, default=0.0, help="Minimum value for color scale")
    parser.add_argument("--vmax", type=float, default=2000.0, help="Maximum value for color scale")
    parser.add_argument("--vlevel", type=float, default=100.0, help="Contour level")
    #parser.add_argument("--zmax", type=float, default=2000.0, help="Maximum z value")
    parser.add_argument("--log", action="store_true", help="Use logarithmic scale for loss")
    parser.add_argument("--show", action="store_true", help="Show the plot")
    parser.add_argument("--config", type=str, default=None, help="Path to the configuration file")

    args = parser.parse_args()
    
    if args.config is not None:
        load_config_from_yaml(args,parser)
        
    if args.surf_file is None or args.output_dir is None:
        parser.error("Arguments 'surf_file' and 'output_dir' must be provided either in the command line or in the config file.")

    plot_surface(surf_file=args.surf_file, output_dir=args.output_dir, use2D=args.use2D,
                 xmin=args.xmin, xmax=args.xmax, loss_max=args.loss_max,
                 vmin=args.vmin, vmax=args.vmax, vlevel=args.vlevel,
                 proj_file=args.proj_file, log=args.log, show=args.show)
    
if __name__ == "__main__":
    main()