"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""

import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import mpi4pytorch as mpi
import dataloader_toplevel
import evaluation
import projection as proj
import net_plotter
import plot_2D
import plot_1D
import model_loader_toplevel
import scheduler
import voxceleb.loss.amsoftmax as vc_amsoftmax
import voxceleb.loss.angleproto as vc_angleproto
import voxceleb.loss.softmaxproto as vc_softmaxproto
from utils import load_config_from_yaml, write_config_to_yaml
import datetime
from os.path import exists, commonprefix
import wandb
from h5_util import SurfaceFileHandler, extend_surface_file

def name_surface_file(args, dir_file):
    """
    Generates a filename for the surface file based on the provided arguments.
    Parameters:
    args (Namespace): A namespace object containing various attributes used to construct the filename.
        - surf_file (str): If specified, this value is returned directly.
        - xmin (float): Minimum value for the x-axis.
        - xmax (float): Maximum value for the x-axis.
        - xnum (int): Number of points along the x-axis.
        - y (bool): Flag indicating if y-axis parameters are provided.
        - ymin (float): Minimum value for the y-axis.
        - ymax (float): Maximum value for the y-axis.
        - ynum (int): Number of points along the y-axis.
        - raw_data (bool): Flag indicating if raw data (without normalization) is used.
        - data_split (int): Number of data splits.
        - split_idx (int): Index of the data split.
    dir_file (str): The directory or prefix for the filename.
    Returns:
    str: The constructed filename for the surface file with an ".h5" extension.
    """
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = "surface"


    file1, file2, file3 = args.model_file, args.model_file2, args.model_file3

    # name for xdirection
    if file2:
        # 1D linear interpolation between two models
        assert exists(file2), file2 + " does not exist!"
        if file1[:file1.rfind('/')] == file2[:file2.rfind('/')]:
            # model_file and model_file2 are under the same folder
            dir_file += file1 + '_' + file2[file2.rfind('/')+1:]
        else:
            # model_file and model_file2 are under different folders
            prefix = commonprefix([file1, file2])
            prefix = prefix[0:prefix.rfind('/')]
            dir_file += file1[:file1.rfind('/')] + '_' + file1[file1.rfind('/')+1:] + '_' + \
                       file2[len(prefix)+1: file2.rfind('/')] + '_' + file2[file2.rfind('/')+1:]
    else:
        dir_file += file1[:file1.rfind(".")]

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    surf_file +=".h5"
    return  os.path.join(args.out_dir, surf_file)

def setup_surface_file(args, surf_file, dir_file):
    """
    Sets up the surface file for storing the coordinates at which the function is evaluated.
    This function checks if the surface file already exists and contains the necessary coordinates.
    If the file exists and has the required coordinates, it skips the setup. Otherwise, it creates
    or appends to the surface file with the specified direction file and coordinates.
    Args:
        args: An object containing the following attributes:
            - xmin (float): The minimum value for the x-axis.
            - xmax (float): The maximum value for the x-axis.
            - xnum (int): The number of points along the x-axis.
            - ymin (float, optional): The minimum value for the y-axis (if applicable).
            - ymax (float, optional): The maximum value for the y-axis (if applicable).
            - ynum (int, optional): The number of points along the y-axis (if applicable).
            - y (bool): A flag indicating whether to include y-axis coordinates.
        surf_file (str): The path to the surface file.
        dir_file (str): The path to the direction file.
    Returns:
        str: The path to the surface file.
    """

    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already setted up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=round(args.xnum))
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=round(args.ynum))
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file

def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
    Calculate the loss values and accuracies of modified models in parallel using MPI reduce.
    Parameters:
    surf_file (str): Path to the HDF5 file where the surface data is stored.
    net (torch.nn.Module): The neural network model.
    w (torch.Tensor): The weights of the neural network.
    s (torch.Tensor): The states of the neural network.
    d (list): List of direction vectors for modifying the weights or states.
    dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    loss_key (str): Key for storing loss values in the HDF5 file.
    acc_key (str): Key for storing accuracy values in the HDF5 file.
    comm (MPI.Comm): MPI communicator for parallel processing.
    rank (int): Rank of the current process in the MPI communicator.
    args (argparse.Namespace): Additional arguments containing configuration options.
    Returns:
    None
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print("\n-------------------------------------------------------------------")
    print('Computing %d values for rank %d'% (len(inds), rank))
    print("-------------------------------------------------------------------")
    start_time = time.time()
    total_sync = 0.0

    # For Voxceleb the loss function is part of the model definition as it contains parameters
    # Set the loss_name as None for voxceleb, for other Models, you can set the loss_name as 'mse' or 'crossentropy'
    if args.dataset == 'cifar10':
        if args.loss_name == 'mse':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.dataset == 'voxceleb':
        criterion = None

                # if args.loss_name == 'angleproto':
                #     #criterion = vc_amsoftmax.AMSoftmax(scale=30, margin=0.35)
                #     #criterion = vc_amsoftmax.AMSoftmax(512, 1211, margin=0.35, scale=30,loss_weight=loss_weight)
                #     criterion = vc_angleproto.LossFunction(init_w=10.0, init_b=-5.0)
                # if args.loss_name == 'amsoftmax':
                #     criterion = vc_amsoftmax.AMSoftmax(512, 1211, margin=0.35, scale=30,loss_weight=loss_weight)
                # if args.loss_name == 'softmaxproto':
                #     #criterion = vc_softmaxproto.LossFunction(512, 1211, margin=0.35, scale=30,loss_weight=loss_weight)
                #     criterion = vc_softmaxproto.LossFunction(nOut=512, nClasses=1211, margin=0.35, scale=30, loss_weight=loss_weight)

    # Loop over all uncalculated loss values
    if args.use_wandb :
        if args.wandb_random_coords is not None:
            use_wandb_coords = np.random.choice(inds, args.wandb_random_coords, replace=False)
            print("wandb logging for random coords: ", use_wandb_coords)
        elif args.wandb_coords is not None:
            use_wandb_coords = [tuple(x) for x in np.array(args.wandb_coords.split(":")).reshape(-1,2).astype(float)]

    for count, ind in enumerate(inds):
        #
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        coord_tuple = tuple([c.item() for c in coord])
        if args.use_wandb:
            if args.wandb_random_coords is not None:
                use_wandb = ind in use_wandb_coords
            elif args.wandb_coords is not None:
                use_wandb = coord_tuple in use_wandb_coords
            else:
                use_wandb = True
            if use_wandb: print("logging to wandb for coord: ", coord_tuple)
        else:
            use_wandb = False

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda, dataset=args.dataset, iter_stop=args.iter_stop, use_wandb=use_wandb, coord_tuple=coord_tuple,trainfunc=args.trainfunc,eval_mode=args.eval_mode)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %s Sync: %.2f' % (rank, str(datetime.timedelta(seconds=total_time)) , total_sync))

    f.close()

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--config', default=None, type=str, help='path to the config file')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10 | voxceleb', choices=['cifar10','voxceleb'])
    parser.add_argument('--datapath', default='cifar10/data', type=str,  metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', type=str,  help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', type=str,  help='path to the testloader with random labels')
    # model parameters
    parser.add_argument('--model', default='resnet56', type=str,  help='model name')
    parser.add_argument('--model_folder', default='', type=str,  help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', type=str, help='path to the trained model file')
    parser.add_argument('--model_file2', default='', type=str, help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', type=str, help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='crossentropy', type=str, help='loss functions: crossentropy | mse')
    parser.add_argument('--eval_mode', default=False, action='store_true', help='if set, model is executed in eval mode, otherwise in train mode')
    parser.add_argument('--save_init', default=False, action='store_true', help='save the initial model parameters')

    # direction parameters
    parser.add_argument('--dir_file', default='', type=str, help='specify the name of direction file, or the path to an existing direction file')
    parser.add_argument('--dir_type', default='weights', type=str, help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', type=str, help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, type=str, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', type=str, help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', type=str, help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', type=str, help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='',type=str,  help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', type=str,  help='customize the name of surface file, could be an existing file.')
    parser.add_argument('--extendby', default=None, type=int,  help='You can extend a given surface file by extending the covered area with the given amount of points. The distance between points is infered from the given surface file')
    parser.add_argument('--increase_res', default=None, type=int,  help='You can extend the resulution of a given surface file by adding points between the already calculated points. If you want to add a point between each other point, set this to 1. Set it to 2 to add two points between each other point and so on.')



    # plot parameters
    parser.add_argument('--proj_file', default='', type=str, help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log_scale_plot', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')
    parser.add_argument('--only_plot', action='store_true', default=False, help='do not calculate surface file but use specified surface file to plot')
    parser.add_argument('--out_dir', default='Output', type=str, help='output directory for the figures')
    parser.add_argument('--zmax', default=0, type=float, help='Set Z direction. Otherwise adjust to highest value')
    parser.add_argument('--circle', action='store_true', default=False, help='cut out a circle in the plot')
    parser.add_argument('--normalise', action='store_true', default=False, help='cut out a circle in the plot')
    # logging parameters
    parser.add_argument('--use_wandb', action='store_true',        help='log to wandb')
    parser.add_argument('--wandb_project', default=None, type=str,        help='Name for project in wandb')
    parser.add_argument('--wandb_coords', default=None, type=str,        help='List of colon seperated coordinates for which we want to log the loss and acc to wandb, e.g : "0.25:0.5:0:0:-0.75:-1" for coordinates [0.25,0.5],[0,0] and [-0.75,-1]')
    parser.add_argument('--wandb_random_coords', default=None, type=int,        help='log the loss and acc for the given amount of random coordinates to wandb. E.g. if set to 3, three randomly chosen coordinates will be logged to wandb')

    parser.add_argument('--iter_stop', default=None, type=int, help='stop after this number of iterations in the dataloader')
    parser.add_argument('--debug', action='store_true',        help='run in debug mode without saving results')

    # Voxceleb Settings
    voxceleb = parser.add_argument_group('Voxceleb Arguments')
    voxceleb.add_argument('--trainfunc',      type=str,   default="angleproto",     help='Loss function')
    voxceleb.add_argument('--augment',        action='store_true',  help='augment input audio')
    voxceleb.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
    voxceleb.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')
    voxceleb.add_argument('--train_list',     type=str,   default="data/vox2/train_list.txt",  help='Train list')
    voxceleb.add_argument('--test_list',      type=str,   default="data/vox2/test_list.txt",   help='Evaluation list')
    voxceleb.add_argument('--train_path',     type=str,   default="data/vox2/voxceleb2", help='Absolute path to the train set')
    voxceleb.add_argument('--test_path',      type=str,   default="data/vox1/voxceleb_test", help='Absolute path to the test set')
    voxceleb.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
    voxceleb.add_argument('--num_eval',       type=int,   default=10,    help='number of evenly spaced segments sampled per evaluation utterance')
    voxceleb.add_argument('--eval_frames',    type=int,   default=300,    help='evaluation segment length in frames; 0 uses the complete utterance')
    voxceleb.add_argument('--nPerSpeaker',    type=int,   default=6,        help='utterances sampled per speaker in each batch item')
    voxceleb.add_argument('--max_seg_per_spk',    type=int,   default=500,        help='maximum segments sampled from one speaker per epoch')
    voxceleb.add_argument('--distributed',    action='store_true',        help='enable distributed evaluation')
    voxceleb.add_argument('--seed',    type=int,   default=10,        help='random seed')

    voxceleb.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
    voxceleb.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
    voxceleb.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
    voxceleb.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
    voxceleb.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')
    voxceleb.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
    voxceleb.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
    voxceleb.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')
    voxceleb.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')

    # parameter in orignal voxceleb - but all models use log_input = True so removed here as it defaults to true later in the code, but will be overritten if it is not given here.
    # parser.add_argument('--log_input',      action='store_true'      ,  help='Log input features')


    args = parser.parse_args()

    # add timestamp
    now = datetime.datetime.now()
    args.timestamp = now.strftime("%d-%mT%H-%M-%S")

    ## Parse YAML
    if args.config is not None:
        load_config_from_yaml(args,parser)

    torch.manual_seed(args.seed)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    if args.debug:
        args.mpi = False
        args.threads = 0

    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
              (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    # compute surface h5 file, ( if not just load it)
    if not args.only_plot:
        net = model_loader_toplevel.load(args.dataset, args.model, args.model_file, kwargs=vars(args)) #LUP - if there are weights needed in the loss, the model_loader returns them to be used in the crunch function
        w = net_plotter.get_weights(net) # initial parameters
        s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
        if args.save_init:
            # save the initial model parameters
            init_file = os.path.join(args.out_dir, 'init_model.pth')
            if rank == 0:
                torch.save(s, init_file)
                print("Initial model saved to %s" % init_file)
        if args.ngpu > 1:
            # data parallel with multiple GPUs on a single node
            net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

        #--------------------------------------------------------------------------
        # Setup the direction file and the surface file
        #--------------------------------------------------------------------------
        # if existing direction/surface file is given, no new direction/surface
        # is calculated and the direction/surface in the file is used

        # add custom directory to output dir
        time_dir = True
        if time_dir:
            if args.extendby is not None:
                extention = '_extendby(' + str(args.extendby) +")"
            elif args.increase_res is not None:
                extention = '_increase_res(' + str(args.increase_res) +")"
            else:
                extention = ''
            better_out_dir_name = args.timestamp + extention
            args.out_dir = os.path.join(args.out_dir,better_out_dir_name )

        if rank == 0:
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir,exist_ok=True)
        mpi.barrier(comm)

        #--------------------------------------------------------------------------
        # Check plotting resolution
        #--------------------------------------------------------------------------
        try:
            args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
            args.ymin, args.ymax, args.ynum = (None, None, None)
            if args.y:
                args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
                assert args.ymin and args.ymax and args.ynum, \
                'You specified some arguments for the y axis, but not all'
        except:
            raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

        if args.extendby is not None:
            if args.surf_file == "":
                raise KeyError("a surface file needs to be specified in the arguments to extend it")
            SF = SurfaceFileHandler(args.surf_file)
            if args.dir_file == '' :
                args.dir_file = SF.getDirFile()

            if rank == 0:
                new_surf = SF.extend_surface_file(dest=args.out_dir, steps=args.extendby)
                args.surf_file = new_surf
            else:
                args.surf_file = SF.extend_surface_file(dest=args.out_dir, steps=args.extendby,create=False) # all mpi processes need to have the same file name
            # if rank != 0:
            #     assert('probabily mpi error because surface file is not updated in process')
            mpi.barrier(comm)

        if args.increase_res is not None:
            if args.surf_file == "":
                raise KeyError("a surface file needs to be specified in the arguments to extend it")
            SF = SurfaceFileHandler(args.surf_file)
            if args.dir_file == '' :
                args.dir_file = SF.getDirFile()

            if rank == 0:
                new_surf = SF.increase_resolution(dest=args.out_dir, steps=args.increase_res)
                args.surf_file = new_surf
            else:
                args.surf_file = SF.increase_resolution(dest=args.out_dir, steps=args.increase_res,create=False) # all mpi processes need to have the same file name
            # if rank != 0:
            #     assert('probabily mpi error because surface file is not updated in process')
            mpi.barrier(comm)

        #--------------------------------------------------------------------------
        print('\n-------------------------------------------------------------------')
        print('setup Files and Directories')
        print('-------------------------------------------------------------------')
        # print out dir
        print("Output directory: ", args.out_dir)
        dir_file = net_plotter.name_direction_file(args) # name the direction file - if one was given, it will be used
        surf_file = name_surface_file(args, dir_file)    # create name for the surface file - if one was given, it will be used

        if rank == 0:

            setup_surface_file(args, surf_file, dir_file)

            net_plotter.setup_direction(args, dir_file, net)
            # add the used config file to output directory
            write_config_to_yaml(args)

            # wandb
            if args.use_wandb:
                if args.wandb_project is not None:
                    project_name = args.wandb_project
                else:
                    project_name = args.model_file[:args.model_file.rfind(".")]
                wandb_dir = "./.wandb"
                wandb_run = wandb.init(project=project_name,
                    config=vars(args),
                    dir=wandb_dir,
                    )

                # if args.wandb_coords is None and args.wandb_random_coords is None:
                #     print("No coordinates specified for wandb logging, the default [0,0] will be used")
                #     args.wandb_coords = "0:0"

        # wait until master has setup the direction file and surface file
        mpi.barrier(comm)

        # load directions
        d = net_plotter.load_directions(dir_file)
        # calculate the consine similarity of the two directions
        #if len(d) == 2 and rank == 0:
        #    similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        #    print('cosine similarity between x-axis and y-axis: %f' % similarity)

        #--------------------------------------------------------------------------
        # Setup dataloader
        #--------------------------------------------------------------------------
        # download CIFAR10 if it does not exit
        if rank == 0 and args.dataset == 'cifar10':
            torchvision.datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)

        mpi.barrier(comm)

        # The batch contains batch_size/nPerSpeaker speakers, each with nPerSpeaker utterances
        trainloader, testloader = dataloader_toplevel.load_dataset(args.dataset, args.datapath,
                                    args.batch_size, args.threads, args.raw_data,
                                    args.data_split, args.split_idx,
                                    args.trainloader, args.testloader, args=args)

        # logging dataloader
        next(iter(trainloader)) # call next once to initalize custom sampler
        print("\n-------------------------------------------------------------------")
        print("Dataset and dataloader Initialization")
        print("-------------------------------------------------------------------")
        num_batches = trainloader.sampler.info()
        if args.iter_stop is not None:
            print("! Iter_Stop further reduced dataset:\n    Stopping after %d batches -> %.1f%% of dataset used" % (args.iter_stop, args.iter_stop/num_batches*100))

        #--------------------------------------------------------------------------
        # Start the computation
        #--------------------------------------------------------------------------
        crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', comm, rank, args)
        # crunch(surf_file, net, loss_weight, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)
    else:
        # check if a surface file was given
        if args.surf_file == "":
            raise KeyError("a surface file needs to be specified in the arguments to plot it ( or else compute the surface file and do not set only-plot)")

        if os.path.exists(args.surf_file):
            surf_file = args.surf_file
        else:
            raise FileNotFoundError("The specified surface file does not exist")

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if ( args.plot or args.only_plot ) and rank == 0 :
        SF = SurfaceFileHandler(surf_file)
        use2D = args.y or SF.is2D()
        if use2D and args.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.zmax, args.show)
            plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_acc', args.zmax, args.show)
        elif use2D:
            plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.zmax, args.show,args.circle,args.log_scale_plot,args.normalise)
            plot_2D.plot_2d_contour(surf_file, 'train_acc', 0, 100, args.vlevel, args.zmax, args.show,args.circle,args.log_scale_plot,args.normalise)
        else:
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log_scale_plot, args.show)




''''
Der code in evaluation.evl_loss() muss auf Voxceleb angepasst werden.
Konkret muss der richtige Loss berechnet werden. Wenn man sich den Training Dataloader anschaut von Voxceleb, dann
sieht es so aus, als ob die Daten in der Form (audio, label) kommen. Der Loss wird dann mit den Labels berechnet.
Auch läd der Traininloader im moment glaube ich alle test daten auf einmal (shape 128x10x400000) oder was auch immer di 128 sind
Das netz erwartet aber nur batch_size x 400000 und gibt batch_size x 512 zurück. Was auch immer die 512 sind, wohl kaum ein Label.
Vielleicht berechnet Voxceleb nur eine intermediate Latent Representation und berechnet den Loss dann auf dieser mit einen kleinen MLP.


'''
