"""
    Plot the optimization path in the space spanned by principal directions.
"""

import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
import model_loader_toplevel
import net_plotter
from projection import setup_PCA_directions, project_trajectory
import plot_2D
from utils import load_config_from_yaml, write_config_to_yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot optimization trajectory')
    parser.add_argument('--config', type=str, default=None, help='config yaml')
    parser.add_argument('--dataset', type=str, default='voxceleb', help='dataset')
    parser.add_argument('--model', type=str, default='resnet56', help='trained models')
    parser.add_argument('--model_folder', type=str, default='', help='folders for models to be projected')
    parser.add_argument('--dir_type', type=str, default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--ignore', type=str, default='', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--prefix', type=str, default='model_', help='prefix for checkpoint files')
    parser.add_argument('--suffix', type=str, default='.t7', help='suffix for checkpoint files')
    parser.add_argument('--start_epoch',   default=0, type=int, help='min index of epochs')
    parser.add_argument('--max_epoch', default=300, type=int, help='max number of epochs')
    parser.add_argument('--save_epoch',  default=1, type=int, help='save models every few epochs')
    parser.add_argument('--dir_file', type=str, default=None, help='load the direction file for projection')

    # voxceleb args
    voxceleb = parser.add_argument_group('Voxceleb Arguments')
    voxceleb.add_argument('--trainfunc',      type=str,   default="angleproto",     help='Loss function')
    voxceleb.add_argument('--nPerSpeaker',    type=int,   default=6,        help='utterances sampled per speaker in each batch item')
    voxceleb.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')
    voxceleb.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
    voxceleb.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
    voxceleb.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
    voxceleb.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions') # 0.2

    args = parser.parse_args()

    if args.config is not None:
        load_config_from_yaml(args,parser)

    #--------------------------------------------------------------------------
    # load the final model
    #--------------------------------------------------------------------------
    # last_model_file = args.model_folder + '/' + args.prefix + str(args.max_epoch) + args.suffix
    last_model_file = os.path.join(args.model_folder,"model%09d.model"%args.max_epoch)
    net = model_loader_toplevel.load(args.dataset, args.model, model_file=last_model_file,kwargs=vars(args))
    w = net_plotter.get_weights(net)
    s = net.state_dict()

    #--------------------------------------------------------------------------
    # collect models to be projected
    #--------------------------------------------------------------------------
    model_files = []
    for epoch in range(args.start_epoch, args.max_epoch + args.save_epoch, args.save_epoch):
        # model_file = args.model_folder + '/' + args.prefix + str(epoch) + args.suffix
        model_file = os.path.join(args.model_folder,"model%09d.model"%epoch)
        assert os.path.exists(model_file), 'model %s does not exist' % model_file
        model_files.append(model_file)

    #--------------------------------------------------------------------------
    # load or create projection directions
    #--------------------------------------------------------------------------
    if args.dir_file:
        dir_file = args.dir_file
    else:
        dir_file = setup_PCA_directions(args, model_files, w, s, **vars(args))
    del args.dir_file  # remove dir_file from args to avoid passing it to project_trajectory
    del args.config  # remove config from args to avoid passing it to project_trajectory
    #--------------------------------------------------------------------------
    # projection trajectory to given directions
    #--------------------------------------------------------------------------
    proj_file = project_trajectory(dir_file, w, s,
                                model_files, proj_method='cos', **vars(args))
    # removed  args.dataset,
    plot_2D.plot_trajectory(proj_file, dir_file)
