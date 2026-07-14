#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
import pickle
from sklearn.metrics import roc_auc_score
import numpy as np

warnings.simplefilter("ignore")

from tqdm import tqdm

import cProfile
import pstats
def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        profile_result = pstats.Stats(prof)
        profile_result.sort_stats(pstats.SortKey.TIME)
        profile_result.print_stats()
        # prof.dump_stats(datafn)
        return retval

    return wrapper


def create_parser():
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Parse arguments
    ## ===== ===== ===== ===== ===== ===== ===== =====
    parser = argparse.ArgumentParser(description = "SpeakerNet")
    parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

    ## Data loader
    parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
    parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
    parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
    parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
    parser.add_argument('--nDataLoaderThread', type=int, default=0,     help='Number of loader threads')
    parser.add_argument('--augment',        action='store_true',        help='Augment input')
    parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')
    parser.add_argument('--torch_seed',     type=int,   default=None,   help='Seed for the layer initalization. If none is given --seed is used')
    parser.add_argument('--hdf5',           action='store_true',        help='Read training files from hdf5')

    ## Training details
    parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
    parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
    parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')

    ## Optimizer
    parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
    parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
    parser.add_argument('--lr_step',        type=str,   default="epoch", help='Learning rate scheduler update step, epoch or iteration')
    parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
    parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
    parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')
    parser.add_argument('--load_optimizer', type=str,   default=None,   help='Load optimizer state from model file')

    parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
    parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
    parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
    parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
    parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
    parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

    ## Evaluation parameters
    parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
    parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
    parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

    ## Load and save
    parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
    parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')
    parser.add_argument('--log_file',       default=False, action="store_true", help='Log print statements of script to file in save_path')
    parser.add_argument('--wandb',                      default=False, action="store_true", help='Log to WandB')
    parser.add_argument('--wandb_resume',   type=str,   default=None,  action="store",      help='Resume a run in WandB. Specify the Run ID')
    parser.add_argument('--wandb_project',  type=str,   default=None,  action="store",      help='Name for the WandB project')
    parser.add_argument('--log_onnx',  default=False, action="store_true", help='save model to a onnx file')


    ## Training and test data
    parser.add_argument('--train_list',     type=str,   default="data/vox2/train_list.txt",  help='Train list')
    parser.add_argument('--test_list',      type=str,   default="data/vox2/test_list.txt",   help='Evaluation list')
    parser.add_argument('--train_path',     type=str,   default="data/vox2/voxceleb2", help='Absolute path to the train set')
    parser.add_argument('--test_path',      type=str,   default="data/vox1/voxceleb1", help='Absolute path to the test set')
    parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
    parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')
    parser.add_argument('--balanced',       type=float,   default=None,     help='generate a more balanced dataset, 0.0 to 1.0, None for no balancing, 1.0 for perfect balance')

    ## Model definition
    parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
    parser.add_argument('--log_input',      action='store_true'      ,  help='Log input features')
    parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
    parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
    parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
    parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')

    ## For test only
    parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
    parser.add_argument('--test_data',                   action='store_true', help='Run through dataloader and look for errors')
    parser.add_argument('--eval_train',     dest='eval_train', action='store_true', help='Evaluate the training set')
    parser.add_argument('--return_full_pred', dest='return_full_pred', action='store_true', help='Return full prediction in the evaluation')
    parser.add_argument('--iter_stop', type=int, default=None, help='Batch size for evaluation')
    ## Distributed and mixed precision training
    parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
    parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
    parser.add_argument('--freeze', default=False, action='store_true', help='Freeze the model')
    parser.add_argument('--cpu', default=False, action='store_true', help='Run on CPU')

    return parser


## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

def load_config(args, parser):
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            if typ is None:
                typ = bool
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):
    """
    Main worker function for training and evaluating the SpeakerNet model.
    This function handles the initialization of the model, data loaders, and training process.
    It supports both distributed and single GPU training. Additionally, it manages the evaluation
    of the model and saving of training parameters and results.
    :param gpu: The GPU id to use for training.
    :type gpu: int
    :param ngpus_per_node: The number of GPUs available per node.
    :type ngpus_per_node: int
    :param args: Command line arguments and configurations for training.
    :type args: argparse.Namespace
    """

    args.gpu = gpu

    if args.torch_seed:
        torch.manual_seed(args.torch_seed)
        torch.cuda.manual_seed(args.torch_seed)
        torch.cuda.manual_seed_all(args.torch_seed)
    else:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ## Load models
    s = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        if args.cpu:
            s = WrappedModel(s)
        else:
            s = WrappedModel(s).cuda(args.gpu)

    it = 1
    eers = [100]

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")

    ## Initialise trainer and data loader
    if args.hdf5:
        print("Using h5py dataloader")
        train_dataset = train_dataset_loader_h5py(**vars(args))
        train_sampler = train_dataset_sampler_h5py(train_dataset, **vars(args))
    else:
        train_dataset = train_dataset_loader(**vars(args))
        train_sampler = train_dataset_sampler(train_dataset, **vars(args))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    model_dirs = glob.glob('%s/model0*_states'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model, args.freeze)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        # trainer.loadOptimizer(args.load_optimizer)
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    # for ii in range(1,it):
        # trainer.__scheduler__.step() # start in 79

    ## Evaluation code - must run on single GPU
    if args.eval == True:

        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        print('Total parameters: ',pytorch_total_params)
        print('Test list',args.test_list)

        sc, lab, _, feat = trainer.evaluateFromList(**vars(args))

        if args.gpu == 0:

            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)


            scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))
            scorefile.flush()
            scorefile.close()


        with open(os.path.join(args.result_save_path,'features.pkl'), 'wb') as f:
            pickle.dump(feat, f)
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), "MinDCF {:2.5f}".format(mindcf))
        return

    if args.eval_train:
        # if args.return_full_pred:
        #     loss, traineer,sc,lab  = trainer.eval_train_network(train_loader,verbose=(args.gpu == 0),iter_stop=args.iter_stop,return_full_pred=args.return_full_pred)

        #     roc_auc_macro = roc_auc_score(np.array(lab), np.array(sc), multi_class='ovr', average='macro')
        #     roc_auc_micro = roc_auc_score(np.array(lab), np.array(sc), multi_class='ovr', average='micro')
        #     if args.gpu == 0:
        #         scorefile   = open(args.result_save_path+"/scores_full_pred.txt", "a+")
        #         scorefile.write("Train epoch_loss {:2.4f},Train epoch_acc {:2.5f}, Train AUC Macro {:2.5f}, Train AUC Micro {:2.5f}\n".format(loss,traineer, roc_auc_macro, roc_auc_micro))
        #         scorefile.flush()
        #         scorefile.close()

        args.return_full_pred = False
        s = SpeakerNet(**vars(args))
        if args.cpu:
            s = WrappedModel(s)
        else:
            s = WrappedModel(s).cuda(args.gpu)
            trainer     = ModelTrainer(s, **vars(args))

        if(args.initial_model != ""):
            trainer.loadParameters(args.initial_model, args.freeze)

        loss, traineer,sc,lab = trainer.eval_train_network(train_loader,verbose=(args.gpu == 0),iter_stop=None)
        roc_auc_macro, roc_auc_micro = 0.0, 0.0
        if args.gpu == 0:
            scorefile   = open(args.result_save_path+"/scores_full_dataset.txt", "a+")
            scorefile.write("Train epoch_loss {:2.4f},Train epoch_acc {:2.5f}\n".format(loss,traineer))
            scorefile.flush()
            scorefile.close()
        return


    if args.test_data:
        print("testing data loader")
        start = time.time()
        datas = []
        for data, label in train_loader:
            d,l = data.shape, label.shape
            # datas.append(data)
            # temp = torch.cat(datas)
            # print("mean: ", round(temp.mean().item(),3), " std: ", round(temp.std().item(),3), " max: ", round(temp.max().item(),3), " min: ", round(temp.min().item(),3))
            end = time.time()
            print("data shape: ", d, " label shape: ", l)
            print("time taken for batch: ", end-start, " per file: ", (end-start)/args.batch_size)
            start = time.time()
            # return
        return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)

    # wandb
    if args.wandb:
        if args.wandb_project is not None:
            project_name = args.wandb_project
        else:
            project_name = args.model
        wandb_dir = "./.wandb"
        if args.wandb_resume:
            wandb_run = wandb.init(project=project_name,
                config=args,
                resume="must",
                id=args.wandb_resume,
                dir=wandb_dir,
                )
        else:
            wandb_run = wandb.init(project=project_name,
                config=args,
                dir=wandb_dir,
                )

    # Save random init model:
    trainer.saveParameters(args.model_save_path+"/model%09d.model"%0)

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.wandb:
            wandb.log({"epoch_loss": loss, "epoch_acc": traineer, "epoch": it}) # "lr": max(clr),

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}\n".format(it, traineer, loss, max(clr)))
            scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, max(clr)))

        if it % args.test_interval == 0:

            sc, lab, _, feat = trainer.evaluateFromList(**vars(args))

            # # new
            # scorefile.write("Epoch {:d}, MAE {:2.4f}, MSE {:2.5f}\n".format(it, sc[0], sc[1], sc[2] ))
            # if args.wandb:
            #  wandb.log({"test_acc": traineer, "test_mae": it, "test_mae": loss, }) # "lr": max(clr),

            if args.gpu == 0:

                result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                # VEER: Validation Equal Error Rate
                print('---------------------------------------------\n',
                    time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf),
                    '\n---------------------------------------------\n')
                scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

                if args.wandb:
                    wandb.log({"VEER": result[1], "MinDCF": mindcf,"epoch": it})

                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)
                trainer.saveStates(args.model_save_path +"/model%09d_states"%it)

                with open(args.model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
                    eerfile.write('{:2.4f}'.format(result[1]))

                scorefile.flush()

    if args.gpu == 0:
        scorefile.close()

## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====

def main():

    parser = create_parser()
    args = parser.parse_args()

    if args.config is not None:
        load_config(args, parser)

    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.logs_save_path      = args.save_path+"/logs"
    args.feat_save_path      = ""

    if args.log_file:
        file_path =  os.path.join(args.logs_save_path,"train.log")
        args.log_file = open(file_path, "a")
        print("change stdout to file: ",file_path)
        sys.stdout = args.log_file
    else:
        args.log_file = sys.__stdout__


    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)
    os.makedirs(args.logs_save_path, exist_ok=True)


    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()