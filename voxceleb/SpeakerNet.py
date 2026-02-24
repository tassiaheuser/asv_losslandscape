#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, sys, random
import time, itertools, importlib
import wandb

from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler

import os


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1])
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        else:

            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)

            if self.__L__.return_full_pred:
                nloss, prec1, scores, labels = self.__L__.forward(outp, label)
                return nloss, prec1, scores, labels
            else:
                nloss, prec1 = self.__L__.forward(outp, label)
                return nloss, prec1


class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")

        self.kwargs = kwargs
        
        # if we freeze the model the parameters that are optimized by the optimizer are only the parameters from the loss function
        if kwargs["freeze"]:
            parameters = self.__model__.module.__L__.parameters()
        else:
            parameters = self.__model__.parameters()
        self.__optimizer__ = Optimizer(parameters, **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        # EER or accuracy

        tstart = time.time()

        for data, data_label in loader:

            data = data.transpose(1, 0)
            label = torch.LongTensor(data_label)

            # if self.kwargs["log_onnx"]:
            #     torch.onnx.export(self.__model__, data, "ResNetSE34L_2.onnx")
            #     exit()
            
            if not self.kwargs["cpu"]:
                data = data.cuda()
                label = label.cuda()

            self.__model__.zero_grad()

            # label = torch.FloatTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__(data, label)
                nloss.backward()
                self.__optimizer__.step()

            tloss = nloss.detach().cpu().item()
            loss += tloss
            tprec = prec1.detach().cpu().item()
            top1 += tprec
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                # sys.stdout.write("Mean loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, stepsize / telapsed))
                sys.stdout.write(" Loss {:f}, Mean loss {:f}, TEER/TAcc {:2.3f}%, Mean Acc {:2.3f}% - {:.2f} Hz ".format(tloss, loss/counter, tprec , top1/counter, stepsize / telapsed))
                sys.stdout.flush()
            
            
            if self.kwargs["wandb"]:
                wandb.log({"lr":self.__scheduler__.get_lr()[0],"loss": tloss, 'acc': tprec, "step": index+(loader.sampler.epoch-1)*len(loader)*stepsize },commit=True) # + epoch*

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter, top1 / counter)
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate for training set
    ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def eval_train_network(self, loader, verbose,iter_stop=None,return_full_pred=False):

        self.__model__.eval()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        scores_list = []
        labels_list = []
        # EER or accuracy

        tstart = time.time()

        with torch.no_grad():

            for idx,(data, data_label) in enumerate(loader):
                if iter_stop is not None and idx > iter_stop: break

                data = data.transpose(1, 0)
                label = torch.LongTensor(data_label)
                
                if not self.kwargs["cpu"]:
                    data = data.cuda()
                    label = label.cuda()

                if return_full_pred:
                    nloss, prec1, scores, labels = self.__model__(data, label)
                    scores_list += scores.detach().cpu().numpy().tolist()
                    labels_list += labels.detach().cpu().numpy().tolist()
                else:
                    nloss, prec1 = self.__model__(data, label)

                tloss = nloss.detach().cpu().item()
                loss += tloss
                tprec = prec1.detach().cpu().item()
                top1 += tprec
                counter += 1
                index += stepsize

                telapsed = time.time() - tstart
                tstart = time.time()

                if verbose:
                    sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                    # sys.stdout.write("Mean loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, stepsize / telapsed))
                    sys.stdout.write(" Loss {:f}, Mean loss {:f}, TEER/TAcc {:2.3f}%, Mean Acc {:2.3f}% - {:.2f} Hz ".format(tloss, loss/counter, tprec , top1/counter, stepsize / telapsed))
                    sys.stdout.flush()
                
        return (loss / counter, top1 / counter, scores_list, labels_list)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, logs_save_path, log_file, print_interval=100, num_eval=10, **kwargs):
        
        log = open(os.path.join(logs_save_path,"eval.log"), "a")
        sys.stdout = log
        print("")
        print("===== ===== ===== ===== ===== ===== ===== =====")
        print("===== ===== == evaluateFromList === ===== =====")
        print("")
        print("test_path:")
        print(test_path)
        print("")
        print("")
        print("test_list:")
        print(test_list[:10])
        print("")
        print("===== ===== ===== ===== ===== ===== ===== =====")
        print("===== ===== ===== ===== ===== ===== ===== =====")
        print("")
        print("")

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()
        
        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0]
            if not kwargs["cpu"]:
                inp1 = inp1.cuda()
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu() # shape 10,256
            feats[data[1][0]] = ref_feat # data[1][0] is the file name, e.g. id10270/5r0dWxy17C8/00001.wav
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)
            
            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]]
                com_feat = feats[data[2]]

                if not kwargs["cpu"]:
                    ref_feat.cuda()
                    com_feat.cuda()

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

                score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()
        sys.stdout = log_file
        return (all_scores, all_labels, all_trials, feats)
    
    def evaluateFromList_h5(self, test_loader,logs_save_path, log_file, print_interval=100, num_eval=10, distributed=False,rank=0, **kwargs):
        
        lines = []
        files = []
        feats = {}
        pred = []
        labels = []
        tstart = time.time()

        log = open(os.path.join(logs_save_path,"eval.log"), "a")
        sys.stdout = log
        print("")
        print("===== ===== ===== ===== ===== ===== ===== =====")
        print("===== ===== == evaluateFromList === ===== =====")
        print("===== ===== ===== ===== ===== ===== ===== =====")
        print("")
        print("")

        ## Extract features for every image
        for idx, (data, data_label) in enumerate(test_loader):
            if not kwargs["cpu"]:
                data.cuda()
            with torch.no_grad():
                ref_feat = self.__model__(data).detach().cpu()
                pred += ref_feat.numpy().tolist()
                labels += data_label.numpy().tolist()
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0: # 
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        # Calculate accuracy between pred and label lists
        pred_array = numpy.array(pred)
        labels_array = numpy.array(labels)
        accuracy = (pred_array.argmax(axis=1) == labels_array).mean()
        mse = numpy.sqrt(numpy.mean((pred_array - labels_array) ** 2))
        mae = numpy.mean(numpy.abs(pred_array - labels_array) )
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"\nAccuracy: {accuracy*100:.2f}%")

        return ([accuracy,mae,mse], [], [])

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def saveStates(self,path):
        os.makedirs(path, exist_ok=True)
        # save optimizer state dict
        torch.save(self.__optimizer__.state_dict(), os.path.join(path, "optimizer.pth"))
        # save scheduler state dict
        torch.save(self.__scheduler__.state_dict(), os.path.join(path, "scheduler.pth"))

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def loadOptimizer(self, path):
        if os.path.isfile(path):
            print("Loading optimizer from {}".format(path))
            self.__optimizer__.load_state_dict(torch.load(path, map_location="cuda:%d" % self.gpu))
        else:
            print("Optimizer file not found: {}".format(path))
        
    def loadParameters(self, path, freeze=False):    
# ______________________________________________________________________________________
        ### Original Voxceleb
# ______________________________________________________________________________________
        # check in loop of parameter names if it is the loss parameter, if not set require_grad to False
        # Thisway only the loss is trained and the voxelleb network is frozen
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]

        # check which parameters were not loaded by creating a list of all layers in the network
        # and removing the layer name if the parameter was loaded
        param_key_list = [key for key in self_state]

        # itersate through the loaded state dict and load parameters into model
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
                
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

            #remove loaded parameter name from list
            param_key_list.remove(name)

        if freeze:
            for name, param in  self.__model__.module.named_parameters():
                if not name.startswith("__L__."):
                    param.requires_grad = False 
        
        # print the name of the layers for which no parameters were found in inital model
        # uses the param_key_list 
        if len(param_key_list) > 0:
            print("\n! For the following layers no parameter were loaded from inital model:")
            for name in param_key_list:
                print(name)
            print("\n")
        print(f"Loaded {len(param_key_list)/len([key for key in self_state])} /% of parameters")

# # ______________________________________________________________________________________
#         ### From Loss Landscape
# # ______________________________________________________________________________________
            
#         if path:
#             assert os.path.isfile(path), f"Model file not found: {path}"
#             print("\n-------------------------------------------------------------------")
#             print(f"loading model file '{path}'")
#             print("-------------------------------------------------------------------")
            
#             # Load the checkpoint
#             checkpoint = torch.load(path,weights_only=False)
            
#             # Check if the checkpoint contains 'model_state_dict' or is a plain state dict
#             if 'model_state_dict' in checkpoint:
#                 state_dict = checkpoint['model_state_dict']
#             else:
#                 state_dict = checkpoint  # If it's just the state dict

#             # check if some parameters in the network are not in the loaded checkpoint
#             param_key_list = [key for key in self.__model__.state_dict()]
#             missing_keys = []
#             detected_issue = False
#             for key in state_dict:
#                 if key in param_key_list:
#                     param_key_list.remove(key)
#                 else:
#                     missing_keys.append(key)
#             if len(missing_keys) > 0:
#                 detected_issue = True
#                 print("Warning: The following parameters in the loaded checkpoint are not in the network: ")
#                 for key in missing_keys:
#                     print("    ",key)   
#             if len(param_key_list) > 0:
#                 detected_issue = True
#                 print("Warning: The following parameters in the network are not in the loaded checkpoint: ")
#                 for key in param_key_list:
#                     print("    ",key)
#             self.__model__.load_state_dict(state_dict,strict=False)

#             if not detected_issue:
#                 print("Model loaded successfully without any issues.")
