"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
import voxceleb.loss.amsoftmax as vc_amsoftmax
import voxceleb.loss.softmaxproto as vc_softmaxproto
import wandb

def eval_loss(net, criterion, loader, use_cuda=False, dataset='cifar10', iter_stop=None,use_wandb=False,coord_tuple=None,trainfunc=None,eval_mode=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
        dataset: if we use the voxceleb dataset, and the corresponding models
            the criterion or loss function has also parameters and need to be handled differently
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    # num_batch = len(loader) # not needed, iter must be called first on dataloader

    if use_cuda:
        net.cuda()
    if eval_mode:
        net.eval()
    else:
        net.train()


    if len(loader.dataset) == 0:
        raise ValueError("The dataset is empty")

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()*100

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()*100

        elif criterion is None:
            for batch_idx, (inputs, targets) in enumerate(loader):
                # if trainfunc in ['amsoftmax', 'aamsoftmax','softmax']:
                #     batch_size = torch.tensor(inputs.shape[:-1]).sum().item()
                #     inputs = inputs.squeeze()
                # else:
                #     batch_size = inputs.size(0)
                #     inputs = inputs.transpose(1, 0)
                batch_size = inputs.size(0)
                inputs = inputs.transpose(1, 0)
                total += batch_size
                targets = torch.LongTensor(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                loss, prec1 = net(inputs,targets)
                total_loss += loss.cpu().item()*batch_size
                correct += prec1.cpu().item()*batch_size
                if use_wandb:
                    wandb.log({"mean loss" + str(coord_tuple): total_loss/total, "mean acc" + str(coord_tuple): correct/total, "batch loss" + str(coord_tuple): loss.cpu().item(), "batch acc" + str(coord_tuple): prec1.cpu().item()})
                if iter_stop is not None and batch_idx >= iter_stop-1:
                    break

    if total == 0:
        total = 1
        
    return total_loss/total, correct/total


# elif isinstance(criterion, vc_amsoftmax.AMSoftmax):
        #     for batch_idx, (inputs, targets) in enumerate(loader):
        #         batch_size = inputs.size(0)
        #         total += batch_size
        #         inputs, targets = Variable(inputs[:,0,:]), Variable(targets)
        #         if use_cuda:
        #             inputs, targets = inputs.cuda(), targets.cuda()

        #         outputs = net(inputs)
        #         loss, batch_accuracy = criterion(outputs, targets)
        #         total_loss += loss.item() * batch_size
        #         correct += batch_accuracy * batch_size
        
                        # elif isinstance(criterion, vc_amsoftmax.AMSoftmax):
                        #     for batch_idx, (inputs, targets) in enumerate(loader):
                        #         batch_size = inputs.size(0)
                        #         total += batch_size
                        #         inputs = Variable(inputs[:,0,:])
                        #         targets = Variable(targets)
                        #         if use_cuda:
                        #             inputs, targets = inputs.cuda(), targets.cuda()
                                
                        #         # Forward pass to get embeddings
                        #         embeddings = net(inputs)
                                
                        #         # Compute loss
                        #         loss = criterion(embeddings, targets)
                        #         total_loss += loss.item() * batch_size
                                
                        #         # Prediction using the cosine similarity scores
                        #         outputs = F.linear(F.normalize(embeddings), F.normalize(criterion.W))
                        #         _, predicted = torch.max(outputs.data, 1)
                        #         correct += predicted.eq(targets).sum().item()
                        
                        # elif isinstance(criterion, vc_softmaxproto.LossFunction):
                        #     for batch_idx, (inputs, targets) in enumerate(loader):
                        #         batch_size = inputs.size(0)
                        #         total += batch_size
                        #         inputs = Variable(inputs[:,0,:])
                        #         targets = Variable(targets)
                        #         if use_cuda:
                        #             inputs, targets = inputs.cuda(), targets.cuda()
                                
                        #         # Forward pass to get embeddings
                        #         embeddings = net(inputs)
                        #         embeddings = embeddings.view(-1, 2, embeddings.size(-1))
                                
                        #         if use_cuda:
                        #             criterion = criterion.cuda()
                                
                        #         # Compute loss and accuracy using the softmax-prototypical loss
                        #         loss, accuracy = criterion(embeddings, targets)
                        #         total_loss += loss.item() * batch_size
                        #         correct += accuracy * batch_size  # accuracy returned as percentage
        