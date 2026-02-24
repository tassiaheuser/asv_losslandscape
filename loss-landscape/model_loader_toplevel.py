import os
import cifar10.model_loader
import voxceleb.model_loader

def load(dataset, model, model_file, data_parallel=False,kwargs={}): # changed model_name to model, changed  kwargs={} to **kwargs
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model, model_file, data_parallel) # changed model_name to model
    elif dataset == 'voxceleb':
        net = voxceleb.model_loader.load(model=model,model_file=model_file,kwargs=kwargs)
    return net
    
    # if dataset == 'voxceleb':
    #     net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    # return net