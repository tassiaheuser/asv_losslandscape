import os
import torch, torchvision
import cifar10.models.vgg as vgg
import cifar10.models.resnet as resnet
import cifar10.models.densenet as densenet
import voxceleb.model.ResNetSE34L as vc_resnetse34l
import voxceleb.model.ResNetSE34V2 as vc_resnetse34v2
import voxceleb.model.ResNeXt6g as ResNeXt6g
import voxceleb.model.Res2Net8s as Res2Net8s
import voxceleb.model.Res2NeXt6s8g as Res2NeXt6s8g
import voxceleb.model.RawNet3 as RawNet3
import voxceleb.model.ResNetSE34L_Bott as ResNetBott

import importlib

# map between model name and function
models = {
    'resnetse34l'         : vc_resnetse34l.MainModel,
    'resnetse34v2'        : vc_resnetse34v2.MainModel,
    'resnext6g'           : ResNeXt6g.MainModel,
    'res2net8s'           : Res2Net8s.MainModel,
    'res2next6s8g'        : Res2NeXt6s8g.MainModel,
    'rawnet3'             : RawNet3.MainModel,
    'resnet_bott'         : ResNetBott.MainModel,
}

class SpeakerNet(torch.nn.Module):
    def __init__(self, model, kwargs): # trainfunc, nPerSpeaker,
        super(SpeakerNet, self).__init__()
        self.trainfunc = kwargs['trainfunc']
        self.__S__ = models[model](**kwargs)

        LossFunction = importlib.import_module("voxceleb.loss." + kwargs['trainfunc']).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = kwargs['nPerSpeaker']

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1])
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        else:
            if not  (self.trainfunc in ['amsoftmax', 'aamsoftmax','softmax'] and self.nPerSpeaker == 1): # reshape for angleproto losses
                outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1



def load(model, model_file=None,kwargs={}):
    model = model.lower()
    if model not in models:
        raise ValueError(f"Model {model} not available for voxceleb dataset. Choose from {models.keys()}")
    net = SpeakerNet(model, kwargs )
    if 'data_parallel' in kwargs and kwargs["data_parallel"] == True: # the model is saved in data parallel mode
        net = torch.nn.DataParallel(net)

    # if model_file:
    #     assert os.path.exists(model_file), model_file + " does not exist."
    #     stored = torch.load(model_file, map_location=lambda storage, loc: storage)
    #     if 'state_dict' in stored.keys():
    #         net.load_state_dict(stored['state_dict'])
    #     else:
    #         net.load_state_dict(stored)

    # if data_parallel: # convert the model back to the single GPU version
    #     net = net.module

    # net.eval()

    #if model_file:
    #    assert os.path.isfile(model_file), f"Model file not found: {model_file}"
    #    print(f"=> loading model file '{model_file}'")
    #    checkpoint = torch.load(model_file)
    #    net.load_state_dict(checkpoint['model_state_dict'])  # Adjust depending on the checkpoint format
        
    if model_file:
        assert os.path.isfile(model_file), f"Model file not found: {model_file}"
        print("\n-------------------------------------------------------------------")
        print(f"loading model file '{model_file}'")
        print("-------------------------------------------------------------------")
        
        # Load the checkpoint
        checkpoint = torch.load(model_file,weights_only=False)
        
        # Check if the checkpoint contains 'model_state_dict' or is a plain state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # If it's just the state dict

        # check if some parameters in the network are not in the loaded checkpoint
        param_key_list = [key for key in net.state_dict()]
        missing_keys = []
        detected_issue = False
        for key in state_dict:
            if key in param_key_list:
                param_key_list.remove(key)
            else:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            detected_issue = True
            print("Warning: The following parameters in the loaded checkpoint are not in the network: ")
            for key in missing_keys:
                print("    ",key)   
        if len(param_key_list) > 0:
            detected_issue = True
            print("Warning: The following parameters in the network are not in the loaded checkpoint: ")
            for key in param_key_list:
                print("    ",key)
        net.load_state_dict(state_dict,strict=False)

        if not detected_issue:
            print("Model loaded successfully without any issues.")

        # Remove "__S__." from all keys in the state dict
        # new_state_dict = {}
        # for key, value in state_dict.items():
        #     new_key = key.replace('__S__.', '')  # Remove the "__S__." substring
        #     new_state_dict[new_key] = value
        
        # Load the modified state dict into the model
        # net.load_state_dict(new_state_dict)
        
        # print(new_state_dict.keys())
        # print("=====================================")
        # print("=====================================")
        # print("=====================================")
        # print("=====================================")
        # print("Number of keys:")
        # print(len(new_state_dict.keys()))
        # print("=====================================")
        # print("=====================================")
        # print("=====================================")
        # print("=====================================")

        
    return net