#! /usr/bin/python
# -*- encoding: utf-8 -*-
import yaml
import torch
import torch.nn.functional as F
import sys
import os
import shutil

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

def find_option_type(key, parser):
        for opt in parser._get_optional_actions():
            if ('--' + key) in opt.option_strings:
                return opt.type
        raise ValueError


def load_config_from_yaml(args,parser):
    """
    Load the configuration from a yaml file
    Used instead of command line arguments
    """
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            if k in args.__dict__:
                try:
                    typ = find_option_type(k, parser)
                except ValueError:
                    print("!!!!!!!!!!!!!!")
                    print("Ignored unknown parameter in yaml: {} .\n".format(k))
                    print("!!!!!!!!!!!!!!")
                    continue
                try:
                    if str(v).lower() == "false":
                        args.__dict__[k] = False
                    elif str(v).lower() == "true":
                        args.__dict__[k] = True
                    elif str(v).lower() in ["none", "null"]:
                        args.__dict__[k] = None
                    else:
                        args.__dict__[k] = typ(v)
                except Exception as e:
                    print("Failed to parse:")
                    print(k,v)
                    print("exiting")
                    exit(0)

            else:
                sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

def write_config_to_yaml(args):
    with open(os.path.join(args.out_dir,'full_config_'+args.timestamp+'.yml'), 'w') as outfile:
        yaml.dump(vars(args), outfile)
    
    if args.config is not None:
        filename = "".join(args.config.split(os.sep)[-1].split('.')[:-1])
        dst = os.path.join(args.out_dir, filename +'_' + args.timestamp + '.yaml')
        shutil.copyfile(args.config, dst)

