#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import loss.softmax as softmax
import loss.angleproto as angleproto

class LossFunction(nn.Module):

    def __init__(self,return_full_pred=False, **kwargs):
        super(LossFunction, self).__init__()
        self.return_full_pred = return_full_pred
        self.test_normalize = True

        self.softmax = softmax.LossFunction(return_full_pred=return_full_pred,**kwargs)
        self.angleproto = angleproto.LossFunction(**kwargs)

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        if self.return_full_pred:
            nlossS, prec1, output, label   = self.softmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(2))

            nlossP, _       = self.angleproto(x,None)
            return nlossS+nlossP, prec1, output, label
        else:
            nlossS, prec1   = self.softmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(2))

            nlossP, _       = self.angleproto(x,None)

            return nlossS+nlossP, prec1