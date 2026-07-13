#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses,return_full_pred=False, **kwargs):
		super(LossFunction, self).__init__()
		
		self.test_normalize = True
		self.return_full_pred = return_full_pred
	    
		self.criterion  = torch.nn.CrossEntropyLoss()
		self.fc 		= nn.Linear(nOut,nClasses)

		print('Initialised Softmax Loss')

	def forward(self, x, label=None):

		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		prec1	= accuracy(x.detach(), label.detach(), topk=(1,))[0]

		if self.return_full_pred:
			return nloss, prec1, F.softmax(x, dim=1), label
		else:
			return nloss, prec1