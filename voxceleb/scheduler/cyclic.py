#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-3)

	# lr_step = 'epoch'

	print('Initialised cyclic scheduler')

	return sche_fn, kwargs["lr_step"]
