#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=max_epoch)

	# lr_step = 'epoch'

	print('Initialised constant LR scheduler')

	return sche_fn, kwargs["lr_step"]
