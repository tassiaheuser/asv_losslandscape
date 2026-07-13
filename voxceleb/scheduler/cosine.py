#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	# sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=test_interval, gamma=lr_decay)
	sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-7, last_epoch=-1)

	# lr_step = 'epoch'

	print('Initialised cosine scheduler')

	return sche_fn, kwargs["lr_step"]
