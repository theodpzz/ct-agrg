import torch

def build_lr_scheduler(params, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, params['scheduler']['name'])(optimizer, params['scheduler']['step_size'], params['scheduler']['gamma'])
    return lr_scheduler
