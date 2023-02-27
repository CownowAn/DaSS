import os
import math
import shutil

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import random


def set_gpu(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return device


def set_seed(manual_seed: int = 0) -> None:
    # Determinism
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(manual_seed)


def l2norm(x):
    norm2 = torch.norm(x, 2, dim=1, keepdim=True)
    x = torch.div(x, norm2)
    return x


class InfIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            del self.iterator
            self.iterator = iter(self.iterable)
            return next(self.iterator)


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

#----- Criterion -----#
def label_smooth(target, n_classes: int, label_smoothing=0.1):
	# convert to one-hot
	batch_size = target.size(0)
	target = torch.unsqueeze(target, 1)
	soft_target = torch.zeros((batch_size, n_classes), device=target.device)
	soft_target.scatter_(1, target, 1)
	# label smoothing
	soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
	return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target):
	logsoftmax = nn.LogSoftmax(dim=1)
	return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
	soft_target = label_smooth(target, pred.size(1), label_smoothing)
	return cross_entropy_loss_with_soft_target(pred, soft_target)


def get_criterion(mixup_alpha, label_smoothing, is_train):
    if is_train:
        if isinstance(mixup_alpha, float):
            raise NotImplementedError
            return cross_entropy_loss_with_soft_target
        elif label_smoothing > 0:
            return lambda pred, target: cross_entropy_with_label_smoothing(pred, target, label_smoothing)
        else:
            return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()


def get_scheduler(scheduler_name, opt, train_steps, milestones=[0.4, 0.7, 0.9], gamma=0.3):
    if scheduler_name == "step":
        milestones = [int(train_steps * v) for v in milestones]
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, train_steps)
    else:
        raise NotImplementedError
    return scheduler


def get_optimizer(opt_type, lr, params, weight_decay=5e-4, momentum=0.9, nesterov=False):
    if opt_type == "adam":
        opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        opt = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    elif opt_type == "rmsprop":
        opt = optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise NotImplementedError
    return opt


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        

def save_model(checkpoint, save_path, is_best, model_name=None):
    os.makedirs(save_path, exist_ok=True)

    if model_name is None:
        model_name = 'checkpoint.pth.tar'

    torch.save(checkpoint, os.path.join(save_path, model_name))

    if is_best:
        shutil.copy(os.path.join(save_path, model_name), 
            os.path.join(save_path, 'model_best.pth.tar'))


class Logger:
    def __init__(
        self,
        exp_name,
        log_dir=None,
        exp_suffix="",
        write_textfile=True,
        use_wandb=False,
        wandb_project_name=None,
        entity='username'
    ):

        self.log_dir = log_dir
        self.write_textfile = write_textfile
        self.use_wandb = use_wandb

        self.logs_for_save = {}
        self.logs = {}

        if self.write_textfile:
            self.f = open(os.path.join(log_dir, 'logs.txt'), 'w')

        if self.use_wandb:
            exp_suffix = "_".join(exp_suffix.split("/")[:-1])
            self.run = wandb.init(
                config=wandb.config,
                entity=entity,
                project=wandb_project_name, 
                name=exp_name + "_" + exp_suffix, 
                group=exp_name,
                reinit=True)\
            
    def update_artifact(self, ds_name, table, step):
        if self.use_wandb:
            wandb.log({f'{ds_name} acc chart': 
                    wandb.plot.scatter(table, 'y_true', 'y_pred', f'{ds_name} Acc.')}, step=step)

    def update_config(self, v, is_args=False):
        if is_args:
            self.logs_for_save.update({'args': v})
        else:
            self.logs_for_save.update(v)
        if self.use_wandb:
            wandb.config.update(v, allow_val_change=True)

    def write_summary(self, k, v):
        if self.use_wandb:
            wandb.run.summary[k] = v

    def write_log_nohead(self, element, step):
        log_str = f"{step} | "
        log_dict = {}
        for key, val in element.items():
            if not key in self.logs_for_save:
                self.logs_for_save[key] =  []
            self.logs_for_save[key].append(val)
            log_str += f'{key} {val} | '
            log_dict[f'{key}'] = val
        
        if self.write_textfile:
            self.f.write(log_str+'\n')
            self.f.flush()

        if self.use_wandb:
            wandb.log(log_dict, step=step)
            
    def write_log(self, element, step, img_dict=None, tbl_dict=None):
        log_str = f"{step} | "
        log_dict = {}
        for head, keys  in element.items():
            for k in keys:
                v = self.logs[k].avg
                if not k in self.logs_for_save:
                    self.logs_for_save[k] = []
                self.logs_for_save[k].append(v)
                log_str += f'{k} {v}| '
                log_dict[f'{head}/{k}'] = v

        if self.write_textfile:
            self.f.write(log_str+'\n')
            self.f.flush()
        
        if img_dict is not None:
            log_dict.update(img_dict)
        
        if tbl_dict is not None:
            log_dict.update(tbl_dict)
            
        if self.use_wandb:
            wandb.log(log_dict, step=step)

    def save_log(self, name=None):
        name = 'logs.pt' if name is None else name
        torch.save(self.logs_for_save, os.path.join(self.log_dir, name))

    def update(self, key, v, n=1):
        if not key in self.logs:
            self.logs[key] = AverageMeter()
        self.logs[key].update(v, n)

    def reset(self, keys=None, except_keys=[]):
        if keys is not None:
            if isinstance(keys, list):
                for key in keys:
                    self.logs[key] =  AverageMeter()
            else:
                self.logs[keys] = AverageMeter()
        else:
            for key in self.logs.keys():
                if not key in except_keys:
                    self.logs[key] = AverageMeter()

    def avg(self, keys=None, except_keys=[]):
        if keys is not None:
            if isinstance(keys, list):
                return {key: self.logs[key].avg for key in keys if key in self.logs.keys()}
            else:
                return self.logs[keys].avg
        else:
            avg_dict = {}
            for key in self.logs.keys():
                if not key in except_keys:
                    avg_dict[key] =  self.logs[key].avg
            return avg_dict 


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""

	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def str2bool(v):
    return v.lower() in ['t', 'true', True]


def str2int(v):
    if v == 'random': return v
    else: return int(v)
