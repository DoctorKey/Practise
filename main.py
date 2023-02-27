import os
import argparse
import logging
import collections
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import time
import torch.optim as optim
import torch.nn.functional as F

from models import *
import dataset
from practise import Practise_one_block, Practise_all_blocks
from finetune import end_to_end_finetune, validate

# Prune settings
parser = argparse.ArgumentParser(description='Accelerate networks by PRACTISE')
parser.add_argument('--dataset', type=str, default='imagenet_fewshot',
                    help='training dataset (default: imagenet_fewshot)')
parser.add_argument('--eval-dataset', type=str, default='imagenet',
                    help='training dataset (default: imagenet)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu_id', default='7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_sample', type=int, default=50,
                    help='number of samples for training')
parser.add_argument('--model', default='resnet34', type=str, 
                    help='model name (default: resnet34)')
parser.add_argument('--teacher', default='', type=str, metavar='PATH',
                    help='path to the pretrained teacher model (default: none)')
parser.add_argument('--save', default='results', type=str, metavar='PATH',
                    help='path to save pruned model (default: results)')
parser.add_argument('--state_dict_path', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--no-pretrained', action='store_true', default=False,
                    help='do not use pretrained weight')

parser.add_argument('--rm_blocks', default='', type=str,
                    help='names of removed blocks, split by comma')
parser.add_argument('--practise', default='', type=str,
                    help='blocks for practise', choices=['', 'one', 'all'])
parser.add_argument('--FT', default='', type=str,
                    help='method for finetuning', choices=['', 'BP', 'MiR'])

parser.add_argument('--opt', default='SGD', type=str,
                    help='opt method (default: SGD)')
parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                    help='learning rate (default: 0.02)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--batch-size', type=int, default=64,
                    help='number of batch size')
parser.add_argument('--epoch', type=int, default=2000,
                    help='number of epoch')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--seed', type=int, default=0,
                    help='seed')


def main():
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    
    args.save = os.path.join(args.save, "{}_{}_{}_{}/{}_{}".format(
        args.model, args.dataset, args.practise, args.FT, args.num_sample, args.seed))
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    LOG = logging.getLogger('main')
    time_now = datetime.now()
    logfile = os.path.join(args.save, 'log_{date:%Y-%m-%d_%H:%M:%S}.txt'.format(
        date=time_now))
    FileHandler = logging.FileHandler(logfile, mode='w')
    LOG.addHandler(FileHandler)
    import builtins as __builtin__
    builtin_print = __builtin__.print
    __builtin__.print = LOG.info

    print(args)

    if args.eval_dataset == 'cifar10':
        args.num_classes = 10
    elif args.eval_dataset == 'cifar100':
        args.num_classes = 100
    elif args.eval_dataset == 'imagenet':
        args.num_classes = 1000

    if args.dataset == 'imagenet':
        train_loader = dataset.__dict__['imagenet'](True, args.batch_size)
        args.eval_freq = 1
    else:
        args.eval_freq = 500
        assert args.seed > 0, "Please set seed"
        train_loader = dataset.__dict__[args.dataset](args.num_sample, seed=args.seed)
        if args.practise:
            metric_loader = dataset.__dict__[args.dataset](args.num_sample, seed=args.seed, train=False)
        try:
            train_loader.dataset.samples_to_file(os.path.join(args.save, "samples.txt"))
        except:
            print('Not save samples.txt')
    
    test_loader = dataset.__dict__[args.eval_dataset](False, args.test_batch_size)

    origin_model, all_blocks, origin_lat = build_teacher(
        args.model, args.num_classes, teacher=args.teacher, cuda=args.cuda
    )
    if args.rm_blocks:
        rm_blocks = args.rm_blocks.split(',')
    else:
        rm_blocks = []
    if args.practise == 'one':
        assert len(rm_blocks) == 1
        pruned_model, _ = Practise_one_block(rm_blocks[0], origin_model, origin_lat, train_loader, metric_loader, args)
    elif args.practise == 'all':
        pruned_model, rm_blocks = Practise_all_blocks(all_blocks, origin_model, origin_lat, train_loader, metric_loader, args)
    else:
        pruned_model, _, pruned_lat = build_student(
            args.model, rm_blocks, args.num_classes, 
            state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
        )
        lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
        print(f'=> latency reduction: {lat_reduction:.2f}%')

    if args.FT:
        validate(test_loader, pruned_model)
        print("=> finetune:")
        end_to_end_finetune(train_loader, test_loader, pruned_model, origin_model, args)

        save_path = 'check_point_{:%Y-%m-%d_%H:%M:%S}.tar'.format(time_now)
        save_path = os.path.join(args.save, save_path)
        check_point = {
            'state_dict': pruned_model.state_dict(),
            'rm_blocks': rm_blocks,
        }
        torch.save(check_point, save_path)



if __name__ == '__main__':
    main()
