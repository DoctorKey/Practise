import os
import argparse
import numpy as np
import time, gc

import torch
import torchvision

from models import *

parser = argparse.ArgumentParser(description='PyTorch test speed')
parser.add_argument('--model', default='resnet34', type=str,
                    help='the name of model')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--batch-size', type=int, default=64,
                    help='number of batch size')
parser.add_argument('--test-time', type=int, default=500,
                    help='number of test times')
parser.add_argument('--cudnn-benchmark', action='store_true', default=False,
                    help='enable cudnn benchmark')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='use cpu')
parser.add_argument('--amp', action='store_true', default=False,
                    help='enable amp in PyTorch')
parser.add_argument('--state_dict', default='', type=str,
                    help='state_dict for pruned resnet')
parser.add_argument('--imgsize', type=int, default=224,
                    help='the size of testing img')
parser.add_argument('--rm_blocks', default='', type=str,
                    help='names of removed blocks, split by comma')

# 224, 192, 160, 128, 96, 64, 32

# Timing utilities
start_time = None


def main():
    global args
    args = parser.parse_args()
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    if not args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print('enable cudnn benchmark')

    pruned_model, origin_model, rm_blocks = build_models(
        args.model, args.rm_blocks.split(','), 1000, 
    )


    data = torch.randn(args.batch_size, 3, args.imgsize, args.imgsize)
    if not args.cpu:
        data = data.cuda()
        origin_model = origin_model.cuda()
        pruned_model = pruned_model.cuda()

    origin_model.eval()
    pruned_model.eval()

    print('data: {}'.format(data.shape))
    print('=> use amp? {}'.format(args.amp))

    t = eval_speed(origin_model, data, amp=args.amp, test_time=args.test_time)
    print('baseline: {} second'.format(t))

    t = eval_speed(pruned_model, data, amp=args.amp, test_time=args.test_time)
    print('test: {} second'.format(t))

    t = eval_speed(origin_model, data, amp=args.amp, test_time=args.test_time)
    print('baseline: {} second'.format(t))

    t = eval_speed(pruned_model, data, amp=args.amp, test_time=args.test_time)
    print('test: {} second'.format(t))



def eval_speed(model, data, amp=False, test_time=500):
    print('=> testing latency. Please wait.')
    with torch.no_grad():
        output = model(data)
    if amp:
        start_timer()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i in range(test_time):
                    output = model(data)
        total_time = end_timer()
        each_time = total_time / test_time
    else:
        start_timer()
        with torch.no_grad():
            for i in range(test_time):
                output = model(data)
        total_time = end_timer()
        each_time = total_time / test_time
    return each_time

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    #torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer():
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    main()

    