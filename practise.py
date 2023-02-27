import os
import gc
import argparse
import logging
import collections
from datetime import datetime
import time
import numpy as np
import torch

from models import *
import dataset
from finetune import AverageMeter, validate, accuracy
from compute_flops import compute_MACs_params
from models.AdaptorWarp import AdaptorWarp


def Practise_one_block(rm_block, origin_model, origin_lat, train_loader, metric_loader, args):
    gc.collect()
    torch.cuda.empty_cache()

    pruned_model, _, pruned_lat = build_student(
        args.model, [rm_block], args.num_classes, 
        state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
    )
    lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
    print(f'=> latency reduction: {lat_reduction:.2f}%')

    print("Metric w/o Recovering:")
    metric(metric_loader, pruned_model, origin_model)

    pruned_model_adaptor = AdaptorWarp(pruned_model)

    start_time = time.time()
    Practise_recover(train_loader, origin_model, pruned_model_adaptor, [rm_block], args)
    print("Total time: {:.3f}s".format(time.time() - start_time))

    print("Metric w/ Recovering:")
    recoverability = metric(metric_loader, pruned_model_adaptor, origin_model)
    pruned_model_adaptor.remove_all_preconv()
    pruned_model_adaptor.remove_all_afterconv()

    score = recoverability / lat_reduction
    print(f"{rm_block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
    return pruned_model, (recoverability, lat_reduction, score)

def Practise_all_blocks(rm_blocks, origin_model, origin_lat, train_loader, metric_loader, args):
    recoverabilities = dict()
    for rm_block in rm_blocks:
        _, results = Practise_one_block(rm_block, origin_model, origin_lat, train_loader, metric_loader, args)
        recoverabilities[rm_block] = results

    print('-' * 50)
    sort_list = []
    for block in recoverabilities:
        recoverability, lat_reduction, score = recoverabilities[block]
        print(f"{block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
        sort_list.append([score, block])
    print('-' * 50)
    print('=> sorted')
    sort_list.sort()
    for score, block in sort_list:
        print(f"{block} -> {score:.4f}")
    print('-' * 50)
    print(f'=> scores of {args.model} (#data:{args.num_sample}, seed={args.seed})')
    print('Please use this seed to recover the model!')
    print('-' * 50)

    drop_blocks = []
    if args.rm_blocks.isdigit():
        for i in range(int(args.rm_blocks)):
            drop_blocks.append(sort_list[i][1])
    pruned_model, _, pruned_lat = build_student(
        args.model, drop_blocks, args.num_classes, 
        state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
    )
    lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
    print(f'=> latency reduction: {lat_reduction:.2f}%')
    return pruned_model, drop_blocks

def insert_one_block_adaptors_for_mobilenet(origin_model, prune_model, rm_block, params, args):
    origin_named_modules = dict(origin_model.named_modules())
    pruned_named_modules = dict(prune_model.model.named_modules())

    print('-' * 50)
    print('=> {}'.format(rm_block))
    has_rm_count = 0
    rm_channel = origin_named_modules[rm_block].out_channels
    key_items = rm_block.split('.')
    block_id = int(key_items[1])

    pre_block_id = block_id-has_rm_count-1
    while pre_block_id > 0:
        pruned_module = pruned_named_modules[f'features.{pre_block_id}']
        if rm_channel != pruned_module.out_channels:
            break
        last_conv_key = 'features.{}.conv.2'.format(pre_block_id)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        params.append({'params': conv.parameters()})
        pre_block_id -= 1
        # break

    after_block_id = block_id - has_rm_count
    while after_block_id < 18:
        pruned_module = pruned_named_modules[f'features.{after_block_id}']
        after_conv_key = 'features.{}.conv.0.0'.format(after_block_id)
        conv = prune_model.add_preconv_for_conv(after_conv_key)
        params.append({'params': conv.parameters()})
        if rm_channel != pruned_module.out_channels:
            break
        after_block_id += 1
        # break

    has_rm_count += 1


   

def insert_one_block_adaptors_for_resnet(prune_model, rm_block, params, args):
    pruned_named_modules = dict(prune_model.model.named_modules())
    if 'layer1.0.conv2' in pruned_named_modules:
        last_conv_in_block = 'conv2'
    elif 'layer1.0.conv3' in pruned_named_modules:
        last_conv_in_block = 'conv3'
    else:
        raise ValueError("This is not a ResNet.")

    print('-' * 50)
    print('=> {}'.format(rm_block))
    layer, block = rm_block.split('.')
    rm_block_id = int(block)
    assert rm_block_id >= 1

    downsample = '{}.0.downsample.0'.format(layer)
    if downsample in pruned_named_modules:
        conv = prune_model.add_afterconv_for_conv(downsample)
        if conv is not None:
            params.append({'params': conv.parameters()})

    for origin_block_num in range(rm_block_id):
        last_conv_key = '{}.{}.{}'.format(layer, origin_block_num, last_conv_in_block)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        if conv is not None:
            params.append({'params': conv.parameters()})

    for origin_block_num in range(rm_block_id+1, 100):
        pruned_output_key = '{}.{}.conv1'.format(layer, origin_block_num-1)
        if pruned_output_key not in pruned_named_modules:
            break
        conv = prune_model.add_preconv_for_conv(pruned_output_key)
        if conv is not None:
            params.append({'params': conv.parameters()})

    # next stage's conv1
    next_layer_conv1 = 'layer{}.0.conv1'.format(int(layer[-1]) + 1)
    if next_layer_conv1 in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_conv1)
        if conv is not None:
            params.append({'params': conv.parameters()})

    # next stage's downsample
    next_layer_downsample = 'layer{}.0.downsample.0'.format(int(layer[-1]) + 1)
    if next_layer_downsample in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_downsample)
        if conv is not None:
            params.append({'params': conv.parameters()})


def insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks, params, args):
    rm_blocks_for_prune = []
    rm_blocks.sort()
    rm_count = [0, 0, 0, 0]
    for block in rm_blocks:
        layer, i = block.split('.')
        l_id = int(layer[-1])
        b_id = int(i)
        prune_b_id = b_id - rm_count[l_id-1]
        rm_count[l_id-1] += 1
        rm_block_prune = f'{layer}.{prune_b_id}'
        rm_blocks_for_prune.append(rm_block_prune)
    for rm_block in rm_blocks_for_prune:
        insert_one_block_adaptors_for_resnet(prune_model, rm_block, params, args)


def Practise_recover(train_loader, origin_model, prune_model, rm_blocks, args):
    params = []

    if 'mobilenet' in args.model:
        assert len(rm_blocks) == 1
        insert_one_block_adaptors_for_mobilenet(origin_model, prune_model, rm_blocks[0], params, args)
    else:
        insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks, params, args)

    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("{} not found".format(args.opt))

    recover_time = time.time()
    train(train_loader, optimizer, prune_model, origin_model, args)
    print("compute recoverability {} takes {}s".format(rm_blocks, time.time() - recover_time))


def train(train_loader, optimizer, model, origin_model, args):
    # Data loading code
    end = time.time()
    criterion = torch.nn.MSELoss(reduction='mean')

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    model.cuda()
    model.eval()
    model.get_feat = 'pre_GAP'
    origin_model.get_feat = 'pre_GAP'

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(0.4 * args.epoch), gamma=0.1)

    torch.cuda.empty_cache()
    iter_nums = 0
    finish = False
    while not finish:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_loader):
            iter_nums += 1
            if iter_nums > args.epoch:
                finish = True
                break
            # measure data loading time
            data_time.update(time.time() - end)
            data = data.cuda()
            with torch.no_grad():
                t_output, t_features = origin_model(data)
            optimizer.zero_grad()
            output, s_features = model(data)
            loss = criterion(s_features, t_features)
            losses.update(loss.data.item(), data.size(0))
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if iter_nums % 50 == 0:
                print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
                   iter_nums, args.epoch, batch_time=batch_time,
                   data_time=data_time, losses=losses))
            scheduler.step()


def metric(metric_loader, model, origin_model):
    criterion = torch.nn.MSELoss(reduction='mean')

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    origin_model.get_feat = 'pre_GAP'
    model.cuda()
    model.eval()
    model.get_feat = 'pre_GAP'
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (data, _) in enumerate(metric_loader):
        with torch.no_grad():
            data = data.cuda()
            data_time.update(time.time() - end)
            t_output, t_features = origin_model(data)
            s_output, s_features = model(data)
            loss = criterion(s_features, t_features)

        losses.update(loss.data.item(), data.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            print('Metric: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
                i, len(metric_loader), batch_time=batch_time,
                data_time=data_time, losses=losses))

    print(' * Metric Loss {loss.avg:.4f}'.format(loss=losses))
    return losses.avg




if __name__ == '__main__':
    main()
