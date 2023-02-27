import os
import logging
import collections
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def end_to_end_finetune(train_loader, test_loader, model, t_model, args):
    # Data loading code
    end = time.time()
    if args.FT == 'MiR':
        criterion = torch.nn.MSELoss(reduction='mean').cuda()
    elif args.FT == 'BP':
        criterion = torch.nn.CrossEntropyLoss().cuda()

    # model.fc.requires_grad = False
    model.freeze_classifier()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(0.4 * args.epoch), gamma=0.1)

    # switch to train mode
    model.train()
    model.get_feat = 'pre_GAP'
    t_model.eval()
    t_model.get_feat = 'pre_GAP'

    iter_nums = 0

    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    finish = False
    while not finish:
        for batch_idx, (data, target) in enumerate(train_loader):
            iter_nums += 1
            if iter_nums > args.epoch:
                finish = True
                break
            # measure data loading time
            data = data.cuda()
            target = target.cuda()
            data_time.update(time.time() - end)
            optimizer.zero_grad()
            output, s_features = model(data)
            with torch.no_grad():
                t_output, t_features = t_model(data)
            if args.FT == 'MiR':
                loss = criterion(s_features, t_features)
            elif args.FT == 'BP':
                loss = criterion(output, target)
            losses.update(loss.data.item(), data.size(0))
            loss.backward()
            optimizer.step()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))
            lr = optimizer.param_groups[0]['lr']
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if iter_nums % args.print_freq == 0:
                print('Train: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'LR {lr}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    iter_nums, args.epoch, batch_time=batch_time, lr=lr,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            if iter_nums % args.eval_freq == 0:
                validate(test_loader, model)
                model.train()
                model.get_feat = 'pre_GAP'
            scheduler.step()
    validate(test_loader, model)


def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
