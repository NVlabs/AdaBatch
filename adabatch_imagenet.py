'''
Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from timeit import default_timer as timer

#global timers
fp_tot = 0.
bp_tot = 0.
zero_grad_freq = 1

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--baseline-batch', default=512, type=int,
                    metavar='N', help='baseline batch size from which to compute linear learning rate scaling factor (default: 512)')
parser.add_argument( '--resize-freq', default=100, type=int,
                    metavar='N', help='frequency for resizing the batch size or decaying the learning rate (default: 100)')
parser.add_argument( '--resize-factor', default=1., type=float,
                 metavar='N', help='factor by which to increase the batch size (default: 1)')
parser.add_argument( '--lr-factor', default=1., type=float,
                    metavar='N', help='factor by which to decay the learning rate every resize-freq epochs (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--filename', default='checkpoint.pth.tar', type=str,
        help='unique filename identifier for checkpointing, if desired. (default: checkpoint.pth.tar')
parser.add_argument('--zero-grad-freq', type=int, default=1, metavar='N', help='frequency that gradients are zeroed.')
parser.add_argument('--warmup', type=int, default=0, metavar='N', help='Number of epochs for LR warmup at the beginning of training.')

best_prec1 = 0


def main():
    
    global args, best_prec1, zero_grad_freq
    args = parser.parse_args()

    zero_grad_freq = args.zero_grad_freq
    new_batch_size = args.batch_size

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler)
    print('Successfully loaded train_dataset from %s' % traindir)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print('Successfully loaded val_dataset from %s' % valdir)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    test_tot = 0.
    train_tot = 0.
    resize_tot = 0.
    torch.cuda.synchronize()
    rt_start = timer()
    
    lrstep = 0
    warmup_lr = 0

    if args.warmup != 0:
        lrstep = (((args.lr * ((args.batch_size*args.zero_grad_freq)/args.baseline_batch))) - args.lr)/(args.warmup - 1)
        warmup_lr = args.lr

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if(args.resize_factor == 1. and ((epoch + 1) % args.resize_freq == 0)):
            args.lr *= args.lr_factor
            reset_learning_rate(optimizer, args.lr)

        # train for one epoch
        
        torch.cuda.synchronize()
        train_start = timer()
        train(train_loader, model, criterion, optimizer, epoch)
        torch.cuda.synchronize()
        train_tot += timer() - train_start

        # evaluate on validation set
        torch.cuda.synchronize()
        test_start = timer()
        prec1 = validate(val_loader, model, criterion)
        torch.cuda.synchronize()
        test_tot += timer() - test_start
        print('EPOCH: %d LR: %f, test accuracy: %f' % (epoch, args.lr, prec1))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        if (epoch+1) < args.warmup:
            warmup_lr += lrstep
            args.lr = warmup_lr
            reset_learning_rate(optimizer, warmup_lr)
        
        resize_start = timer()
        if(args.resize_factor != 1. and ((epoch+1) % args.resize_freq == 0)):

            reset_cnt = 0
            if(zero_grad_freq == 1):
                new_batch_size = math.floor(new_batch_size*args.resize_factor)
                print('Increasing batch size to %d' % new_batch_size)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=new_batch_size, shuffle=(train_sampler is None), num_workers=args.workers, sampler=train_sampler)
            else:
                zero_grad_freq *= args.resize_factor
                print('Increasing batch size to %f' % math.floor(new_batch_size*zero_grad_freq))

            args.lr *= args.lr_factor
            reset_learning_rate(optimizer, args.lr)
        torch.cuda.synchronize()
        resize_tot += timer() - resize_start
        #uncomment to checkpoint this run.
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=args.filename)

    torch.cuda.synchronize()
    rt_tot = timer() - rt_start
    print('Best acc:')
    print(best_prec1)

    print('\nTotal Forward Prop. Time: %.3f s' %(fp_tot))
    print('Total Backward Prop. Time: %.3f s' %(bp_tot))
    print('Total Training Time: %.3f s' %(train_tot))
    print('Total Test Time: %.3f s' %(test_tot))
    print('Total Batch Resize Time: %.3f s' % (resize_tot))
    print('Total Running Time: %.3f s'%(rt_tot))

def train(train_loader, model, criterion, optimizer, epoch):
    global fp_tot, bp_tot, zero_grad_freq

    batches_processed = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    # Since we are accumulating gradients, we need to have a running sum of losses.
    accum_loss = 0

    end = time.time()
    print('Starting an epoch of training')
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        torch.cuda.synchronize()
        fp_start = timer()
        output = model(input_var)
        torch.cuda.synchronize()
        fp_tot += timer() - fp_start

        loss = criterion(output, target_var)/zero_grad_freq

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        if (i % zero_grad_freq == 0): 
            accum_loss = 0
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        bp_start = timer()
        loss.backward()
        torch.cuda.synchronize()
        bp_tot += timer() - bp_start
        
        if (i+1) % zero_grad_freq == 0: 
            optimizer.step()
            stored_loss = accum_loss
            batches_processed += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename+'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename+'checkpoint.pth.tar', filename+'model_best.pth.tar')


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

def reset_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    #lr = args.lr * (factor ** (epoch // freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
