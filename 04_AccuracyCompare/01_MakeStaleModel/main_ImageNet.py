import argparse
import datetime
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from shutil import copy
import random

from models_imagenet import *
from utils import save_checkpoint, load_checkpoint, print_log, set_log_path, save_fig

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary


def val(split, model, criterion, dataloader, epoch, args):
    val_losses_epoch = []
    acc_top1 = 0.0
    acc_top5 = 0.0
    val_size = 0

    # no grad for evaluation
    with torch.no_grad():
        # set model for evaluation
        model.eval()

        # empty cache
        torch.cuda.empty_cache()

        for i, batch in enumerate(dataloader):
            inputs = batch[0].cuda()
            targets = batch[1].cuda()

            output = model(inputs)  # outputs.data.shape= batches_num * num_class

            _, pred = torch.max(output, 1)
            val_size += targets.size(0)
            acc_top1 += (pred == targets).sum().item()

            _, rank5 = output.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(targets.view(1, -1).expand_as(rank5))

            for k in range(6):
                correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

            acc_top5 += correct_k.item()

            loss = criterion(output, targets)

            # convert to numpy
            loss_cpu = loss.detach().cpu().numpy()

            # append to all
            val_losses_epoch.append(loss_cpu)

            if i % args.print_freq_val == 0:
                print_log(
                    '[{} {}][{}/{}], loss: {:.3f} (avg: {:.3f})'.format(split, epoch, i, len(dataloader), loss_cpu,
                                                                        np.mean(val_losses_epoch)))

        acc_top1 /= val_size
        acc_top5 /= val_size

    return np.mean(val_losses_epoch), acc_top5, acc_top1


def main(args):
    # for reproduction
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    # for a little bit faster computing
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # create model
    # model = resnet50(args, pretrained=True, num_classes=1000)
    model = resnet50(args, pretrained=False, num_classes=1000)

    # resume model
    if args.resume is not None:
        model = load_checkpoint(model, args.resume)

    if args.freeze is not None:
        print_log("Freezing all params, insert AE...")
        for param in model.parameters():
            param.requires_grad = False

        for param in model.ae_encoder.parameters():
            param.requires_grad = True
        for param in model.ae_decoder.parameters():
            param.requires_grad = True

    # set device
    model = model.to(args.device)
    if args.device == torch.device("cuda"):
        print("using multi-gpu")
        model = torch.nn.DataParallel(model)
    else:
        print("non-multi gpu setting, device:", args.device)

    # if args.summary:
    #     summary(model, (3,224,224))

    # create loss
    criterion = nn.CrossEntropyLoss().to(args.device)

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    testset = torchvision.datasets.ImageNet(root='~/tensorflow_datasets/downloads/manual/', split='val', transform=transform_test)
    # testset = torchvision.datasets.ImageNet(root='../dataset', split='val', transform=transform_test)

    print(testset.__len__())
    a = []
    for idx, i in enumerate(testset.targets):
        if i > 10000:#977/987/10000
            pass
        else:
            a.append(idx)
    testset = torch.utils.data.Subset(testset, a)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    
    if args.evaluate:
        test_losses_epoch, test_metrics_top5, test_metrics = val('TEST', model, criterion, testloader, 0, args)
        print_log(' * TEST metrics: {:.5f}, loss: {:.5f}, Top-5 Accuracy: {:.5f}'.format(test_metrics, test_losses_epoch, test_metrics_top5))
        exit()

    trainset = torchvision.datasets.ImageNet(root='~/tensorflow_datasets/downloads/manual/', split='train', transform=transform_train)

    a = []
    for idx, i in enumerate(trainset.targets):
        if i > 933:
            pass
        else:
            if idx%50<37:
                a.append(idx)
    trainset = torch.utils.data.Subset(trainset, a)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    print(len(trainset), len(testset))
    # create optimizer & learning rete scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.initial_lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_period,
                                                           eta_min=args.lr_threshold*10
                                                           )

    # set device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=args.device, non_blocking=True)

    best_val_metrics = None
    # best_tr_loss = None
    best_val_loss = None

    all_tr_losses = []
    all_val_losses = []
    all_val_metrics = []
    best_epoch = 0

    ############################################################
    # start Training

    for epoch in range(args.epochs):
        model.train()

        lr = np.mean([param_group['lr'] for param_group in optimizer.param_groups])  # set current learning rate to args

        # Method 0 : using cosine&plateau scheduler

        if best_epoch + args.lr_scheduler_patience <= epoch:
            print_log('There is nearly no performance update, Reduce lr by a half!(epoch: {})'.format(epoch))

            # reduce lr by half per capturing (nearly) no performance update
            lr = lr * 0.3
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
            if args.tr_step != 1:
                copy(os.path.join(args.result, 'model_best.pth'),
                     os.path.join(args.result, 'model_best_until_{}.pth'.format(epoch)))

            best_epoch = epoch

        # Training Step changing
        if lr < args.lr_threshold:
            copy(os.path.join(args.result, 'model_best.pth'),
                 os.path.join(args.result, 'model_best_step_{}.pth'.format(args.tr_step + 1)))

            # Early stop for just fine tuning
            if args.fine_tuning:
                print_log('no more progress, end training (epoch: {})'.format(epoch))
                args.epochs = epoch
                break

            # STEP 0: transfer learning
            if args.tr_step == 0:
                print_log(
                    'Transfer learning ended, insert auto-encoder, freeze all parameters(epoch: {})'.format(epoch))

                # change scheduler
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128)

                # insert auto encoder
                args.auto_encoder = True

                # parameter freeze
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.ae_encoder.parameters():
                    param.requires_grad = True
                for param in model.ae_decoder.parameters():
                    param.requires_grad = True

                # reset leaning rate
                lr = args.initial_lr / 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # STEP 1: first converging point(unfreeze model param)
            if args.tr_step == 1:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=args.lr_threshold*10)

                # Unfreeze model
                for param in model.parameters():
                    param.requires_grad = True

                # reset leaning rate
                lr = args.initial_lr / 20
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                print_log(
                    'Reached first converging point, unfreeze all parameters(epoch: {}), lr: {}'.format(epoch, lr))

            # STEP 2: second converging point(end training)
            if args.tr_step == 2:
                print_log('no more progress, end training (epoch: {})'.format(epoch))
                args.epochs = epoch
                break

            # get to the next step
            args.tr_step += 1

        train_losses_epoch = []

        for i, batch in enumerate(trainloader):
            # set model for training
            optimizer.zero_grad()

            inputs, targets = batch[0].to(args.device, non_blocking=True), batch[1].to(args.device, non_blocking=True)
            outputs = model(inputs)
            del inputs

            loss = criterion(outputs, targets)

            # backward and update model
            loss.backward()
            optimizer.step()

            # convert to numpy
            loss_cpu = loss.detach().cpu().numpy()

            # append to all
            train_losses_epoch.append(loss_cpu)

            if i % args.print_freq == 0:
                print_log('[TRAIN {}/{}][{}/{}], loss: {:.5f} (avg: {:.5f}), lr: {:.10f}'.format(epoch, args.epochs, i,
                                                                                    len(trainloader),
                                                                                    loss_cpu,
                                                                                    np.mean(train_losses_epoch), lr))

        # append to all_tr_losses
        all_tr_losses.append(np.mean(train_losses_epoch))

        ############################################################
        # start validation
        val_losses_epoch, val_metrics_top5, val_metrics = val('TEST', model, criterion, testloader, epoch, args)

        # append to all_val
        all_val_losses.append(val_losses_epoch)
        all_val_metrics.append(val_metrics)

        # compare best eval criterion
        if best_val_metrics is None:
            best_val_metrics = all_val_metrics[-1]

        # compare best eval criterion
        # if best_tr_loss is None:
        #     best_tr_loss = all_tr_losses[-1]
        if best_val_loss is None:
            best_val_loss = all_tr_losses[-1]

        print_log(' * Epoch {} TEST metrics: {:.5f}, (best: {:.5f}), loss: {:.5f}, Top-5 acc: {:.5f}'.format(epoch, all_val_metrics[-1],
                                                                                          best_val_metrics,
                                                                                          all_val_losses[-1], val_metrics_top5))

        if epoch < args.lr_period - 1:
            scheduler.step()
        ############################################################

        # save last checkpoint
        save_checkpoint(os.path.join(args.result, 'model_last.pth'), model, epoch)

        # save best checkpoints
        if all_val_metrics[-1] >= best_val_metrics:  # same or larger
            best_val_metrics = all_val_metrics[-1]
            print_log("saving best checkpoint (epoch: {})".format(epoch))
            save_checkpoint(os.path.join(args.result, 'model_best.pth'), model, epoch)

        # remember tr best epoch for plateau lr
        # if all_tr_losses[-1] <= best_tr_loss:  # same or larger
        #     best_tr_loss = all_tr_losses[-1]
        #     best_epoch = epoch
        if all_val_losses[-1] <= best_val_loss:  # same or larger
            best_val_loss = all_val_losses[-1]
            best_epoch = epoch

    # plot
    if args.loss_plot:
        plt.plot(range(args.epochs), all_tr_losses, label='Training loss', color='royalblue')
        plt.plot(range(args.epochs), all_val_losses, label='Val loss', color='darkorange')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.grid(linestyle='--', color='lightgray')
        plt.legend(loc='upper right')
        plt.show()
        save_fig('CIFAR10_Loss', args.result)
        plt.clf()

    if args.acc_plot:
        plt.plot(range(args.epochs), all_val_metrics, label='Val Acc', color='green')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy(%)')
        plt.grid(linestyle='--', color='lightgray')
        plt.legend(loc='lower right')
        plt.show()
        save_fig('CIFAR10_acc', args.result)
        plt.clf()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--initial_lr', default=1e-1, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--lr_scheduler_patience', default=30, type=int, help='lr scheduler patience')
    parser.add_argument('--lr_threshold', default=1e-9, type=float, help='lr threshold for early stopping')
    parser.add_argument('--gpu', default=0, type=int, help='set gpu (0 ~ N), -1 for CPU')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--result', default='result', help='path to result')

    parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
    parser.add_argument('--print_freq_val', default=200, type=int, help='print frequency for validation')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--freeze', default=None, type=str, help='freeze weights or not')
    parser.add_argument('--auto_encoder', dest='auto_encoder', action='store_true', help='insert auto encoder')
    parser.add_argument('--freeze_to', default=4, type=int, help='where to freeze')   
    parser.add_argument('--loss_plot', dest='loss_plot', action='store_true', help='plot the loss graph')
    parser.add_argument('--acc_plot', dest='acc_plot', action='store_true', help='plot the accuracy graph')

    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--ae_compress_rate', default=16, type=int, help='dim compress rate of ae')
    parser.add_argument('--tr_step', default=0, type=int, help='training sequence index')
    parser.add_argument('--fine_tuning', dest='fine_tuning', action='store_true', help='for only fine tunning')
    parser.add_argument('--lr_period', default=100, type=int,
                        help='period(rate) of lr diminishing following cosine function')
    # parser.add_argument('--summary', dest='summary', action='store_true', help='show summary of model on console')
    args = parser.parse_args()

    # create result directory
    os.makedirs(args.result, exist_ok=True)

    # set log directory
    set_log_path(os.path.join(args.result, 'log_{}.txt'.format(datetime.datetime.now().strftime('%m%d_%H%M%S'))))

    # set visible devices & get device
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,2,4"
    args.device = torch.device("cuda" if args.gpu >= 0 else "cpu")

    # print command line arguments
    print_log('=' * 40)
    print_log(' ' * 14 + 'Arguments')
    for arg in sorted(vars(args)):
        print_log(arg + ':', getattr(args, arg))
    print_log('=' * 40)

    print('Device:', args.device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    # start main
    main(args)
