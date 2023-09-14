import imp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from data import *
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from loss.CEL import CEL
# from model_lg_base import model 
# from model_resnet_new import model 
# from my_model import model 
# from model3 import model
# from model5 import model
# from baseline_model5 import model
# from model6_0109 import model
# from model8_gate import model
# from resnet50_0113 import model
# from sample_0114 import model
# from model5_resnet50_sample import model
# from resnet50_0123_new import model     
# from model import model
from model import model
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# Oof
import eval as eval_script
import pdb
# from utils import crf_proc
import skimage.io as io

from tensorboardX import SummaryWriter

def seed_torch(seed=666):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
seed_torch()
torch.autograd.set_detect_anomaly(True)
class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net, criterion):
        super().__init__()

        self.net = net
        self.criterion = criterion 
    
    def forward(self, images, word_idx, targets, masks, num_crowds):
        preds = self.net(images, word_idx)
        losses = self.criterion(preds, masks)
        return losses

class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)
            
        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out

def prepare_data(datum, devices: list = None, allocation: list = None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation))  # The rest might need more/less

        images, word_idx, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx] = gradinator(images[cur_idx].to(device))
                word_idx[cur_idx] = gradinator(word_idx[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx] = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images) - 1)].size()

            for idx, (image, word_idx, target, mask, num_crowd) in enumerate(
                    zip(images, word_idx, targets, masks, num_crowds)):
                images[idx], word_idx[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, word_idx, target, mask, num_crowd, w, h)

        cur_idx = 0
        split_images, split_word_idx, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(5)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx] = torch.stack(images[cur_idx:cur_idx + alloc], dim=0)
            split_word_idx[device_idx] = torch.stack(word_idx[cur_idx:cur_idx + alloc], dim=0)
            split_targets[device_idx] = targets[cur_idx:cur_idx + alloc]
            split_masks[device_idx] = torch.stack(masks[cur_idx:cur_idx + alloc], dim=0)
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx + alloc]

            cur_idx += alloc

        return split_images, split_word_idx, split_targets, split_masks, split_numcrowds

def gradinator(x):
    '''
    x.requires_grad = False
    '''
    x.requires_grad = False
    return x

def train(args):
    cur_lr = args.lr

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset_unc = ReferDataset(image_path=cfg.train_images,
                           info_file=cfg.train_info,
                           augment=True,
                           img_size=cfg.max_size)
    # dataset_uncp = ReferDataset(image_path='../datasets/generated_data/unc+/train_batch/',
    #                        info_file='../datasets/generated_data/unc+/train_batch/unc+_train.npy',
    #                        img_size=cfg.max_size)
    # dataset_uncp = ReferDataset(image_path='../datasets/generated_data/Gref/train_batch/',
    #                        info_file='../datasets/generated_data/Gref/train_batch/Gref_train.npy',
    #                        img_size=cfg.max_size)
    # all_dataset = dataset_uncp + dataset_unc + dataset_uncp
    all_dataset = dataset_unc
    
    # im, text_batch, (gt, masks, num_crowds) = dataset[0]
    
    # validate, print validation information
    if args.validation_epoch > 0:
        eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])
        val_dataset = ReferDataset(image_path=cfg.valid_images,
                                   info_file=cfg.valid_info,
                                   augment=False, 
                                   img_size=cfg.max_size, resize_gt=False)

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = model()
    net = yolact_net
    net.train()   # self.training = True

    if args.log:
        import json
        args_json = json.dumps(dict(args._get_kwargs()))
        log = Log(cfg.name, args.log_folder, args_json,
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)


    # trans_param = []  
    # res_param = []
    # for name, p in net.named_parameters():
    #     if 'attention3' in name or 'attention4' in name or 'attention5' in name:
    #         trans_param.append(p)
    #     else:
    #         res_param.append(p)

    # res_optimizer = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    # # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_iter - args.lr_warmup_until)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(trans_optimizer, T_max=cfg.max_iter - args.lr_warmup_until)
    print(optimizer)
    # print(res_optimizer)
    # print(trans_optimizer)
    print(args.lr_warmup_until)
    print() 

    logfile = open("logs/acc_"+ args.names +'.txt', 'a')
    print(optimizer, file=logfile)
    print(args.lr_warmup_until, file=logfile)
    logfile.close()


    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        # yolact_net.load_weights(args.resume)

        checkpoint = torch.load(args.resume)
        yolact_net.load_state_dict(checkpoint['state_dict'], strict=False)
        # print('loading checkpoint!')
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # cur_lr = checkpoint['cur_lr']

        # if args.start_iter == -1:
        #     args.start_iter = SavePath.from_str(args.resume).iteration

    criterion = CEL()

    net = CustomDataParallel(NetLoss(net, criterion))
    # net = nn.DataParallel(NetLoss(net, criterion))

    if args.cuda:
        net = net.cuda()
    
    # Initialize everything
    if not cfg.freeze_bn: 
        print('Freeze bn .....')
        yolact_net.freeze_bn() # Freeze bn so we don't kill our means

    # yolact_net(torch.zeros(2, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(2, 20).cuda().long())
    if not cfg.freeze_bn: 
        print('Freeze bn again .....')
        yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(all_dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    print('data_length = ', len(all_dataset), ', batch_size = ', args.batch_size)
    print('num_epoch: ', num_epochs)

    logfile = open("logs/acc_"+ args.names +'.txt', 'a')
    print('data_length = ', len(all_dataset), ', batch_size = ', args.batch_size, file=logfile)
    print('num_epoch: ', num_epochs, '\n', file=logfile)
    logfile.close()
    
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0
    
    # pdb.set_trace()
    data_loader = data.DataLoader(all_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, 
                                  collate_fn=detection_collate, 
                                  pin_memory=True)  # 把tensor放到GPU中，加快速度

    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    save_latest_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_last_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }
    
    best = {
            'epoch': 0,
            'iteration': 0,
            'path': '',
            'best_acc': -1.,
            'last': ''
        }

    print('Begin training!')
    print()
    writer = SummaryWriter(args.board_folder)
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue
            
            for datum in data_loader:
                '''
                images, word_idx, (targets, masks, num_crowds) = datum
                 - images: [b, 3, 320, 320]
                 - text: [12, 20]
                 - labels: 
                    - target: [B, 1, 5]
                    - masks: [B, 2, 320, 320],  2 segmentation + edge
                    - num_crowd: [0,0,...]
                '''
                images, word_idx, (targets, masks, num_crowds) = datum

                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if args.lr_warmup_until > 0 and iteration <= args.lr_warmup_until:                    
                    # cur_lr = set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / args.lr_warmup_until) + cfg.lr_warmup_init)           
                    cur_lr = set_lr(optimizer, iteration / args.lr_warmup_until * args.lr) 
                else:
                    scheduler.step()
                    cur_lr = scheduler.get_lr()[0]

                # # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                # # 'lr_steps': (100000, 200000) 
                # while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                #     step_index += 1
                #     # cur_lr = set_lr(res_optimizer, args.lr * (args.gamma ** step_index))
                #     res_optimizer.param_groups[0]['lr'] = args.lr * (args.gamma ** step_index)
                #     print('learning rate change to ', cur_lr)
                
                # res_lr = get_lr(res_optimizer)
                # writer.add_scalar('optim/res_lr', res_lr, iteration)
                # writer.add_scalar('optim/trans_lr', cur_lr, iteration)
                # # Zero the grad to get ready to compute gradients
                # trans_optimizer.zero_grad()
                # res_optimizer.zero_grad()

                writer.add_scalar('optim/lr', cur_lr, iteration)
                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)                

                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])

                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # # Backprop
                # loss.backward() # Do this to free up vram even if loss is not finite
                loss.backward()

                writer.add_scalar('train/loss', loss.data.cpu().numpy(), iteration)

                # trans_optimizer.step()
                # res_optimizer.step() 
                if torch.isfinite(loss).item():
                    optimizer.step()
                else:
                    print('loss is NaN')

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    writer.add_scalar(f'train/{k}', losses[k].data.cpu().numpy(), iteration)
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                    
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    
                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)
                    logfile = open("logs/logfile_"+ args.names+'.txt', 'a')
                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                          % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True,
                          file=logfile)
                    logfile.close()

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(losses[k].item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                        
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu
                
                iteration += 1

                # if iteration % args.save_interval == 0 and iteration != args.start_iter:
                if iteration >= cfg.max_iter - 3*args.save_interval and iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    acc = compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
                    writer.add_scalar('test/acc', acc*100, epoch)

                    # save best model
                    if acc > best['best_acc']:
                        path = save_path(epoch, iteration) 
                        yolact_net.save_weights(path, epoch, iteration, cur_lr, optimizer)
                        if os.path.exists(best['path']):
                            print('remove ', best['path'])
                            os.remove(best['path'])
                        best['best_acc'] = acc
                        best['epoch'] = epoch
                        best['iteration'] = iteration
                        best['path'] = path 
                    # save latest model 
                    if os.path.exists(best['last']):
                        print('remove last: ', best['last'])
                        os.remove(best['last'])
                    last_path = save_latest_path(epoch, iteration)
                    yolact_net.save_weights(last_path, epoch, iteration, cur_lr, optimizer)
                    best['last'] = last_path 
                    print(best)

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)
            # This is done per epoch
            # if args.validation_epoch > 0:
            #     if epoch % args.validation_epoch == 0 and epoch > 0:
            #         # compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
            #         compute_validation_map(epoch, iteration, net, val_dataset, log if args.log else None)

        # Compute validation mAP after training is finished
        # acc = compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
        acc = compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)

        # save best model
        if acc > best['best_acc']:
            path = save_path(epoch, iteration) 
            yolact_net.save_weights(path, epoch, iteration, cur_lr, optimizer)
            if os.path.exists(best['path']):
                print('remove ', best['path'])
                os.remove(best['path'])
            best['best_acc'] = acc
            best['epoch'] = epoch
            best['iteration'] = iteration
            best['path'] = path 
        # save latest model 
        if os.path.exists(best['last']):
            print('remove last: ', best['path'])
            os.remove(best['last'])
        # last_path = save_latest_path(epoch, iteration)
        # yolact_net.save_weights(last_path, epoch, iteration, cur_lr, optimizer)
        # best['last'] = last_path 
        print(best)

    except KeyboardInterrupt:
        # if args.interrupt:
        #     print('Stopping early. Saving network...')
            
        #     # Delete previous copy of the interrupted network so we don't spam the weights folder
        #     SavePath.remove_interrupt(args.save_folder)
            
        #     yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'), epoch, iteration, cur_lr, optimizer)
        exit()

    # yolact_net.save_weights(save_path(epoch, iteration), epoch, iteration, cur_lr, optimizer)


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    # global cur_lr
    # cur_lr = new_lr
    return new_lr


def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """
    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            
            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        for k in losses:
            losses[k] /= iterations
                   
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    with torch.no_grad():
        yolact_net.eval()
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info, acc = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()
        logfile = open("logs/logfile_"+ args.names+'.txt', 'a')
        print('iter:{} epoch:{}: {}'.format(int(iteration / args.save_interval), epoch, val_info), file=logfile)
        logfile.close()
        logfile = open("logs/acc_"+ args.names +'.txt', 'a')
        print('iter:{} epoch:{}: {}'.format(int(iteration / args.save_interval), epoch, val_info), file=logfile)
        logfile.close()
        val_info = {"val_info":val_info}

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

        return acc 

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(
        description='Yolact Training Script')
    #'weights_my/Base_4_67500.pth'
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                            ', the model will resume training from the interrupt file.')
    parser.add_argument('--start_iter', default=-1, type=int,
                        help='Resume training at this iter. If this is -1, the iteration will be'\
                            'determined from the file name.')
    parser.add_argument('--num_workers', default=4, type=int,   ###########################
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning_rate', default=None, type=float,#0.00006
                        help='Initial learning rate. Leave as None to read this from the config.')
    parser.add_argument('--momentum', default=None, type=float,
                        help='Momentum for SGD. Leave as None to read this from the config.')
    parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                        help='Weight decay for SGD. Leave as None to read this from the config.')
    parser.add_argument('--gamma', default=None, type=float,
                        help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
    parser.add_argument('--log_folder', default='logs/',
                        help='Directory for saving logs.')
    parser.add_argument('--config', default='default_cfg',
                        help='The config object to use.')
    parser.add_argument('--save_interval', default=5000, type=int,
                        help='The number of iterations between saving the model.')
    parser.add_argument('--validation_size', default=-1, type=int,
                        help='The number of images to use for validation.')
    parser.add_argument('--validation_epoch', default=1, type=int,
                        help='Output validation information every n iterations. If -1, no validation.')
    parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                        help='Only keep the latest checkpoint instead of each one.')
    parser.add_argument('--keep_latest_interval', default=5000, type=int,
                        help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--no_log', dest='log', action='store_false',
                        help='Don\'t log per iteration information into log_folder.')
    parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                        help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
    parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                        help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
    parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                        help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

    parser.add_argument('--text_model', type=str, default='lstm', help='bert or RNN')
    parser.add_argument('--backbone', type=str)

    parser.add_argument('--lr_warmup_until', default=1500, type=int)
    parser.add_argument('--board_folder', default='runs',
                        help='board_folder')                
    parser.add_argument('--save_folder', default='iter6_ckpt/',  # swin_downsample_small_TMMM
                        help='Directory for saving checkpoint models.')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--batch_alloc', default='24', type=str,  #'10,12' default 34, single gpu
                        help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
    parser.add_argument('--batch_size', default=24, type=int,
                        help='Batch size for training')
    parser.add_argument('--names', default='iter46full', type=str)
    # parser.add_argument('--logfile', default='my_model3_1230', type=str)

    parser.set_defaults(keep_latest=False, log=False, log_gpu=False, interrupt=True, autoscale=True)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # 30,30,6: gpu1 30, gpu2 30, gpu3 6,  need 12h
    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    if args.config is not None:
        set_cfg(args.config)

    if args.dataset is not None:
        set_dataset(args.dataset)

    # adjust lr according to batchsize
    if args.autoscale and args.batch_size != 8:
        factor = args.batch_size / 8
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

        cfg.lr *= factor
        cfg.max_iter //= factor
        cfg.lr_steps = [x // factor for x in cfg.lr_steps]
        # print('no scaling learning rate')

        print('max_iter: ', cfg.max_iter)
        print('lr_steps: ', cfg.lr_steps)

    # Update training parameters from the config if necessary
    def replace(name):
        if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
    replace('lr')
    replace('decay')
    replace('gamma')
    replace('momentum')

    if torch.cuda.device_count() == 0:
        print('No GPUs detected. Exiting...')
        exit(-1)

    if args.batch_size // torch.cuda.device_count() < 6:
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
        cfg.freeze_bn = True

    logfile = open("logs/acc_"+ args.names +'.txt', 'a')
    print('model_name: ', args.names, file=logfile)
    print('batch_size: ', args.batch_size, file=logfile)
    print('max_iter: ', cfg.max_iter, file=logfile)
    print('lr_steps: ', cfg.lr_steps, file=logfile)
    print('lr: ', args.lr, file=logfile)
    print('train_info:', cfg.train_info, file=logfile)
    print('train_images:', cfg.train_images, file=logfile)
    print('valid_info:', cfg.valid_info, file=logfile)
    print('valid_images:', cfg.valid_images, file=logfile)
    logfile.close()

    # loss_types = ['A','B', 'C', 'M', 'P', 'D', 'E', 'S', 'I', 'N', 'M1', 'M2', 'M3', 'M4', 'M5', 'B1', 'B2', 'B3', 'B4', 'B5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F0', 'F1']
    loss_types = ['sal', 'fpn', 'fpn5', 'fpn4', 'fpn3', 'fpn2', 'D1', 'D2', 'D3', 'D4', 'D5']

    # if torch.cuda.is_available():
    #     if args.cuda:
    #         torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #     if not args.cuda:
    #         print("WARNING: It looks like you have a CUDA device, but aren't " +
    #             "using CUDA.\nRun with --cuda for optimal training speed.")
    #         torch.set_default_tensor_type('torch.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')

    train(args)

