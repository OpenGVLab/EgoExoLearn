# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import math
import os
import pandas as pd
import sys
import time
import pickle
import random 

import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn.parallel
import wandb
from omegaconf import OmegaConf
from copy import deepcopy
import numpy as np
import logging

from models.builder import build_model
from models import model_utils
from models.tokenizer import generate_tokenizer
from function.meter import AverageMeter, ProgressMeter
from function import distributed as dist_utils
from function.utils import build_train_loader, build_val_loader, build_optimizer, resume_checkpoint, build_scheduler
from function.config import get_config
from function.logger import get_logger

def get_args_parser():
    parser = argparse.ArgumentParser(description='EgoExoLearn Association training and evaluation', add_help=False)
    # Data
    parser.add_argument('--config', default='configs/default.yml', type=str)

    # System
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
   
    parser.add_argument('--testonly', action='store_true', help='whether to perform test only')
    return parser


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    
def main(args):
    ### Prepare env ###
    cfg = get_config(args)
    os.makedirs(cfg.output, exist_ok=True)
    
    dist_utils.init_distributed_mode(args)
    logger = get_logger(cfg)   
    ### save config file ###
    if dist_utils.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config save to {path}')

    ### log config ###
    logger.info(OmegaConf.to_yaml(cfg))

    global best_acc1
    random_seed(cfg.train.seed, dist_utils.get_rank())
    logger.info(f'Creating model:{cfg.model.name}')
    model = build_model(cfg.model)

    if cfg.model.freeze_temperature:
        logger.info('Freeze logit temperature')
        if hasattr(model, 'logit_scale'):
            model.logit_scale.requires_grad = False

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            # find_unused_parameters=cfg.train.find_unused_parameters
            find_unused_parameters=True,
        )
    tokenizer = generate_tokenizer(cfg.model.name)    

    criterion = model_utils.get_loss(cfg.model.name, args, cfg, tokenizer=tokenizer).cuda(args.gpu)
    optimizer = build_optimizer(cfg.train, model, criterion)
    scaler = amp.GradScaler(enabled=not cfg.train.disable_amp)
    lr_schedule = build_scheduler(cfg)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    loaded_resume = resume_checkpoint(cfg, model, optimizer, scaler, criterion)
    start_epoch, best_acc1 = loaded_resume['start_epoch'], loaded_resume['best_acc1']
    cudnn.benchmark = True
    
    logger.info("=> creating dataset")
    train_loader, train_sampler = build_train_loader(args, cfg, tokenizer)
    egobridge_v2v_loader = build_val_loader(args, cfg, dataset_name='egobridge_v2v', tokenizer=deepcopy(tokenizer))
    
    if dist_utils.is_main_process() and cfg.wandb:
        wandb_id = os.path.split(cfg.output)[-1]
        wandb.init(project='egoexo', id=wandb_id, config=args, resume='allow')

    if cfg.test.testonly:    
        ### V2V ###
        metrics = validate_v2v_mcq(egobridge_v2v_loader, model, use_half=False, cfg=cfg, args=args)
        print(metrics)
        exit(0)

    best_metric = 0.
    print("=> beginning training")
    for epoch in range(start_epoch, cfg.train.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger)
        
        ### logging training stats ###
        for k, v in train_stats.items():
            logger.info(f'Epoch {epoch}: Train_{k}: {round(v, 3)}')

        ### saving per epoch model ckpt before evaluation ###
        logger.info('=> saving per-epoch checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
            'scaler': scaler.state_dict(),
            'best_acc1': best_metric,
            'cfg': cfg,
        }, False, cfg.output, is_epoch=True)

        logger.info('=> 0-shot on MCQ')
        v2v_metrics = validate_v2v_mcq(egobridge_v2v_loader, model, use_half=False, cfg=cfg, args=args)
        logger.info('V2V Ego->Exo: {:.3f} | V2T Exo->Ego: {:.3f}'.format(v2v_metrics['Ego->Exo'], v2v_metrics['Exo->Ego']))
        avg_map = 0.5 * (v2v_metrics['Ego->Exo'] + v2v_metrics['Exo->Ego'])
        
        if avg_map > best_metric:
            is_best = True
            best_metric = avg_map
        else:
            is_best = False   

        ### save checkpoint ###
        is_epoch = ((epoch + 1) % cfg.train.save_freq) == 0

        if args.distributed and cfg.train.use_zero:
            logger.info("=> consolidating state_dict before saving (due to ZeRO)")
            optimizer.consolidate_state_dict()

        logger.info('=> saving the best checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
            'scaler': scaler.state_dict(),
            'best_acc1': best_metric,
            'cfg': cfg,
        }, is_best, cfg.output, is_epoch=is_epoch)


def train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = model_utils.get_metric_names(cfg)
    
    iters_per_epoch = len(train_loader) // cfg.train.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // cfg.train.update_freq
                
        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]
        
        batch_size = inputs['text'].size(0)
        model_inputs = [inputs['video'].cuda(args.gpu), inputs['text'].cuda(args.gpu)]
        
        # compute output
        with amp.autocast(enabled=not cfg.train.disable_amp):
            outputs = model(
                *model_inputs,
                use_checkpoint=cfg.train.use_checkpoint,
                norm_embed=cfg.model.norm_embed
            )     
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= cfg.train.update_freq

        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % cfg.train.update_freq != 0:
            continue

        if cfg.train.clip_grad_value is not None:
            scaler.unscale_(optimizer)
            if cfg.train.clip_grad_type == 'norm':
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.clip_grad_value, norm_type=2.
                )
            elif cfg.train.clip_grad_type == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg.train.clip_grad_value)
            else:
                assert False, f"Unknown clip mode ({cfg.train.clip_grad_type})."
        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        ### adjust logit scale ###
        if hasattr(dist_utils.get_model(model), 'logit_scale'):
            # clamp logit scale to [0, 100]
            dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = dist_utils.get_model(model).logit_scale.exp().item()
        else:
            logit_scale = torch.nan

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), cfg.train.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % cfg.train.print_freq == 0:
            if dist_utils.is_main_process():
                train_iter_log = {
                            'iter': data_iter,
                            **{k: round(v.item(), 3) for k, v in loss_dict.items()},
                           'scaler': round(scaler.get_scale(), 3), 
                           'logit': round(logit_scale, 3)}
                train_iter_log_str = ''
                for logk, logv in train_iter_log.items():
                    train_iter_log_str += f'{logk}:{logv}  '

                logger.info(train_iter_log_str)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate_v2v_mcq(val_loader, model, use_half=False, cfg=None, args=None):
    model.eval()
    if use_half:
        model.half()
    with torch.no_grad():
        print('=> start forwarding')
        all_preds = []
        all_gts = []
        all_types = []
        all_uids = []
        end_time = time.time()
        for i, inputs in enumerate(val_loader):
            if i % 10 == 0:
                print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            
            frame_query = inputs[0].cuda(non_blocking=True)
            frames_options = inputs[1].cuda(non_blocking=True)
            if use_half:
                frames_options = frames_options.half()
        
            answer = inputs[2]
            q_type = inputs[3]
            uid = inputs[-1]
            
            batch_size = frames_options.shape[0]
            frames_options = frames_options.view(-1, *frames_options.shape[2:])

            ### encode videos ###
            image_query_features = dist_utils.get_model(model).encode_image(frame_query)
            image_options_features = dist_utils.get_model(model).encode_image(frames_options)
            
            image_options_features = image_options_features.view(batch_size, -1, *image_options_features.shape[1:])

            all_gts.append(answer)
            all_types.append(q_type)
            all_uids.append(uid)
            for j in range(batch_size):
                similarity_matrix = torch.matmul(image_query_features[j], image_options_features[j].T)
                similarity_matrix = similarity_matrix.cpu().detach()
                all_preds.append(similarity_matrix)          

        
        all_uids = torch.cat(all_uids)
        all_preds = torch.stack(all_preds)
        all_gts = torch.cat(all_gts)
        all_types = torch.cat(all_types)
        
        # save_pred_results(all_uids, all_preds, all_gts)
        metrics = egomcq_accuracy_metrics(all_preds, all_gts, all_types)
        print(metrics)
        return metrics
    
def save_pred_results(uids, preds, gts):
    import pandas as pd
    all_data = []
    predictions = torch.max(preds, 1)[1]
    
    for i in range(len(uids)):
        uid = int(uids[i].cpu().numpy())
        prediction = int(predictions[i].cpu().numpy())
        gt = int(gts[i].cpu().numpy())
        all_data.append([uid, prediction, gt])
    
    df = pd.DataFrame(all_data, columns=['uid', 'pred', 'GT'])
    df.to_csv('pred.csv', index=0)

def egomcq_accuracy_metrics(preds, labels, types):
    metrics = {}
    type_list = torch.unique(types)
    group_list = ['Ego->Exo', 'Exo->Ego']
    for type_i, group_i in zip(type_list, group_list):
        correct = 0
        total = 0
        for pred, label, typer in zip(preds, labels, types):
            if typer == type_i:
                pred_ = torch.argmax(pred)
                if pred_.item() == label.item():
                    correct += 1
                total += 1
        accuracy = correct/total
        metrics[group_i] = accuracy * 100
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EgoExoLearn Association training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
