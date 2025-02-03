# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model, Detect, RepConv, Faster_Block, C3_Faster, IFM, TopBasicLayer, SimFusion_3in, SimFusion_4in, InjectionMultiSum_Auto_pool, AdvPoolFusion, PyramidPoolAgg, Detect_DyHead_Prune, replace_c2f_with_c2f_v2
from models.convnextv2 import LayerNorm
from models.EfficientFormerv2 import Attention4D, Attention4DDownsample
from models.dyhead_prune import DyHeadBlock_Prune
# from timm.models.layers import SqueezeExcite
from timm.models._efficientnet_blocks import SqueezeExcite
from models.fasternet import MLPBlock
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

import val

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()

import copy
import torch_pruning as tp
from functools import partial
from thop import clever_format
import matplotlib.pylab as plt
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

from MyPrune import RepConvPruner, LayerNormPruner, DyHeadBlockPruner
def get_pruner(opt, model, example_inputs):
    sparsity_learning = False
    if opt.prune_method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "l1":
        # https://arxiv.org/abs/1608.08710
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "lamp":
        # https://arxiv.org/abs/2010.07611
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "slim":
        # https://arxiv.org/abs/1708.06519
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_slim":
        # https://tibshirani.su.domains/ftp/sparse-grlasso.pdf
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning, group_lasso=True)
    elif opt.prune_method == "group_norm":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_sl":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "growing_reg":
        # https://arxiv.org/abs/2012.09243
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=opt.reg, delta_reg=opt.delta_reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_hessian":
        # https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
        imp = tp.importance.HessianImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_taylor":
        # https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf
        imp = tp.importance.TaylorImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    else:
        raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    customized_pruners = {}
    round_to = None
    
    # ignore output layers##############################################################################
    # for yolov5n.yaml
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            ignored_layers.append(m)
        # if isinstance(m, C3_Faster):
        #     ignored_layers.append(m)

    # for models/yolov5n-C3-Faster.yaml
    #      ignored_layers.append(model.model[77]) ##不剪枝第77行
    # RTDETRDecoder
    # for models/yolov5n-C3-Faster-Rep.yaml
    # customized_pruners[RepConv] = RepConvPruner()
    # for k, m in model.named_modules():
    #     if isinstance(m, Detect):
    #         ignored_layers.append(m)
    #     if isinstance(m, Faster_Block):
    #         ignored_layers.append(m.mlp[-1])
    
    # for models/yolov5n_repvit.yaml
    # for k, m in model.named_modules():
    #     if isinstance(m, Detect):
    #         ignored_layers.append(m)
    #     if isinstance(m, SqueezeExcite):
    #         ignored_layers.append(m)
    #     if isinstance(m, Faster_Block):
    #         ignored_layers.append(m.mlp[-1])

    # for models/yolov5n-C3-Faster.yaml
    # for k, m in model.named_modules():
    #     if isinstance(m, Detect):
    #         ignored_layers.append(m)
    #     if isinstance(m, SqueezeExcite):
    #         ignored_layers.append(m)
    #     if isinstance(m, Faster_Block):
    #         ignored_layers.append(m.mlp[-1])



    # for models/yolov5-fasternet-goldyolo-asf.yaml
    # for k, m in model.named_modules():
    #     if isinstance(m, Detect):
    #         ignored_layers.append(m)
    #     if isinstance(m, MLPBlock):
    #         ignored_layers.append(m.mlp[-1])
    #     if isinstance(m, (IFM, TopBasicLayer)):
    #         ignored_layers.append(m)
    
    # for models/yolov5-dyhead-prune.yaml
    # customized_pruners[DyHeadBlock_Prune] = DyHeadBlockPruner()
    # for k, m in model.named_modules():
    #     if isinstance(m, Detect_DyHead_Prune):
    #         ignored_layers.append(m.m)
    #         ignored_layers.append(m.conv)
        
        
    print(ignored_layers)
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=opt.iterative_steps,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=opt.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        customized_pruners=customized_pruners,
        round_to=round_to,
        root_module_types=[nn.Conv2d, nn.Linear]
    )
    return sparsity_learning, imp, pruner

linear_trans = lambda epoch, epochs, reg, reg_ratio: (1 - epoch / (epochs - 1)) * (reg - reg_ratio) + reg_ratio
def model_prune(opt, model, imp, prune, example_inputs, testloader, imgsz_test, trainloader):
    N_batchs = 10
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    
    # model.eval()
    base_model = copy.deepcopy(model)
    with HiddenPrints():
        ori_flops, ori_params = tp.utils.count_ops_and_params(base_model, example_inputs)
    ori_flops = ori_flops * 2.0
    ori_flops_f, ori_params_f = clever_format([ori_flops, ori_params], "%.3f")
    ori_result, _, _ = val.run(data_dict, None, batch_size=opt.batch_size * 2,
                                 imgsz=imgsz_test, plots=False, model=base_model, dataloader=testloader)
    ori_map50, ori_map = ori_result[2], ori_result[3]
    iter_idx, prune_flops = 0, ori_flops
    speed_up = 1.0
    LOGGER.info('begin pruning...')
    while speed_up < opt.speed_up:
        model.train()
        loss_func = ComputeLoss(model)
        if isinstance(imp, tp.importance.HessianImportance):
            for k, (imgs, targets, paths, _) in enumerate(trainloader):
                if k >= N_batchs: break
                imgs = imgs.to(device, non_blocking=True).float() / 255
                output = model(imgs) 
                # compute loss for each sample
                loss = loss_func(output, targets.to(device))[0]
                imp.zero_grad() # clear accumulated gradients
                model.zero_grad() # clear gradients
                loss.backward(retain_graph=True) # simgle-sample gradient
                imp.accumulate_grad(model) # accumulate g^2
        elif isinstance(imp, tp.importance.TaylorImportance):
            for k, (imgs, targets, paths, _) in enumerate(trainloader):
                if k >= N_batchs: break
                imgs = imgs.to(device, non_blocking=True).float() / 255
                output = model(imgs)
                loss = loss_func(output, targets.to(device))[0]
                loss.backward()
        
        model.eval()
        iter_idx += 1
        prune.step(interactive=False)
        prune_result, _, _ = val.run(data_dict, None, batch_size=opt.batch_size * 2,
                                 imgsz=imgsz_test, plots=False, model=copy.deepcopy(model), dataloader=testloader)
        prune_map50, prune_map = prune_result[2], prune_result[3]
        with HiddenPrints():
            prune_flops, prune_params = tp.utils.count_ops_and_params(model, example_inputs)
        prune_flops = prune_flops * 2.0
        prune_flops_f, prune_params_f = clever_format([prune_flops, prune_params], "%.3f")
        speed_up = ori_flops / prune_flops # ori_model_GFLOPs / prune_model_GFLOPs
        LOGGER.info(f'pruning... iter:{iter_idx} ori model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%) map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f}) map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f}) Speed Up:{ori_flops / prune_flops:.2f}')
        
        if prune.current_step == prune.iterative_steps:
            break
    
    if isinstance(imp, tp.importance.HessianImportance):
        imp.zero_grad()
    model.zero_grad()
    torch.cuda.empty_cache()
    
    LOGGER.info('pruning done...')
    LOGGER.info(f'model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) Speed Up:{ori_flops / prune_flops:.2f}')
    LOGGER.info(f'model params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%)')
    LOGGER.info(f'model map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f})')
    LOGGER.info(f'model map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f})')
    return model

def sparsity_learning_train(opt, model, prune, dataloader, testloader, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, data, nosave, workers = \
        increment_path(Path(opt.project) / f'{opt.name}_sl', exist_ok=opt.exist_ok), opt.sl_epochs, opt.batch_size, opt.weights, opt.single_cls, opt.data, \
        opt.nosave, opt.workers
    cuda = device.type != 'cpu'
    plots = True
    callbacks.run('on_pretrain_routine_start')
    
    # check AMP
    amp = check_amp(model)  
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'visual').mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    
    # Hyperparameters
    # if isinstance(opt.sl_hyp, str):
    with open(opt.sl_hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('sparsity_learning hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    # save hyp and opt
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))
    
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
    # Register actions
    for k in methods(loggers):
        callbacks.register_action(k, callback=getattr(loggers, k))
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch, best_sl = 0.0, 0, {}
    
    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            'WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
            'See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started.'
        )
        model = torch.nn.DataParallel(model)
    
    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    
    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)
    
    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=0), False
    compute_loss = ComputeLoss(model)  # init loss class
    compute_loss_ota = ComputeLossOTA(model) 
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            # with torch.cuda.amp.autocast(amp):
            pred = model(imgs)  # forward
            if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
            else:
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.

            # Backward
            # scaler.scale(loss).backward()
            loss.backward()

            # with amp.autocast(enabled=cuda):
            if isinstance(prune, (tp.pruner.BNScalePruner,)):
                if opt.reg_decay_type == 'linear':
                    reg = linear_trans(epoch, epochs, opt.reg, opt.reg * opt.reg_decay)
                elif opt.reg_decay_type == 'step':
                    reg = opt.reg * (opt.reg_decay ** (epoch // opt.reg_decay_step))
                elif opt.opt.reg_decay_type == 'constant':
                    reg = opt.reg
                prune.regularize(model, reg=reg)
            elif isinstance(prune, (tp.pruner.GroupNormPruner, tp.pruner.GrowingRegPruner)):
                reg = opt.reg
                prune.regularize(model)
            
            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                # scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                # scaler.step(optimizer)  # optimizer.step
                # scaler.update()
                optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------
        
        if isinstance(prune, (tp.pruner.GrowingRegPruner,)):
            prune.update_reg()
        
        if ema:
            model_sl = ema.ema.state_dict()
        else:
            model_sl = model.state_dict()
        
        bn_weight = []
        for name in model_sl:
            if 'weight' in name and len(model_sl[name].size()) == 1:
                weight = model_sl[name].data.cpu().abs().clone().numpy().reshape((-1))
                bn_weight.append(weight)
        bn_weight = np.concatenate(bn_weight)
        bn_weight = np.sort(bn_weight)
        bn_weight_percent = np.percentile(bn_weight, [1, 5, 10, 25, 50, 75])
        sparsity_ratio = np.sum(bn_weight < 1e-6) / bn_weight.shape[0]
        if f'{sparsity_ratio:.3f}' not in best_sl:
            best_sl[f'{sparsity_ratio:.3f}'] = 0.0
        
        del model_sl
        
        plt.figure(figsize=(15, 5), clear=True)
        plt.plot(bn_weight)
        plt.title(f'sparsity_ratio:{sparsity_ratio:.3f}\n')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/visual/{epoch}_sl_{sparsity_ratio:.3f}.png')
        plt.close('all')
        LOGGER.info(f'epoch:{epoch} reg:{reg:.5f} sparsity_ratio:{sparsity_ratio:.5f} bn_weight_1:{bn_weight_percent[0]:.10f} bn_weight_5:{bn_weight_percent[1]:.8f} bn_weight_10:{bn_weight_percent[2]:.8f}\nbn_weight_25:{bn_weight_percent[3]:.5f} bn_weight_50:{bn_weight_percent[4]:.5f} bn_weight_75:{bn_weight_percent[5]:.5f}')
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            results, maps, _ = validate.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            half=amp,
                                            model=ema.ema,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            plots=False,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            if fi > best_sl[f'{sparsity_ratio:.3f}']:
                best_sl[f'{sparsity_ratio:.3f}'] = fi
            
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                if best_sl[f'{sparsity_ratio:.3f}'] == fi:
                    torch.save(ckpt, w / 'best_sl_{:.3f}.pt'.format(sparsity_ratio))
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    
    sl = sorted(best_sl.keys(), key=lambda x:float(x))[-1]
    best_sl_model = w / 'best_sl_{}.pt'.format(sl)
    
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best, best_sl_model:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best_sl_model:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)
    torch.cuda.empty_cache()
    return best_sl_model

def finetune(opt, model, dataloader, testloader, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, data, nosave, workers = \
        increment_path(Path(opt.project) / f'{opt.name}', exist_ok=opt.exist_ok), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.data, \
        opt.nosave, opt.workers
    cuda = device.type != 'cpu'
    plots = True
    callbacks.run('on_pretrain_routine_start')
    
    # check AMP
    amp = check_amp(model)  
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best, prune = w / 'last.pt', w / 'best.pt', w / 'prune.pt'
    
    # save prune model before finetune
    model.zero_grad() # clear grad
    ckpt = {'model': deepcopy(de_parallel(model)).half(), 'ema': None}
    torch.save(ckpt, prune)
    LOGGER.info(colorstr(f'Pruning after Finetune before the model is saved in:{prune}'))
    
    # Hyperparameters
    # if isinstance(opt.hyp, str):
    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('finetune hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    # save hyp and opt
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))
    
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
    # Register actions
    for k in methods(loggers):
        callbacks.register_action(k, callback=getattr(loggers, k))
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    
    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            'WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
            'See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started.'
        )
        model = torch.nn.DataParallel(model)
    
    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    
    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)
    
    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    compute_loss_ota = ComputeLossOTA(model)
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            results, maps, _ = validate.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            half=amp,
                                            model=ema.ema,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            plots=False,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results

def parse_opt(known=False):
    # 代码来自:BiliBli 魔鬼面具up主
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    # ------------------------------BiliBili 魔鬼面具up主------------------------------
    # prune
    parser.add_argument('--prune_method', type=str, default=None, help='prune method')
    parser.add_argument('--speed_up', type=float, default=2.0, help='speed up')
    parser.add_argument("--global_pruning", action="store_true")
    parser.add_argument("--max_sparsity", type=float, default=1.0)
    parser.add_argument("--iterative_steps", type=int, default=200)
    
    # ------------------------------BiliBili 魔鬼面具up主------------------------------
    # sparsity learning
    parser.add_argument("--reg", type=float, default=5e-4)
    parser.add_argument("--delta_reg", type=float, default=1e-4, help='for growing regularization')
    parser.add_argument("--sl_hyp", type=str, default=r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\data\hyps\hyp.scratch.sl.yaml', help='hyperparameters path for sparsity learning')
    parser.add_argument("--sl_epochs", type=int, default=300)
    parser.add_argument("--sl_model", type=str, default="", help='sparsity learning trained model weights')
    
    # ------------------------------BiliBili 魔鬼面具up主------------------------------
    parser.add_argument("--reg_decay_type", type=str, default='linear', choices=['constant', 'linear', 'step'], help='reg decay type choice in sparsity learning')
    parser.add_argument("--reg_decay", type=float, default=0.01)
    parser.add_argument("--reg_decay_step", type=int, default=10, help='reg decay step size in sparsity learning and reg_decay_type==step')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt))
    
    # Init Callback
    callbacks = Callbacks()
    
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')
    
    # Config
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
    
    # Model
    weights = opt.weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        # model = attempt_load(weights, device=device, inplace=False, fuse=False)
        model = torch.load(weights, map_location=device)
        if model['ema']:
           model = model['ema'].float()
        else:
            model = model['model'].float()
        for p in model.parameters():
            p.requires_grad_(True)
        model.info(img_size=opt.imgsz)
        # for c2f
        replace_c2f_with_c2f_v2(model.model)
        model.to(device)
        LOGGER.info(f'Loaded {weights}')  # report
    else:
        assert weights.endswith('.pt'), "compress need weights."
    
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    
    # Hyperparameters
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              opt.batch_size // WORLD_SIZE,
                                              gs,
                                              opt.single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=opt.workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {opt.data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       opt.batch_size // WORLD_SIZE * 2,
                                       gs,
                                       opt.single_cls,
                                       hyp=hyp,
                                       cache=opt.cache,
                                       rect=False,
                                       rank=-1,
                                       workers=opt.workers,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        callbacks.run('on_pretrain_routine_end', labels, names)
    
    # prune dataloader
    prune_loader, _ = create_dataloader(train_path,
                                              imgsz,
                                              opt.batch_size // WORLD_SIZE // 2,
                                              gs,
                                              opt.single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=opt.workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('prune: '),
                                              shuffle=True,
                                              seed=opt.seed)
    
    # get prune
    example_inputs = torch.randn((1, 3, imgsz, imgsz)).to(device)
    sparsity_learning, imp, prune = get_pruner(opt, model, example_inputs)
    
    # sparsity_learning
    if sparsity_learning and not opt.sl_model:
        opt.sl_model = sparsity_learning_train(opt, model, prune, train_loader, val_loader, device, callbacks)
    
    if sparsity_learning:
        model = torch.load(opt.sl_model, map_location=device)
        model = model['ema' if model.get('ema') else 'model'].float()
        # model = attempt_load(opt.sl_model, device=device, inplace=False, fuse=False)
        for p in model.parameters():
            p.requires_grad_(True)
        _, imp, prune = get_pruner(opt, model, example_inputs)
        LOGGER.info(f'Loaded sparsity_learning weights from {opt.sl_model}')  # report
    
    # pruning
    model = model_prune(opt, model, imp, prune, example_inputs, val_loader, imgsz, prune_loader)
    
    # test fuse
    fuse_model = deepcopy(model)
    for p in fuse_model.parameters():
            p.requires_grad_(False)
    fuse_model.fuse()
    del fuse_model

    # finetune
    finetune(opt, model, train_loader, val_loader, device, callbacks)