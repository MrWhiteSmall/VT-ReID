import os
import time
import datetime
import argparse
import logging
import os.path as osp
import numpy as np
from torch import distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.multiprocessing import Process

from configs.default_img import get_img_config

from data import build_dataloader
from models import build_model
from losses import build_losses

from tools.utils import save_checkpoint, set_seed, get_logger

from train import train_model

from test import test, test_prcc

from models.img_resnet import GAP_Classifier

# 监听的端口
port = '12901'

def parse_option():
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--port','-p', type=str, help="e.g. 12902,12903...")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--gpu', default='0,1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    # config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET) 已更改存储目录
    config = get_img_config(args)

    return config

'''
Config:
-----------------------------------------
AUG:
  RC_PROB: 0.5
  RE_PROB: 0.5
  RF_PROB: 0.5
DATA:
  DATASET: ltcc
  HEIGHT: 384
  NUM_INSTANCES: 8
  NUM_WORKERS: 4
  ROOT: /public/home/yangzhe/ltt/lsj/ccdatasets
  TEST_BATCH: 128
  TRAIN_BATCH: 32
  WIDTH: 192
EVAL_MODE: False
GPU: 0,1
LOSS:
  CLA_LOSS: crossentropy
  CLA_M: 0.0
  CLA_S: 16.0
  PAIR_LOSS: triplet
  PAIR_LOSS_WEIGHT: 0.0
  PAIR_M: 0.3
  PAIR_S: 16.0
MODEL:
  FEATURE_DIM: 2048
  NAME: resnet50
  POOLING:
    NAME: avg
    P: 3
  RES4_STRIDE: 1
  RESUME: 
OUTPUT: /public/home/yangzhe/ltt/lsj/ccdatasets/output/ltcc
SEED: 1
TEST:
  EVAL_STEP: 5
  START_EVAL: 0
TRAIN:
  AMP: False
  MAX_EPOCH: 150
  OPTIMIZER:
    NAME: adam
  START_EPOCH: 0
-----------------------------------------
=> LTCC loaded
Dataset statistics:
  ----------------------------------------
  subset   | # ids | # images | # clothes
  ----------------------------------------
  train    |    77 |     9576 |       256
  test     |    75 |     7543 |       221
  query    |    75 |      493 |
  gallery  |    75 |     7050 |
  ----------------------------------------
  total    |   152 |    17119 |       477
  ----------------------------------------
Initializing model: resnet50
'''

'''
Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" 
        to /public/home/yangzhe/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
'''

def main(rank, world_size, config):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    set_seed(config.SEED)
    # get logger
    if not config.EVAL_MODE:
        output_file = osp.join(config.OUTPUT, 'log_train_.log')
    else:
        output_file = osp.join(config.OUTPUT, 'log_test.log')
    logger = get_logger(output_file, rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")
    lr_min = 3.5e-6
    lr_max = 3.5e-4
    step_size_up = 10
    step_size_down = [40, 80]
    gamma = 0.1

    def lr_lambda(current_step):
        if current_step < step_size_up:
            return (lr_max - lr_min) / step_size_up * current_step + lr_min
        else:
            factor = 1
            for s in step_size_down:
                if current_step > s:
                    factor *= gamma
            return lr_max * factor

    # Build dataloader
    if config.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler = build_dataloader(
            config)
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)

    # Build model
    # model就是 Res50
    '''这个就是attention，超简单！ 输入 bz*2048*... 输出 bz*4*...
        1 X 1 空间卷积
    class Part_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2048, out_channels=4, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x
    '''
    model, attention= build_model(config)
    # origin human 三个分类器
    gap_classifier = GAP_Classifier(config, dataset.num_train_pids)
    # human parsing
    gap_classifier_h = GAP_Classifier(config, dataset.num_train_pids)
    # black cloth
    gap_classifier_b = GAP_Classifier(config, dataset.num_train_pids)

    # Define identity loss and triplet loss
    # 交叉熵损失（id损失） + triplet 损失
    criterion_cla, criterion_pair = build_losses(config, dataset.num_train_clothes)

    # Build optimizer
    # 2+3 个 模型的参数一起做优化
    parameters = list(model.parameters()) + list(attention.parameters()) + list(gap_classifier.parameters()) + list(gap_classifier_h.parameters()) + list(gap_classifier_b.parameters())

    # Print number of parameters of the model
    logger.info("Total Model size: {:.5f}M".format(sum(p.numel() for p in parameters) / 1000000.0))

    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=1)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR,
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9,
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))

    # Build lr_scheduler
    scheduler = LambdaLR(optimizer, lr_lambda)

    start_epoch = config.TRAIN.START_EPOCH

    if config.MODEL.RESUME:
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model_state_dict'])
        attention.load_state_dict(checkpoint['attention_state_dict'])
        gap_classifier.load_state_dict(checkpoint['gap_classifier_state_dict'])
        gap_classifier_h.load_state_dict(checkpoint['gap_classifier_h_state_dict'])
        gap_classifier_b.load_state_dict(checkpoint['gap_classifier_b_state_dict'])

    # 五个模型的定义
    model = model.cuda(rank)
    attention = attention.cuda(rank)
    gap_classifier = gap_classifier.cuda(rank)
    gap_classifier_h = gap_classifier_h.cuda(rank)
    gap_classifier_b = gap_classifier_b.cuda(rank)

    model = nn.DataParallel(model, device_ids=[rank])
    attention = nn.DataParallel(attention, device_ids=[rank])
    gap_classifier = nn.DataParallel(gap_classifier, device_ids=[rank])
    gap_classifier_h = nn.DataParallel(gap_classifier_h, device_ids=[rank])
    gap_classifier_b = nn.DataParallel(gap_classifier_b, device_ids=[rank])

    if config.EVAL_MODE:
        logger.info("Evaluate only")
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc(config, 
                        model, attention, 
                        gap_classifier, gap_classifier_h, gap_classifier_b,
                        queryloader_same, queryloader_diff, 
                        galleryloader, dataset)
            else:
                test(config, 
                    model, attention, 
                    gap_classifier, gap_classifier_h, gap_classifier_b, 
                    queryloader,
                    galleryloader, dataset)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    logger.info("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        train_sampler.set_epoch(epoch)
        start_train_time = time.time()
        train_model(config, epoch, 
            model, attention, 
            gap_classifier, gap_classifier_h, gap_classifier_b,
            criterion_cla, criterion_pair, 
            optimizer, trainloader)
        train_time += round(time.time() - start_train_time)
        if (epoch + 1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
                (epoch + 1) % config.TEST.EVAL_STEP == 0 or (epoch + 1) == config.TRAIN.MAX_EPOCH:
            logger.info("==> Test")
            torch.cuda.empty_cache()
            if config.DATA.DATASET == 'prcc':
                rank1 = test_prcc(config, model, attention, 
                            gap_classifier, gap_classifier_h, gap_classifier_b,
                            queryloader_same, queryloader_diff, 
                            galleryloader, dataset)
            else:
                rank1 = test(config, model, attention, 
                            gap_classifier, gap_classifier_h, gap_classifier_b, 
                            queryloader,
                            galleryloader, dataset)

            if rank == 0:
                is_best = rank1 > best_rank1
                if is_best:
                    best_rank1 = rank1
                    best_epoch = epoch + 1

                model_state_dict = model.module.state_dict()
                attention_state_dict = attention.module.state_dict()
                gap_classifier_state_dict = gap_classifier.module.state_dict()
                gap_classifier_h_state_dict = gap_classifier_h.module.state_dict()
                gap_classifier_b_state_dict = gap_classifier_b.module.state_dict()

                save_checkpoint({
                    'model_state_dict': model_state_dict,
                    'attention_state_dict': attention_state_dict,
                    'gap_classifier_state_dict': gap_classifier_state_dict,
                    'gap_classifier_h_state_dict': gap_classifier_h_state_dict,
                    'gap_classifier_b_state_dict': gap_classifier_b_state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(config.OUTPUT, config.DATA.DATASET + str(epoch + 1) + '.pth.tar'))
                # osp.join(config.OUTPUT, 'prcc_' + str(epoch + 1) + '.pth.tar')

        scheduler.step()

    logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    config = parse_option()
    port = config.port
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    world_size = 2
    processes = []
    # Create Process Group
    for rank in range(world_size):
        p = Process(target=main, args=(rank, world_size, config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
