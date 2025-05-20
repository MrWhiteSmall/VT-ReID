import os
from PIL import Image
import time
import argparse
import logging
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process
from torch import distributed as dist
from torch.utils.data import DataLoader

from configs.default_img import get_img_config

import data.img_transforms as T
from data.datasets.ltcc import LTCC
from data.dataset_loader import  ImageDataset_test
from data.samplers import  DistributedInferenceSampler

from models import build_model

from tools.utils import set_seed, get_logger
from tools.eval_metrics import evaluate, evaluate_with_clothes

'''
Traceback (most recent call last):
  File "/data1/lsj/anaconda/envs/ccreid/lib/python3.10/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/data1/lsj/anaconda/envs/ccreid/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/data1/lsj/git_workspace/SCNet/main_test.py", line 135, in main
    dataset = LTCC(root='/data1/lsj/ccdatasets/LTCC-res')
  File "/data1/lsj/git_workspace/SCNet/data/datasets/ltcc.py", line 31, in __init__
    self._check_before_run()
  File "/data1/lsj/git_workspace/SCNet/data/datasets/ltcc.py", line 67, in _check_before_run
    raise RuntimeError("'{}' is not available".format(self.dataset_dir))
RuntimeError: '/data1/lsj/ccdatasets/LTCC-res/LTCC_ReID' is not available

Traceback (most recent call last):
  File "/data1/lsj/git_workspace/SCNet/main_test.py", line 207, in <module>
    main(rank,config)
  File "/data1/lsj/git_workspace/SCNet/main_test.py", line 159, in main
    sampler=DistributedInferenceSampler(dataset.gallery),
  File "/data1/lsj/git_workspace/SCNet/data/samplers.py", line 202, in __init__
    num_replicas = dist.get_world_size()
  File "/data1/lsj/anaconda/envs/ccreid/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 1196, in get_world_size
    return _get_group_size(group)
  File "/data1/lsj/anaconda/envs/ccreid/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 576, in _get_group_size
    default_pg = _get_default_group()
  File "/data1/lsj/anaconda/envs/ccreid/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 707, in _get_default_group
    raise RuntimeError(
RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.
'''
'''
1， 修改数据集
2， 修改best.pth.tar 路径
'''
best_ckp = '/data/lsj/OOTDiffusion/SCNet/best_ltcc.pth.tar'
root='/data/lsj/3090'
dataset = LTCC(root)


# 监听的端口
port = '12901'
def parse_option():
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss')
    # Datasets
    parser.add_argument('--root',       type=str, help="your root path to data directory")
    parser.add_argument('--port','-p',  type=str,default='12905', help="e.g. 12902,12903...")
    parser.add_argument('--dataset',    type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")

    args, unparsed = parser.parse_known_args()
    # config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET) 已更改存储目录
    config = get_img_config(args)

    return config

def init_logger(config):
    output_file = osp.join(config.OUTPUT, 'log_test.log')
    rank = 0
    logger = get_logger(output_file, rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")
    return logger

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB') # numpy数据
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        outputs.append(output[:num_total_examples])
    return outputs
@torch.no_grad()
def extract_img_feature(model, dataloader):
    avgpool = nn.AdaptiveAvgPool2d(1)
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()

        batch_features = model(imgs)
        batch_features = avgpool(batch_features).view(batch_features.size(0), -1)
        batch_features_flip = model(flip_imgs)
        batch_features_flip = avgpool(batch_features_flip).view(batch_features_flip.size(0), -1)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)
        features.append(batch_features.cpu())

        pids        = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids      = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)

    features = torch.cat(features, 0)
    return features, pids, camids, clothes_ids


def main_ltcc():
    config = parse_option()
    port = config.port
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    rank = 0

    dist.init_process_group("nccl", rank=rank, world_size=1)
    torch.cuda.set_device(rank)
    set_seed(config.SEED)
    # get logger
    logger = init_logger(config)

    # 1 model
    # 2 dataloader会产生的数据
    # 3 extract_img_feature提取特征
    # 4 compute_ap_cmc 计算预测准确度
    ##################### 1 model #############################
    # torch.load('resnet50_train_60_epochs-c8e5653e.pth.tar')
    # .pth.tar 可以直接加载模型
    # best_ckp = '/data1/lsj/ccdatasets/output/ltcc/best_model.pth.tar'
    model, attention= build_model(config)
    checkpoint = torch.load(best_ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    attention.load_state_dict(checkpoint['attention_state_dict'])
    model = model.cuda(rank)
    attention = attention.cuda(rank)
    model.eval()
    attention.eval()

    logger = logging.getLogger('reid.test')
    ##################### 2 dataloader会产生的数据 #############################
    # dataset = LTCC(root='/data1/lsj/ccdatasets') # 修改ltcc.py 中 目标数据文件夹
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    galleryloader = DataLoader(
        dataset=ImageDataset_test(dataset.gallery, transform=transform_test),
        sampler=DistributedInferenceSampler(dataset.gallery),
        batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=False, shuffle=False)
    queryloader = DataLoader(
        dataset=ImageDataset_test(dataset.query, transform=transform_test),
        sampler=DistributedInferenceSampler(dataset.query),
        batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=False, shuffle=False)

    ##################### 3 extract_img_feature提取特征 #############################
    qf, q_pids, q_camids, q_clothes_ids = extract_img_feature( model, queryloader)
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature( model, galleryloader)
    print('qf',qf.shape,'gf',gf.shape)

    qf, q_pids, q_camids, q_clothes_ids = \
        concat_all_gather([qf, q_pids, q_camids, q_clothes_ids],
                            len(dataset.query))
    gf, g_pids, g_camids, g_clothes_ids = \
        concat_all_gather([gf, g_pids, g_camids, g_clothes_ids],
                            len(dataset.gallery))
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    print('distmat',distmat.shape)
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine distance
    for i in range(m):
        distmat[i] = (-torch.mm(qf[i:i + 1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    

    ############### 4 compute_ap_cmc 计算预测准确度 #####################
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))


if __name__ == '__main__':
    main_ltcc()
