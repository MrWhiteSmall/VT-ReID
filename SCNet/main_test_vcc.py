import os,random
from PIL import Image
import cv2
import argparse
import logging
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributed as dist
from torch.utils.data import DataLoader

from configs.default_img import get_img_config

import data.img_transforms as T
from data.datasets.ltcc import LTCC
from data.datasets.vcclothes import VCClothes
from data.dataset_loader import  ImageDataset_test
from data.samplers import  DistributedInferenceSampler

from models import build_model

from tools.utils import set_seed, get_logger
from tools.eval_metrics import evaluate, evaluate_with_clothes

'''
1， 修改数据集
2， 修改best.pth.tar 路径
'''
best_ckp = '/data/lsj/OOTDiffusion/SCNet/best_vcc.pth.tar'
root='/data/lsj/3090'
dataset = VCClothes(root)

# 监听的端口
port = '12901'
def parse_option():
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss')
    # Datasets
    parser.add_argument('--root',       type=str, help="your root path to data directory")
    parser.add_argument('--port','-p',  type=str,default='12905', help="e.g. 12902,12903...")
    parser.add_argument('--dataset',    type=str, default='vcclothes', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
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

cloth_dir = '/data/lsj/VITON-HD/train/cloth'
cloth_names = os.listdir(cloth_dir)
def get_random_cloth_path():
    random.shuffle(cloth_names)
    cloth_path = os.path.join(cloth_dir,cloth_names[0])
    return cloth_path

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

def concat_single_gather(tensor, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''

    tensor = tensor.cuda()
    tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor)
    output = torch.cat(tensors_gather, dim=0).cpu()
    output = output[:num_total_examples]
    return output
# @torch.no_grad()
def extract_img_feature(model, dataloader,one_batch=False):
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

        if one_batch:
            break
        print(batch_idx,flush=True)

    features = torch.cat(features, 0)
    return features, pids, camids, clothes_ids,None,None

mean = [0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
def denormalize(img):
    # 反归一化处理: (img * std) + mean
    return (img * std) + mean
# @torch.no_grad()
def extract_img_feature_by_vton(model,learnable_model, dataloader,vton_model=None,get_vton_by_model=None,one_batch=False,have_batch=None):
  
    transform_test = T.Compose([
        T.Normalize(mean=mean, std=std),
    ])

    avgpool = nn.AdaptiveAvgPool2d(1)
    features_ori = []
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    # 尝试直接取出第一个

    if have_batch is None:
        print('第一次生成！！')
        save_name = 'save_grid1.jpg'
        output_name = 'save_output1.jpg'

        cloth_path = get_random_cloth_path()
        
        for batch_idx, (img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
            if one_batch:
                break
        # 需要cloth path
        have_batch = (img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids,cloth_path)
        
    else:
        print('第二次生成！！老样子！')
        save_name = 'save_grid2.jpg'
        output_name = 'save_output2.jpg'
        
        img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids,cloth_path = have_batch
    
    imgs_ori = imgs.clone()
        
    # print(img_paths,imgs.shape) # torch.Size([1, 3, 384, 192] 所以 img需要重新读取，vton后resize到这个h w
    ############################# 重新读取img，cloth，vton后resize到384 192 ###############################
    # ori_batch,ori_channel,ori_h,ori_w = imgs.shape
    # 尝试 以 bacth的形式送入vton 模型 是否有用？不行的话，就循环batch每一个都替换掉，然后合并
    test_img = None
    prompt_loss = None
    vton_imgs = None
    # try:
        
        
    # 现在改成 返回预处理的东西了 并且 图片的分辨率仍然是 1024 * 768 
    cloth_img, densepose_mask ,person_ori,\
    agnostic, densepose, warped_cloth,parse = get_vton_by_model(cloth_path,img_paths[0],save=True,no_grad=False)
    
    # prompt loss
    # 待会补充 prompt loss
    prompt_loss = None
    agnostic_ori = agnostic.clone()
    agnostic = learnable_model(agnostic)
    prompt_loss = F.mse_loss(agnostic_ori,agnostic)
    print('prompt_loss',prompt_loss)
    images = vton_model(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse) # 3+3+3
    
    
    images = (images + 1) * 0.5 * 255
    images = images.clamp(0, 255)
    
    images = images.squeeze(0).permute(1,2,0)
    # images <class 'torch.Tensor'> torch.Size([1024, 768, 3]) 0.14753848314285278 251.09475708007812 True
    # print(f'person_ori {type(person_ori)} {person_ori.shape} {person_ori.min()} {person_ori.max()}')
    # images <class 'torch.Tensor'> torch.Size([1024, 768, 3]) 0.14753848314285278 251.09475708007812 True
    # print(f'images {type(images)} {images.shape} {images.min()} {images.max()} {images.requires_grad}')
    # person_ori[densepose_mask] <class 'numpy.ndarray'> (628007, 3) 0 255
    # print(f'person_ori[densepose_mask] {type(person_ori[densepose_mask])} {person_ori[densepose_mask].shape} {person_ori[densepose_mask].min()} {person_ori[densepose_mask].max()}')
    images[densepose_mask] = torch.tensor(person_ori[densepose_mask],device='cuda:0').float()
    images = images.clamp(0, 255)
    # images <class 'torch.Tensor'> torch.Size([1024, 768, 3]) 0.0 255.0 True
    # print(f'images {type(images)} {images.shape} {images.min()} {images.max()} {images.requires_grad}')
    
    images = images.permute(2,0,1).unsqueeze(0)
    # images <class 'torch.Tensor'> torch.Size([1, 3, 1024, 768]) 0.0 255.0 True
    # print(f'images {type(images)} {images.shape} {images.min()} {images.max()} {images.requires_grad}')
    # images <class 'torch.Tensor'> torch.Size([1, 3, 1024, 768]) 13.0 255.0 True
    # print(f'images {type(images)} {images.shape} {images.min()} {images.max()} {images.requires_grad}')
    
    ########################### 尝试 array 保存图片 #####################
    array = images[0].cpu().detach().numpy().astype('uint8')
    # array  (3, 1024, 768) 0 249
    # print('array ',array.shape,array.min(),array.max())
    array = array.swapaxes(0, 1).swapaxes(1, 2)
    # array  (1024, 768, 3) 0 248
    # print('array ',array.shape,array.min(),array.max())
    im = Image.fromarray(array)
    im.save(f'./{output_name}', format='JPEG')
    #####################################################################
    
    # 使用 torch.nn.functional.interpolate 进行 resize
    # F.interpolate( images <class 'torch.Tensor'> torch.Size([1, 3, 1024, 768]) 0.0 255.0 True
    # print(f'F.interpolate( images {type(images)} {images.shape} {images.min()} {images.max()} {images.requires_grad}')
    images = F.interpolate(images, size=(384, 192), mode='bilinear', align_corners=False)
    # F.interpolate( images <class 'torch.Tensor'> torch.Size([1, 3, 384, 192]) 0.26037904620170593 244.75003051757812 True
    # print(f'F.interpolate( images {type(images)} {images.shape} {images.min()} {images.max()} {images.requires_grad}')

    test_imgs = images
    test_img = images[0].unsqueeze(0) # 此处出来的就是torch # 3x256x192  batch是4，得只拿一个
    # print('vton后',type(test_img),test_img.shape,test_img.requires_grad)
    
    
    # except Exception as e:
    #     print(e)
        # print('reid，合成了图片',type(test_img),type(optimizer))
    
    # test_img 不取出 第一个，而是取出所有
    if test_img is not None:
        # test_img_res = test_img.resize((ori_w,ori_h)) # w h
        # test_img_np = np.array(test_img_res)
        # test_img_np = test_img_np.astype(np.float32)
        # print('vton img',test_img_np.shape)
        # test_img_tensor = torch.tensor(test_img_np).cuda()
        # print('vton img tensor',test_img_tensor.shape) # vton img tensor torch.Size([1024, 768, 3])
        # res_img_tensor = test_img_tensor.view(ori_channel,ori_h,ori_w).unsqueeze(0)
        # print('res vton img tensor',res_img_tensor.shape)
        # vton后还得找一下原来的img做了什么transform
        # test_imgs <class 'torch.Tensor'> torch.Size([1, 3, 384, 192]) 0.26037904620170593 244.75003051757812 True
        # print(f'test_imgs {type(test_imgs)} {test_imgs.shape} {test_imgs.min()} {test_imgs.max()} {test_imgs.requires_grad}')
        test_imgs = test_imgs / 255.0  # 将范围缩放到 [0, 1]
        normalized_imgs = transform_test(test_imgs)
        # imgs_ori tensor torch.Size([1, 3, 384, 192]) tensor(-1.9295) tensor(2.4134) False
        # normalized vton img tensor torch.Size([1, 3, 384, 192]) 
        #                   tensor(-0.6472, device='cuda:0', grad_fn=<MinBackward1>) 
        #                   tensor(1090.5984, device='cuda:0', grad_fn=<MaxBackward1>) True
        # normalized vton img tensor torch.Size([1, 3, 384, 192]) 
        #                   tensor(-2.1019, device='cuda:0', grad_fn=<MinBackward1>) 
        #                   tensor(2.3640, device='cuda:0', grad_fn=<MaxBackward1>) True
        # print('imgs_ori tensor',imgs_ori.shape,imgs_ori.min(),imgs_ori.max(),imgs_ori.requires_grad,)
        # print('normalized vton img tensor',normalized_imgs.shape,normalized_imgs.min(),normalized_imgs.max(),normalized_imgs.requires_grad,)
        
        # test_img 做了去归一化，不如不做？做一下吧，好像均值方差不太一样
        '''
        vton后 <class 'torch.Tensor'> torch.Size([4, 3, 256, 192]) True
        normalized vton img tensor torch.Size([4, 3, 256, 192]) True
        '''

        imgs = normalized_imgs.float() # 现在做了vton，形状不太一样了
        vton_imgs = torch.mean(imgs,dim=0).unsqueeze(0)
        
    #########################################################################################################
    ################################## 做一次vton后的feature #################################################
    # float tensor imgs torch.Size([1, 3, 384, 192]) True
    # print('float tensor imgs',imgs.shape,imgs.requires_grad)
    
    flip_imgs = torch.flip(imgs, [3])
    imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()

    def try_show(tensor,num):
        np_tensor = tensor.clone().permute(0,2,3,1).cpu().detach().numpy()
        d_np_tensor = denormalize(np_tensor)
        img_tensor = np.clip((d_np_tensor * 255), 0, 255).astype(np.uint8)
        img_tensor = img_tensor[0]
        img_tensor = Image.fromarray(img_tensor)
        img_tensor.save(f'./try{num}.jpg')

    # imgs   <class 'torch.Tensor'> torch.Size([1, 3, 384, 192]) -2.071012258529663 2.364039421081543
    print(f'imgs   {type(imgs)} {imgs.shape} {imgs.min()} {imgs.max()}')
    try_show(imgs,1)
    batch_features = model(imgs)
    batch_features = avgpool(batch_features).view(batch_features.size(0), -1)
    batch_features_flip = model(flip_imgs)
    batch_features_flip = avgpool(batch_features_flip).view(batch_features_flip.size(0), -1)
    batch_features += batch_features_flip
    batch_features = F.normalize(batch_features, p=2, dim=1)
    features.append(batch_features.cpu())
    ###########################################################################################################
    
    ################################## 做一次origin 的feature #################################################
    flip_imgs_ori = torch.flip(imgs_ori, [3])
    imgs_ori, flip_imgs_ori = imgs_ori.cuda(), flip_imgs_ori.cuda()

    # origin_img   <class 'torch.Tensor'> torch.Size([1, 3, 384, 192]) -1.9295316934585571 2.4134206771850586
    print(f'imgs_ori   {type(imgs_ori)} {imgs_ori.shape} {imgs_ori.min()} {imgs_ori.max()}')
    try_show(imgs_ori,2)
    batch_features_ori = model(imgs_ori)
    batch_features_ori = avgpool(batch_features_ori).view(batch_features_ori.size(0), -1)
    batch_features_flip_ori = model(flip_imgs_ori)
    batch_features_flip_ori = avgpool(batch_features_flip_ori).view(batch_features_flip_ori.size(0), -1)
    batch_features_ori += batch_features_flip_ori
    batch_features_ori = F.normalize(batch_features_ori, p=2, dim=1)
    features_ori.append(batch_features_ori.cpu())
    
    ###########################################################################################################
    
    # 计算第二个特征对的余弦相似度
    cosine_sim = F.cosine_similarity(vton_imgs, imgs_ori) # 这里应该是生成的图片 前后的 特征相似度，vton-img使用avg的方式获得特征
    sim_cosine_loss = 1 - cosine_sim.mean()  # 余弦相似度越高，loss越小，所以1减去它
    print('sim_cosine_loss',sim_cosine_loss)

    pids        = torch.cat((pids, batch_pids.cpu()), dim=0)
    camids      = torch.cat((camids, batch_camids.cpu()), dim=0)
    clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)

    features = torch.cat(features, 0)
    features_ori = torch.cat(features_ori, 0)
    
    ######################################################################################
    ################### 收集一下，原图 + 衣服 + vton-img  #################################
    ######################################################################################
    # 收集一下，原图 + 衣服 + vton-img
    # origin_img   <class 'torch.Tensor'> torch.Size([1, 3, 384, 192]) -1.9295316934585571 2.4134206771850586
    # vton_img    <class 'torch.Tensor'> torch.Size([1, 3, 384, 192]) -0.6472042202949524 1090.598388671875
    # print(f'origin_img   {type(imgs_ori)} {imgs_ori.shape} {imgs_ori.min()} {imgs_ori.max()}')
    # print(f'vton_img    {type(imgs)} {imgs.shape} {imgs.min()} {imgs.max()}')
    origin_img = imgs_ori.clone().permute(0,2,3,1).cpu().detach().numpy()
    target_cloth = np.array(cloth_img)
    # target_cloth <class 'numpy.ndarray'> (1024, 768, 3) 0 255
    # print(f'target_cloth {type(target_cloth)} {target_cloth.shape} {target_cloth.min()} {target_cloth.max()}')
    vton_img = imgs.clone().permute(0,2,3,1).cpu().detach().numpy()
    
    # 反归一化 origin_img 和 vton_imgs
    origin_img = denormalize(origin_img)  # 修改为适当的mean和std
    '''
    vton_img <class 'numpy.ndarray'> (1, 384, 192, 3) -2.071012258529663 2.364039421081543
    vton_img <class 'numpy.ndarray'> (1, 384, 192, 3) 0.06219348192215951 244.75715629577635
    '''
    # vton_img <class 'numpy.ndarray'> (1, 384, 192, 3) -1.5280283689498901 1090.630126953125
    # print(f'vton_img {type(vton_img)} {vton_img.shape} {vton_img.min()} {vton_img.max()}')
    vton_img = denormalize(vton_img)    # 修改为适当的mean和std
    # vton_img <class 'numpy.ndarray'> (1, 384, 192, 3) 0.062193616986274736 244.7571484375
    # print(f'vton_img {type(vton_img)} {vton_img.shape} {vton_img.min()} {vton_img.max()}')
    
    # origin_img <class 'numpy.ndarray'> (1, 384, 192, 3) 0.043137242197990366 0.9568627228736877
    # vton_img <class 'numpy.ndarray'> (1, 384, 192, 3) 0.26037905043363574 244.7500390625
    # print(f'origin_img {type(origin_img)} {origin_img.shape} {origin_img.min()} {origin_img.max()}')
    # print(f'vton_img {type(vton_img)} {vton_img.shape} {vton_img.min()} {vton_img.max()}')
    # 将反归一化后的图像值从浮点数转为整型像素值 [0, 255]
    origin_img = np.clip((origin_img * 255), 0, 255).astype(np.uint8)
    # vton_img = np.clip((vton_img), 0, 255).astype(np.uint8)
    vton_img = np.clip((vton_img * 255), 0, 255).astype(np.uint8)
    
    # 去掉 origin_img 的 batch 维度 (1, 384, 192, 3) -> (384, 192, 3)
    origin_img = origin_img[0]
    
    # 调整尺寸
    target_height = 512
    target_width = 384

    origin_img_resized = cv2.resize(origin_img, (target_width,target_height))
    target_cloth_resized = cv2.resize(target_cloth, (target_width,target_height))
    vton_img_resized = [cv2.resize(vton_img[i], (target_width,target_height))
                                                for i in range(vton_img.shape[0])]
    vton_img_resized = [ np.expand_dims(v, axis=0) for v in vton_img_resized]
    vton_img_resized = np.concatenate(vton_img_resized,axis=0)
    
    # origin_img_resized   <class 'numpy.ndarray'> (512, 384, 3)
    # origin_img_resized   0 255  
    # print(f'origin_img_resized   {type(origin_img_resized)} {origin_img_resized.shape}')
    # print(f'origin_img_resized   {origin_img_resized.min()} {origin_img_resized.max()}')
    # target_cloth_resized <class 'numpy.ndarray'> (512, 384, 3)
    # target_cloth_resized 0 254
    # print(f'target_cloth_resized {type(target_cloth_resized)} {target_cloth_resized.shape}')
    # print(f'target_cloth_resized {target_cloth_resized.min()} {target_cloth_resized.max()}')
    # vton_img_resized    <class 'numpy.ndarray'> (4, 512, 384, 3)
    # vton_img_resized    0 255
    # print(f'vton_img_resized    {type(vton_img_resized)} {vton_img_resized.shape}')
    # print(f'vton_img_resized    {vton_img_resized.min()} {vton_img_resized.max()}')
    # collections = (origin_img,target_cloth,vton_img)
    
    
    # 拼接图片: 两行三列，大小为 (2 * 384, 3 * 192)
    # 左边一列是 origin_img 和 target_cloth，右边是 vton_imgs 中的四张图片
    origin_img_resized = cv2.cvtColor(origin_img_resized,cv2.COLOR_BGR2RGB)
    vton_img_resized_tmp = cv2.cvtColor(vton_img_resized[0],cv2.COLOR_BGR2RGB)
    upper_row = np.hstack([origin_img_resized, target_cloth_resized, vton_img_resized_tmp])
    # upper_row = np.hstack([origin_img_resized, vton_img_resized[0], vton_img_resized[1]])
    # lower_row = np.hstack([target_cloth_resized, vton_img_resized[2], vton_img_resized[3]])
    # 垂直拼接两行
    # final_image = np.vstack([upper_row, lower_row])
    final_image = upper_row
    # 保存图片
    # if have_batch is None:
    #     save_name = 'first.jpg'
    # else:
    #     save_name = 'second.jpg'
    save_path = f'./{save_name}'
    cv2.imwrite(save_path, final_image)
    print(f"拼接图片已保存到: {save_path}")
    
    # # origin_img   <class 'numpy.ndarray'> (1, 384, 192, 3)
    # # origin_img   -1.9295316934585571 2.4134206771850586
    # print(f'origin_img   {type(origin_img)} {origin_img.shape}')
    # print(f'origin_img   {origin_img.min()} {origin_img.max()}')
    # # target_cloth <class 'numpy.ndarray'> (384, 192, 3)
    # # target_cloth 0 255
    # print(f'target_cloth {type(target_cloth)} {target_cloth.shape}')
    # print(f'target_cloth {target_cloth.min()} {target_cloth.max()}')
    # # vton_img    <class 'numpy.ndarray'> (4, 384, 192, 3)
    # # vton_img    -2.119140625 2.638671875
    # print(f'vton_img    {type(vton_img)} {vton_img.shape}')
    # print(f'vton_img    {vton_img.min()} {vton_img.max()}')
    # # collections = (origin_img,target_cloth,vton_img)
    
    return features, pids, camids, clothes_ids, features_ori,\
            prompt_loss,sim_cosine_loss,\
            have_batch, 
            


def main_vcc(get_vton_model = None,get_vton_by_model=None):
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
    
    ############## learnable ##################
    from SCNet.leanable_model import SmallModel
    learnable_model = SmallModel(channels=3).cuda()
    print('learnable_model 初始化',type(learnable_model))
    ## 注册hook，确保的确有反传的行为发生
    # 为 conv 层的参数注册 hook，用于打印反向传播的梯度
    def print_grad(grad):
        print("Gradient detected: ", grad.shape)
    for param in learnable_model.conv.parameters():
        param.register_hook(print_grad)

    # 1 model
    # 2 dataloader会产生的数据
    # 3 extract_img_feature提取特征
    # 4 compute_ap_cmc 计算预测准确度
    ##################### 1 model #############################
    # torch.load('resnet50_train_60_epochs-c8e5653e.pth.tar')
    # .pth.tar 可以直接加载模型
    model, attention= build_model(config)
    del attention
    checkpoint = torch.load(best_ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    # attention.load_state_dict(checkpoint['attention_state_dict'])
    model = model.cuda(rank)
    # attention = attention.cuda(rank)
    model.eval()
    # attention.eval()
    
    #########################################################

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
        batch_size=config.DATA.TEST_BATCH_gallery, num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=False, shuffle=False)
    queryloader = DataLoader(
        dataset=ImageDataset_test(dataset.query, transform=transform_test),
        sampler=DistributedInferenceSampler(dataset.query),
        batch_size=config.DATA.TEST_BATCH_query, num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=False, shuffle=False) # 有sampler时无法使用shuffle，互斥的

    ##################### 3 extract_img_feature提取特征 #############################
    # 获取vton model
    vton_model = get_vton_model()
    optimizer = optim.Adam(learnable_model.parameters(), lr=1e-3)
    
    gf, g_pids, g_camids, g_clothes_ids,_,_ = extract_img_feature( model, galleryloader,one_batch=True)
    gf, g_pids, g_camids, g_clothes_ids = \
        concat_all_gather([gf, g_pids, g_camids, g_clothes_ids],
                            len(dataset.gallery))
    gf = gf.cuda()
    
    # 将传入的vton等一系列内容给到qf的提取特征中，用于提取前 换衣 效果
    qfs, q_pids, q_camids, q_clothes_ids, features_ori,sim_prompt_loss,sim_cosine_loss,have_batch =  \
                                            extract_img_feature_by_vton( model,learnable_model, queryloader,
                                                                      vton_model = vton_model,
                                                                      get_vton_by_model=get_vton_by_model,
                                                                      one_batch=True)
    # qf, q_pids, q_camids, q_clothes_ids, features_ori,vton_model = extract_img_feature( model, queryloader,
    #                                                                   one_batch=True)
    
    qfs.unsqueeze_(1) # [4,1,2048]
    print('qfs',qfs.shape,'gf',gf.shape,'features_ori',features_ori.shape)
        
    m, n = qfs[0].size(0), gf.size(0)
    print('m,n',m,n)
        

    #################### 3.1 distmat是最终的预测矩阵 #######################
    if features_ori is not None:
        # 优化器只更新小模型的参数
        q_pids_ori, q_camids_ori, q_clothes_ids_ori = q_pids.clone(), q_camids.clone(), q_clothes_ids.clone()
        
        qf_ori, q_pids_ori, q_camids_ori, q_clothes_ids_ori = \
            concat_all_gather([features_ori, q_pids_ori, q_camids_ori, q_clothes_ids_ori],
                                len(dataset.query))
        qf_ori = qf_ori.cuda()
        
        distmat_ori = torch.zeros((m, n))
        # Cosine distance
        for i in range(m):
            distmat_ori[i] = (-torch.mm(qf_ori[i:i + 1], gf.t())).cpu()
        
        # print('distmat_ori',distmat_ori.shape,distmat_ori.size())
        
    ########################################################################
    
    ########################### 只取出loss最大的 qf #####################################
    ########################## 3.2 找出与最终预测矩阵，相差最大的矩阵 #######################
    
    
    q_pids = concat_single_gather(q_pids,len(dataset.query))
    q_camids = concat_single_gather(q_camids,len(dataset.query))
    q_clothes_ids = concat_single_gather(q_clothes_ids,len(dataset.query))
    
    distmat = None # 把最终选择的预测矩阵放在这里
    distmats = []
    for i in range(qfs.size(0)):
        qf = qfs[i]
        
        qf = concat_single_gather(qf, len(dataset.query))
       
        vton_distmat = torch.zeros((m, n))
        # print('vton_distmat',i,vton_distmat.shape)
        qf = qf.cuda()
        # Cosine distance
        for j in range(m):
            vton_distmat[j] = (-torch.mm(qf[j:j + 1], gf.t())).cpu()
        distmats.append(vton_distmat)
        
    # 计算每张图的 L1 loss
    losses = [F.l1_loss(d, distmat_ori, reduction='none').sum() for d in distmats]
    # 找到最大 loss 及其对应索引
    max_loss_idx = torch.argmax(torch.tensor(losses))
    pred_max_loss = losses[max_loss_idx]
    distmat = distmats[max_loss_idx]
    ########################################################################
    
    ########################## 3.3 计算loss ################################
        
    # 定义损失函数
    # criterion = nn.CrossEntropyLoss()
    # 计算第二个特征对的余弦相似度
    # cosine_sim = F.cosine_similarity(distmat, distmat_ori) # 这里应该是生成的图片 前后的 特征相似度，vton-img使用avg的方式获得特征
    # cosine_loss = 1 - cosine_sim.mean()  # 余弦相似度越高，loss越小，所以1减去它
    
    # 计算损失
    lambda_prompt_loss = 1 # 10
    lambda_cosine_loss = 1
    lambda_max_loss = 1
    loss = lambda_prompt_loss*sim_prompt_loss + lambda_cosine_loss*sim_cosine_loss + lambda_max_loss*pred_max_loss
    # loss = sim_prompt_loss
    # loss = sim_cosine_loss
    # loss = pred_max_loss

    # 反向传播并更新小模型的参数
    print('开始梯度反传')
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新小模型的参数
    print('结束梯度反传')
    
    ########################################################################
    
    
    distmat = distmat.detach().numpy() # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    

    ############### 4 compute_ap_cmc 计算预测准确度 #####################
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat_ori.detach().numpy(), q_pids, g_pids, q_camids, g_camids)

    logger.info("ori Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("Computing CMC and mAP")
    
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info("vton Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))


    #############################################################################################
    ########################## 再来生成一次！have_batch ##########################################
    ########################## gt 不动 ，不做反传，再来一次 ##########################################
    #############################################################################################
    # 将传入的vton等一系列内容给到qf的提取特征中，用于提取前 换衣 效果
    qfs, q_pids, q_camids, q_clothes_ids, features_ori,sim_prompt_loss,sim_cosine_loss,have_batch =  \
                                            extract_img_feature_by_vton( model,learnable_model, queryloader,
                                                                      vton_model = vton_model,
                                                                      get_vton_by_model=get_vton_by_model,
                                                                      one_batch=True,
                                                                      have_batch=have_batch)
    qfs.unsqueeze_(1) # [4,1,2048]
    # print('qfs',qfs.shape,'gf',gf.shape,'features_ori',features_ori.shape)
    #################### 3.1 distmat是最终的预测矩阵 #######################
        
    ########################################################################
    
    ########################### 只取出loss最大的 qf #####################################
    ########################## 3.2 找出与最终预测矩阵，相差最大的矩阵 #######################
        
    q_pids = concat_single_gather(q_pids,len(dataset.query))
    q_camids = concat_single_gather(q_camids,len(dataset.query))
    q_clothes_ids = concat_single_gather(q_clothes_ids,len(dataset.query))
    
    distmat = None # 把最终选择的预测矩阵放在这里
    distmats = []
    for i in range(qfs.size(0)):
        qf = qfs[i]
        
        qf = concat_single_gather(qf, len(dataset.query))
       
        vton_distmat = torch.zeros((m, n))
        # print('vton_distmat',i,vton_distmat.shape)
        qf = qf.cuda()
        # Cosine distance
        for j in range(m):
            vton_distmat[j] = (-torch.mm(qf[j:j + 1], gf.t())).cpu()
        distmats.append(vton_distmat)
        
    # 计算每张图的 L1 loss
    losses = [F.l1_loss(d, distmat_ori, reduction='none').sum() for d in distmats]
    # 找到最大 loss 及其对应索引
    max_loss_idx = torch.argmax(torch.tensor(losses))
    pred_max_loss = losses[max_loss_idx]
    distmat = distmats[max_loss_idx]


    distmat = distmat.detach().numpy() # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    ############### 4 compute_ap_cmc 计算预测准确度 #####################
    logger.info("Again ! Computing CMC and mAP")
    cmc, mAP = evaluate(distmat_ori.detach().numpy(), q_pids, g_pids, q_camids, g_camids)

    logger.info("ori Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("Computing CMC and mAP")
    
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info("vton Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))


    # 最后可以保存 原图 + vton 1图 + vton 2图
    return cmc[0], cmc[4], cmc[9], cmc[19], mAP


def new_main_vcc(get_vton_model = None,get_vton_by_model=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
    
    ############## learnable ##################
    from SCNet.leanable_model import SmallModel
    learnable_model = SmallModel(channels=3).cuda()
    print('learnable_model 初始化',type(learnable_model))
    ## 注册hook，确保的确有反传的行为发生
    # 为 conv 层的参数注册 hook，用于打印反向传播的梯度
    def print_grad(grad):
        print("Gradient detected: ", grad.shape)
    for param in learnable_model.conv.parameters():
        param.register_hook(print_grad)

    # 1 model
    # 2 dataloader会产生的数据
    # 3 extract_img_feature提取特征
    # 4 compute_ap_cmc 计算预测准确度
    ##################### 1 model #############################
    # torch.load('resnet50_train_60_epochs-c8e5653e.pth.tar')
    # .pth.tar 可以直接加载模型
    model, attention= build_model(config)
    del attention
    checkpoint = torch.load(best_ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda(rank)
    model.eval()
    
    #########################################################

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
        batch_size=config.DATA.TEST_BATCH_gallery, num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=False, shuffle=False)
    queryloader = DataLoader(
        dataset=ImageDataset_test(dataset.query, transform=transform_test),
        sampler=DistributedInferenceSampler(dataset.query),
        batch_size=config.DATA.TEST_BATCH_query, num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=False, shuffle=False) # 有sampler时无法使用shuffle，互斥的

    ##################### 3 extract_img_feature提取特征 #############################
    # 获取vton model
    vton_model = get_vton_model()
    optimizer = optim.Adam(learnable_model.parameters(), lr=1e-3)


 

    ####################### start query #######################################
    transform_test = T.Compose([
        T.Normalize(mean=mean, std=std),
    ])

    # features_ori = []
    # features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    for q_batch_idx, (img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(queryloader):
        cloth_path = get_random_cloth_path()
        for i in range(1,3):

            # start gallery 当前显存占用: 584.69 MB
            print(f"start gallery 当前显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
            ####################### start gallery #######################################
            avgpool = nn.AdaptiveAvgPool2d(1)
            features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])

            for g_batch_idx, (img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(galleryloader):
                flip_imgs = torch.flip(imgs, [3])
                imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()

                with torch.no_grad():
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
                # break
            
            features = torch.cat(features, 0)
            gf, g_pids, g_camids, g_clothes_ids = features,pids,camids,clothes_ids
            gf, g_pids, g_camids, g_clothes_ids = \
                concat_all_gather([gf, g_pids, g_camids, g_clothes_ids],
                                    len(dataset.gallery))
            gf = gf.cuda()
            g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
            ####################### end gallery #######################################
            # end gallery 当前显存占用: 35508.69 MB
            print(f"end gallery 当前显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")


            avgpool = nn.AdaptiveAvgPool2d(1)
            features_ori = []
            features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    
            save_name = f'save_grid{i}.jpg'
            output_name = f'save_output{i}.jpg'

            imgs_ori = imgs.clone()
            
            test_img = None
            prompt_loss = None
            vton_imgs = None
            
            cloth_img, densepose_mask ,person_ori,\
            agnostic, densepose, warped_cloth,parse = get_vton_by_model(cloth_path,img_paths[0],save=True,no_grad=False)
            # prompt loss
            prompt_loss = None
            agnostic_ori = agnostic.clone()
            agnostic = learnable_model(agnostic)
            prompt_loss = F.mse_loss(agnostic_ori,agnostic)
            print('prompt_loss',prompt_loss)
            images = vton_model(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse) # 3+3+3

            images = (images + 1) * 0.5 * 255
            images = images.clamp(0, 255)
            
            images = images.squeeze(0).permute(1,2,0)
            images[densepose_mask] = torch.tensor(person_ori[densepose_mask],device='cuda:0').float()
            images = images.clamp(0, 255)
            
            images = images.permute(2,0,1).unsqueeze(0)
            
            ########################### 尝试 array 保存图片 #####################
            array = images[0].cpu().detach().numpy().astype('uint8')
            array = array.swapaxes(0, 1).swapaxes(1, 2)
            im = Image.fromarray(array)
            im.save(f'./{output_name}', format='JPEG')
            #####################################################################
            
            images = F.interpolate(images, size=(384, 192), mode='bilinear', align_corners=False)
            
            test_imgs = images
            test_img = images[0].unsqueeze(0) # 此处出来的就是torch # 3x256x192  batch是4，得只拿一个
            
            test_imgs = test_imgs / 255.0  # 将范围缩放到 [0, 1]
            normalized_imgs = transform_test(test_imgs)
                
            imgs = normalized_imgs.float() # 现在做了vton，形状不太一样了
            vton_imgs = torch.mean(imgs,dim=0).unsqueeze(0)
            
            flip_imgs = torch.flip(imgs, [3])
            imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()

            def try_show(tensor,num):
                np_tensor = tensor.clone().permute(0,2,3,1).cpu().detach().numpy()
                d_np_tensor = denormalize(np_tensor)
                img_tensor = np.clip((d_np_tensor * 255), 0, 255).astype(np.uint8)
                img_tensor = img_tensor[0]
                img_tensor = Image.fromarray(img_tensor)
                img_tensor.save(f'./try{num}.jpg')
                
            print(f'imgs   {type(imgs)} {imgs.shape} {imgs.min()} {imgs.max()}')
            try_show(imgs,1)
            batch_features = model(imgs)
            batch_features = avgpool(batch_features).view(batch_features.size(0), -1)
            batch_features_flip = model(flip_imgs)
            batch_features_flip = avgpool(batch_features_flip).view(batch_features_flip.size(0), -1)
            batch_features += batch_features_flip
            batch_features = F.normalize(batch_features, p=2, dim=1)
            features.append(batch_features.cpu())
            
            ################################## 做一次origin 的feature #################################################
            flip_imgs_ori = torch.flip(imgs_ori, [3])
            imgs_ori, flip_imgs_ori = imgs_ori.cuda(), flip_imgs_ori.cuda()

            # origin_img   <class 'torch.Tensor'> torch.Size([1, 3, 384, 192]) -1.9295316934585571 2.4134206771850586
            print(f'imgs_ori   {type(imgs_ori)} {imgs_ori.shape} {imgs_ori.min()} {imgs_ori.max()}')
            try_show(imgs_ori,2)
            with torch.no_grad():
                batch_features_ori = model(imgs_ori)
                batch_features_ori = avgpool(batch_features_ori).view(batch_features_ori.size(0), -1)
                batch_features_flip_ori = model(flip_imgs_ori)
                batch_features_flip_ori = avgpool(batch_features_flip_ori).view(batch_features_flip_ori.size(0), -1)
            batch_features_ori += batch_features_flip_ori
            batch_features_ori = F.normalize(batch_features_ori, p=2, dim=1)
            features_ori.append(batch_features_ori.cpu())
            
            # 计算第二个特征对的余弦相似度
            cosine_sim = F.cosine_similarity(vton_imgs, imgs_ori) # 这里应该是生成的图片 前后的 特征相似度，vton-img使用avg的方式获得特征
            sim_cosine_loss = 1 - cosine_sim.mean()  # 余弦相似度越高，loss越小，所以1减去它
            print('sim_cosine_loss',sim_cosine_loss)

            pids        = torch.cat((pids, batch_pids.cpu()), dim=0)
            camids      = torch.cat((camids, batch_camids.cpu()), dim=0)
            clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)

            features = torch.cat(features, 0)
            features_ori = torch.cat(features_ori, 0)
            
            ######################################################################################
            ################### 收集一下，原图 + 衣服 + vton-img  #################################
            ######################################################################################
            # 收集一下，原图 + 衣服 + vton-img
            origin_img = imgs_ori.clone().permute(0,2,3,1).cpu().detach().numpy()
            target_cloth = np.array(cloth_img)
            vton_img = imgs.clone().permute(0,2,3,1).cpu().detach().numpy()
            
            # 反归一化 origin_img 和 vton_imgs
            origin_img = denormalize(origin_img)  # 修改为适当的mean和std
            vton_img = denormalize(vton_img)    # 修改为适当的mean和std
            # 将反归一化后的图像值从浮点数转为整型像素值 [0, 255]
            origin_img = np.clip((origin_img * 255), 0, 255).astype(np.uint8)
            vton_img = np.clip((vton_img * 255), 0, 255).astype(np.uint8)
            
            origin_img = origin_img[0]
            
            # 调整尺寸
            target_height = 512
            target_width = 384

            origin_img_resized = cv2.resize(origin_img, (target_width,target_height))
            target_cloth_resized = cv2.resize(target_cloth, (target_width,target_height))
            vton_img_resized = [cv2.resize(vton_img[i], (target_width,target_height))
                                                        for i in range(vton_img.shape[0])]
            vton_img_resized = [ np.expand_dims(v, axis=0) for v in vton_img_resized]
            vton_img_resized = np.concatenate(vton_img_resized,axis=0)
            
            origin_img_resized = cv2.cvtColor(origin_img_resized,cv2.COLOR_BGR2RGB)
            vton_img_resized_tmp = cv2.cvtColor(vton_img_resized[0],cv2.COLOR_BGR2RGB)
            upper_row = np.hstack([origin_img_resized, target_cloth_resized, vton_img_resized_tmp])
            final_image = upper_row
            
            save_path = f'./{save_name}'
            cv2.imwrite(save_path, final_image)
            print(f"拼接图片已保存到: {save_path}")
            
            qfs, q_pids, q_camids, q_clothes_ids, features_ori,\
            sim_prompt_loss,sim_cosine_loss = features, pids, camids, clothes_ids, features_ori,\
                                                prompt_loss,sim_cosine_loss,\
                                                    
            qfs.unsqueeze_(1) # [4,1,2048]
            print('qfs',qfs.shape,'gf',gf.shape,'features_ori',features_ori.shape)
                
            m, n = qfs[0].size(0), gf.size(0)
            print('m,n',m,n)
            
            #################### 3.1 distmat是最终的预测矩阵 #######################
            # 优化器只更新小模型的参数
            q_pids_ori, q_camids_ori, q_clothes_ids_ori = q_pids.clone(), q_camids.clone(), q_clothes_ids.clone()
            
            qf_ori, q_pids_ori, q_camids_ori, q_clothes_ids_ori = \
                concat_all_gather([features_ori, q_pids_ori, q_camids_ori, q_clothes_ids_ori],
                                    len(dataset.query))
            qf_ori = qf_ori.cuda()
            
            distmat_ori = torch.zeros((m, n))
            # Cosine distance
            for i in range(m):
                distmat_ori[i] = (-torch.mm(qf_ori[i:i + 1], gf.t())).cpu()
                
            ########################### 只取出loss最大的 qf #####################################
            ########################## 3.2 找出与最终预测矩阵，相差最大的矩阵 #######################
            q_pids = concat_single_gather(q_pids,len(dataset.query))
            q_camids = concat_single_gather(q_camids,len(dataset.query))
            q_clothes_ids = concat_single_gather(q_clothes_ids,len(dataset.query))
            
            distmat = None # 把最终选择的预测矩阵放在这里
            distmats = []
            for i in range(qfs.size(0)):
                qf = qfs[i]
                
                qf = concat_single_gather(qf, len(dataset.query))
            
                vton_distmat = torch.zeros((m, n))
                # print('vton_distmat',i,vton_distmat.shape)
                qf = qf.cuda()
                # Cosine distance
                for j in range(m):
                    vton_distmat[j] = (-torch.mm(qf[j:j + 1], gf.t())).cpu()
                distmats.append(vton_distmat)
                
            # 计算每张图的 L1 loss
            losses = [F.l1_loss(d, distmat_ori, reduction='none').sum() for d in distmats]
            # 找到最大 loss 及其对应索引
            max_loss_idx = torch.argmax(torch.tensor(losses))
            pred_max_loss = losses[max_loss_idx]
            distmat = distmats[max_loss_idx]
            
            lambda_prompt_loss = 1 # 10
            lambda_cosine_loss = 1
            lambda_max_loss = 1
            loss = lambda_prompt_loss*sim_prompt_loss + lambda_cosine_loss*sim_cosine_loss + lambda_max_loss*pred_max_loss
            # loss = lambda_prompt_loss*sim_prompt_loss + lambda_cosine_loss*sim_cosine_loss
            
            # before loss当前显存占用: 78863.57 MB
            # before loss当前显存占用: 44158.57 MB
            print(f"before loss当前显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")

        
            # 反向传播并更新小模型的参数
            print('开始梯度反传')
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新小模型的参数
            print('结束梯度反传')
            
            # after loss当前显存占用: 2050.46 MB
            # after loss当前显存占用: 2046.73 MB
            print(f"after loss当前显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
            
            
            distmat = distmat.detach().numpy() # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
            q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
            
            ############### 4 compute_ap_cmc 计算预测准确度 #####################
            if i==1:
                logger.info("Computing CMC and mAP")
            else:
                logger.info("Again ! Computing CMC and mAP")
            cmc, mAP = evaluate(distmat_ori.detach().numpy(), q_pids, g_pids, q_camids, g_camids)

            logger.info("ori Results ---------------------------------------------------")
            logger.info(
                'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            logger.info("Computing CMC and mAP")
            
            cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

            logger.info("vton Results ---------------------------------------------------")
            logger.info(
                'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))

        break
    
    ####################### end query #######################################
    
    # 最后可以保存 原图 + vton 1图 + vton 2图
    return cmc[0], cmc[4], cmc[9], cmc[19], mAP
        
if __name__ == '__main__':
    main_vcc()
