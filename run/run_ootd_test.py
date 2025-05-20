'''
python run/run_ootd.py --model_path run/examples/model/01008_00.jpg --cloth_path run/examples/garment/00055_00.jpg --scale 2.0 --sample 4

'''
from pathlib import Path
import sys

prefix = '/data/lsj/OOTDiffusion/run/'
import os
import numpy as np
from os.path import join as osj

from PIL import Image

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
# 将项目地址放入 系统变量 中
sys.path.insert(0, str(PROJECT_ROOT))

from utils_ootd_test import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_path', type=str, default="", required=False)
parser.add_argument('--cloth_path', type=str, default="", required=False)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
args = parser.parse_args()


openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = 'dc' # "hd" or "dc"
# category = args.category # 0:upperbody; 1:lowerbody; 2:dress
cloth_path = args.cloth_path
model_path = args.model_path

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed

if model_type == "hd":
    model = OOTDiffusionHD(args.gpu_id)
elif model_type == "dc":
    model = OOTDiffusionDC(args.gpu_id)
else:
    raise ValueError("model_type must be \'hd\' or \'dc\'!")


################## 准备数据 #######################
# 循环获取cloth 和 model图片
# 文件路径 
"""
dataset_prefix = '/data/lsj/VITON-HD'
dataset_type='test'
# cloth 文件路径 : {dataset_prefix} {dataset_type} cloth xxx_00.jpg
# model 文件路径 : {dataset_prefix} {dataset_type} image xxx_00.jpg
pairs_file = osj(dataset_prefix,'test_pairs.txt')
cloth_names = []
model_names = []
with open(pairs_file,'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        # cloth_name,model_name = line.split(' ')
        model_name,cloth_name = line.split(' ')
        # print(cloth_name,model_name)
        model_names.append(model_name)
        cloth_names.append(cloth_name)

# 测试数据：12704_00.jpg 03697_00.jpg
test_model=lambda model_name:osj(dataset_prefix,dataset_type,'image',model_name)
test_cloth=lambda cloth_name:osj(dataset_prefix,dataset_type,'cloth',cloth_name)
# 最后存储在这个文件夹下  文件名为{cloth}_00_{model}_00.jpg
output_dir = '/public/home/yangzhe/ltt/lsj/OOTD_result'
output_name = lambda cloth,model:model.split('.')[0]+"_" \
                                +cloth.split('.')[0]+'.png'
"""
dataset_prefix = '/root/datasets/DressCode_1024'
pairs_file = '/root/datasets/DressCode_1024/train_pairs_230729.txt'
cloth_names = []
model_names = []
with open(pairs_file,'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        # cloth_name,model_name = line.split(' ')
        model_name,cloth_name,cloth_type = line.split(' ')
        if cloth_type=='upper' or cloth_type=='lower':
            continue
        model_name = model_name.replace('.png','.jpg')
        cloth_name = cloth_name.replace('.png','.jpg')
        model_names.append((model_name,cloth_type))
        cloth_names.append((cloth_name,cloth_type))
test_model=lambda model_name,cloth_type:osj(dataset_prefix,cloth_type,'image',model_name)
test_cloth=lambda cloth_name,cloth_type:osj(dataset_prefix,cloth_type,'cloth',cloth_name)
def get_save_keypoints_path(model_name,cloth_type):
    keypoints_prefix = osj(dataset_prefix,cloth_type,'openpose_json')
    os.makedirs(keypoints_prefix,exist_ok=True)
    save_name = model_name.replace('.jpg', '_keypoints.json')
    return osj(keypoints_prefix,save_name)
def get_save_parser_path(model_name,cloth_type):
    parser_prefix = osj(dataset_prefix,cloth_type,'image-parse-v3')
    os.makedirs(parser_prefix,exist_ok=True)
    save_name = model_name.replace('.jpg', '.png')
    return osj(parser_prefix,save_name)

import json
def save_dict_to_json(file_path,data):
    # 将字典保存为 JSON 文件
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def start_process(cloth_name,model_name):
    cloth_name,cloth_type = cloth_name
    model_name,cloth_type = model_name
    
    cloth_path = test_cloth(cloth_name,cloth_type)
    model_path = test_model(model_name,cloth_type)
    # 开始数据处理.......
    cloth_img = Image.open(cloth_path).resize((768, 1024))
    model_img = Image.open(model_path).resize((768, 1024))
    
    # 使用模型得到openpose和parse
    try:
        keypoints = openpose_model(model_img.resize((384, 512)))
    except Exception as e:
        return
    print(keypoints,type(keypoints)) # dict
    model_parse, _ = parsing_model(model_img.resize((384, 512)))
    
    '''save keypoints and parse'''
    '''model_name .json'''
    '''model_name .png'''
    keyponts_save_path = get_save_keypoints_path(model_name,cloth_type)
    parse_save_path = get_save_parser_path(model_name,cloth_type)
    
    save_dict_to_json(keyponts_save_path,keypoints)
    print(type(model_parse))            # <class 'PIL.Image.Image'>
    print(np.array(model_parse).shape)  # (512, 384)
    model_parse.save(parse_save_path)
    return 

    # 得到mask
    mask, mask_gray = get_mask_location(model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    mask.save(join_path(prefix,'images_output/mask-cloth.png'))
    
    return
    
    # 得到agnostic
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    masked_vton_img.save( join_path(prefix,'images_output/mask.jpg'))

    print('保存图片')

    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )

    image = images[0]
    image.save( join_path(output_dir,output_name(cloth_name,model_name)) )

if __name__ == '__main__':
    ##################### start ############################
    for cloth_name,model_name in zip(cloth_names,model_names):
        print(model_name,cloth_name)
        start_process(cloth_name,model_name)
        # break

    
