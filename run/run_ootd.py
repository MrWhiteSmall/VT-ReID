'''
python run/run_ootd.py --model_path run/examples/model/01008_00.jpg --cloth_path run/examples/garment/00055_00.jpg --scale 2.0 --sample 4

'''
'''
gpu 101 114 √
gpu 06 ×
'''
'''
    return forward_call(*args, **kwargs)
  File "/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion/ootd/pipelines_ootd/unet_vton_2d_condition.py", line 689, in forward
    sample, res_samples, spatial_attn_inputs, spatial_attn_idx = downsample_block(
  File "/public/home/yangzhe/miniconda3/envs/ootd/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion/ootd/pipelines_ootd/unet_vton_2d_blocks.py", line 429, in forward
    hidden_states, spatial_attn_inputs, spatial_attn_idx = attn(
  File "/public/home/yangzhe/miniconda3/envs/ootd/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion/ootd/pipelines_ootd/transformer_vton_2d.py", line 254, in forward
    hidden_states, spatial_attn_inputs, spatial_attn_idx = block(
  File "/public/home/yangzhe/miniconda3/envs/ootd/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion/ootd/pipelines_ootd/attention_vton.py", line 196, in forward
    if encoder_hidden_states:
RuntimeError: Boolean value of Tensor with more than one value is ambiguous

  File "/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion/ootd/pipelines_ootd/transformer_vton_2d.py", line 254, in forward
    hidden_states, spatial_attn_inputs, spatial_attn_idx = block(
  File "/public/home/yangzhe/miniconda3/envs/ootd201/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion/ootd/pipelines_ootd/attention_vton.py", line 321, in forward
    attn_output = self.attn2(
  File "/public/home/yangzhe/miniconda3/envs/ootd201/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/public/home/yangzhe/miniconda3/envs/ootd201/lib/python3.10/site-packages/diffusers/models/attention_processor.py", line 522, in forward
    return self.processor(
TypeError: AttnProcessor2_0.__call__() got an unexpected keyword argument 'save_qkv'
'''


from pathlib import Path
import sys

prefix = '/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion/run/'
import os
def join_path(*args):
    path = ''
    for a in args:
        path=os.path.join(path,a)
    return path

from PIL import Image

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
# 将项目地址放入 系统变量 中
sys.path.insert(0, str(PROJECT_ROOT))
'''
['/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion', 
'/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion/run', 
'/public/home/yangzhe/miniconda3/lib/python310.zip', 
'/public/home/yangzhe/miniconda3/lib/python3.10', 
'/public/home/yangzhe/miniconda3/lib/python3.10/lib-dynload', 
'/public/home/yangzhe/miniconda3/lib/python3.10/site-packages', 
'/public/home/yangzhe/pw/mmdetection']
'''
print(sys.path)

from utils_ootd import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_path', type=str, default="", required=True)
parser.add_argument('--cloth_path', type=str, default="", required=True)
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

model_type = args.model_type # "hd" or "dc"
category = args.category # 0:upperbody; 1:lowerbody; 2:dress
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


if __name__ == '__main__':

    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

    cloth_img = Image.open(cloth_path).resize((768, 1024))
    model_img = Image.open(model_path).resize((768, 1024))
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    print('保存图片')
    # masked_vton_img.save('./images_output/mask.jpg')
    masked_vton_img.save( join_path(prefix,'images_output/mask.jpg'))

    images = model(
        model_type='hd',
        category='upperbody',
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=4,
        num_steps=20,
        image_scale=2.0,
        seed=seed,
    )

    # image_idx = 0
    # for image in images:
    #     image.save( join_path(prefix, 'images_output/out3_' + model_type + '_' + str(image_idx) + '.png' )   )
    #     image_idx += 1
    images[0].save( join_path(prefix, 'images_output/out_' + model_type  + '.png' )   )

