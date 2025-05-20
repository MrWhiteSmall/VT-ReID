from pathlib import Path
import sys,os,random,torch,cv2
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
# 将项目地址放入 系统变量 中
sys.path.insert(0, '/data/lsj/OOTDiffusion/SCNet')
sys.path.insert(0, '/data/lsj/OOTDiffusion/HRVITON')
sys.path.insert(0, '/data/lsj/OOTDiffusion/Densepose')
sys.path.insert(0, str(PROJECT_ROOT))
# print(sys.path)

prefix = '/data/lsj/OOTDiffusion/run'
cloth_path = '/data/lsj/VITON-HD/test/cloth/00064_00.jpg'
model_path = '/data/lsj/VITON-HD/test/image/00034_00.jpg'

def join_path(*args):
    path = ''
    for a in args:
        path=os.path.join(path,a)
    return path


from utils_ootd_test import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD

# from SCNet.main_test_ltcc import main_ltcc

####################### HR VITON ############################
from HRVITON.networks import ConditionGenerator, load_checkpoint, make_grid
from HRVITON.network_generator import SPADEGenerator

import torchgeometry as tgm
from collections import OrderedDict

import torchvision.transforms as transforms
transform = transforms.Compose([  \
        transforms.ToTensor(),   \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#################### Densepose #######################
from Densepose.test_run import get_densepose # 输入为 person path

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='run ootd')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
    # parser.add_argument('--model_path', type=str, default="", required=True)
    # parser.add_argument('--cloth_path', type=str, default="", required=True)
    parser.add_argument('--model_type', type=str, default="hd", required=False)
    parser.add_argument('--category', '-c', type=int, default=0, required=False)
    parser.add_argument('--scale', type=float, default=2.0, required=False)
    parser.add_argument('--step', type=int, default=20, required=False)
    parser.add_argument('--sample', type=int, default=4, required=False)
    parser.add_argument('--seed', type=int, default=-1, required=False)
    args = parser.parse_args()
    return args

def get_pose_parse_model():
    openpose_model = OpenPose(0)
    parsing_model = Parsing(0)
    return openpose_model,parsing_model


def get_opt():
    # 创建一个 Namespace 对象来存储默认参数
    opt = argparse.Namespace()

    # 设置默认值
    opt.gpu_ids = "0"
    opt.workers = 4
    opt.batch_size = 1
    opt.fp16 = False
    opt.cuda = True
    opt.test_name = 'test'
    opt.dataroot = "./HRVITON/data/zalando-hd-resize"
    opt.datamode = "test"
    opt.data_list = "test_pairs.txt"
    opt.output_dir = "./Output"
    opt.datasetting = "unpaired"
    opt.fine_width = 768
    opt.fine_height = 1024
    opt.tensorboard_dir = './HRVITON/data/zalando-hd-resize/tensorboard'
    opt.checkpoint_dir = 'checkpoints'
    opt.tocg_checkpoint = './HRVITON/eval_models/weights/v0.1/mtviton.pth'
    opt.gen_checkpoint = './HRVITON/eval_models/weights/v0.1/gen.pth'
    opt.tensorboard_count = 100
    opt.shuffle = False
    opt.semantic_nc = 13
    opt.output_nc = 13
    opt.gen_semantic_nc = 7
    opt.warp_feature = "T1"
    opt.out_layer = "relu"
    opt.clothmask_composition = 'warp_grad'
    opt.upsample = 'bilinear'
    opt.occlusion = True
    opt.norm_G = 'spectralaliasinstance'
    opt.ngf = 64
    opt.init_type = 'xavier'
    opt.init_variance = 0.02
    opt.num_upsampling_layers = 'most'

    return opt
def load_checkpoint_G(model, checkpoint_path,opt):
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    model.cuda()


import matplotlib.pyplot as plt
def save_parse_agnostic(image_parse_agnostic):
    # 将 PyTorch 张量转换为 numpy 数组以进行 Matplotlib 可视化
    input_parse_agnostic_np = image_parse_agnostic.detach().cpu().numpy()

    # 创建一个 4x4 的网格，展示每一层
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    # 将每一层显示在网格中
    for i, ax in enumerate(axes.flat):
        if i < input_parse_agnostic_np.shape[0]:  # 检查是否有足够的层
            ax.imshow(input_parse_agnostic_np[i], cmap='gray')
            ax.set_title(f"Layer {i+1}")
            ax.axis('off')  # 隐藏坐标轴
        else:
            ax.axis('off')  # 隐藏空的子图
    # 调整布局，避免子图重叠
    plt.tight_layout()

    # 保存图像到文件
    plt.savefig("save_parse_agnostic_layers.jpg")

def get_parse_agnostic(image_parse_agnostic,img_cm):
    opt = get_opt()
    
    labels = {
        0:  ['background',  [0, 10]],
        1:  ['hair',        [1, 2]],
        2:  ['face',        [3, 11]],
        3:  ['upper',       [4, 7]],
        4:  ['bottom',      [5, 6]],
        5:  ['left_arm',    [14]],
        6:  ['right_arm',   [15]],
        7:  ['left_leg',    [12]],
        8:  ['right_leg',   [13]],
        9:  ['left_shoe',   [9]],
        10: ['right_shoe',  [10]],
        11: ['socks',       []],
        12: ['noise',       [8, 16,17,18]]
    }

    semantic_nc, fine_height, fine_width = 13,opt.fine_height,opt.fine_width
    # load image-parse-agnostic
    # image_parse_agnostic = Image.open('parse.png')
    image_parse_agnostic = transforms.Resize(fine_width, interpolation=0)(image_parse_agnostic)
    image_parse_agnostic = np.array(image_parse_agnostic)

    # img_cm = Image.open('mask-cloth.png')
    img_cm = np.array( img_cm.resize((opt.fine_width,opt.fine_height)) )
    # print(img_cm.max()) # 255
    img_cm = img_cm.astype(np.uint8)
    image_parse_agnostic[img_cm==255]= 0

    # (512, 384)
    # print(image_parse_agnostic.shape)
    # print(np.unique(image_parse_agnostic))
    # plt.imshow(image_parse_agnostic,cmap='gray')

    # parse_agnostic = np.array(image_parse_agnostic)
    parse_agnostic = torch.from_numpy(image_parse_agnostic[None]).long()

    # torch.Size([1, 512, 384])
    # print(parse_agnostic.shape)
    # [ 0  2  4  6 11 14 15 18]
    # print(np.unique(parse_agnostic))
    parse_agnostic_map = torch.FloatTensor(19, fine_height, fine_width).zero_()
    parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
    new_parse_agnostic_map = torch.FloatTensor(semantic_nc, fine_height, fine_width).zero_()
    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_agnostic_map[i] += parse_agnostic_map[label]

    return new_parse_agnostic_map,img_cm
def make_grid(N, iH, iW,opt):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    if opt.cuda :
        grid = torch.cat([grid_x, grid_y], 3).cuda()
    else:
        grid = torch.cat([grid_x, grid_y], 3)
    return grid
def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], 
                                        seg_out[:, 5:, :, :]], dim=1)) \
                            .sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_vton_model2_tocg():
    opt = get_opt()
    ## Model
    # tocg
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = 13 + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)

    
    load_checkpoint(tocg, opt.tocg_checkpoint,opt)
    tocg.cuda()
    tocg.eval()
    
    output_dir  = os.path.join('./output', 'output')
    grid_dir    = os.path.join('./output', 'grid')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
      
    # tocg_checkpoint = './HRVITON/eval_models/weights/v0.1/mtviton.pth'
    # gen_checkpoint = './HRVITON/eval_models/weights/v0.1/gen.pth'
    
    return tocg
def get_vton_model2_generator():
    opt = get_opt()
    # generator
    opt.semantic_nc = 7
    generator = SPADEGenerator(opt, 3+3+3)
    load_checkpoint_G(generator, opt.gen_checkpoint,opt)
    generator.eval()
    
    return generator

def process_densepose(densepose):
    # 将图像转换为浮点型以便处理
    image = densepose.astype(np.float32) / 255.0

    # 设置背景为0 (背景颜色为 (48, 0, 58) 对应 [48/255, 0, 58/255])
    background_color = np.array([48/255, 0, 58/255])

    # 生成背景掩码，计算图像每个像素是否接近背景颜色
    background_mask = np.all(np.isclose(image, background_color, atol=0.05), axis=-1)

    # 将背景区域设为0
    image[background_mask] = 0    
    
    # 增亮非背景部分 (可以使用任意增强系数，比如增亮50%)
    bright_factor = 1.4
    image[~background_mask] = np.clip(image[~background_mask] * bright_factor, 0, 1)

    # 对每个颜色区域边缘进行平滑处理 (高斯模糊)
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigmaX=2, sigmaY=2)
    image[~background_mask] = blurred_image[~background_mask]

    # 将图像转换回 8 位格式以保存
    output_image = (image * 255).astype(np.uint8)

    # output_image = cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
    return output_image

    
def process_data(cloth_path,model_path):
    opt = get_opt()
    ################# 模型 ###########################
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()
    tocg  = get_vton_model2_tocg()
    generator = get_vton_model2_generator()
    
    ###################### person + cloth + cloth-mask ##########################
    cpath_split = cloth_path.split('/')
    cpath_split[-2] = 'cloth-mask'
    cm_path = '/'.join(cpath_split)
    mpath_split = model_path.split('/')
    mpath_split[-2] = 'image-densepose'
    densepose_path = '/'.join(mpath_split)
    
    # person image
    person = Image.open(model_path)
    person_copy = np.array(person).copy()
    person = transforms.Resize(opt.fine_width, interpolation=2)(person)
    person = transform(person)
    # cloth
    c = Image.open(cloth_path).convert('RGB')
    c.save('save_cloth.jpg')
    c = transforms.Resize(opt.fine_width, interpolation=2)(c)
    c = transform(c)
    # cloth-msk
    cloth_mask = Image.open(cm_path)
    cloth_mask.save('save_cloth_mask.jpg')
    cloth_mask = transforms.Resize(opt.fine_width, interpolation=2)(cloth_mask)
    cloth_mask = np.array(cloth_mask)
    cloth_mask = (cloth_mask >= 128).astype(np.float32)
    cloth_mask = torch.from_numpy(cloth_mask)  # [0,1]
    cloth_mask.unsqueeze_(0)
    
    
    
    person.unsqueeze_(0)
    c.unsqueeze_(0)
    cloth_mask.unsqueeze_(0)
    
    # 两个输入
    im              = person.cuda() # 没用 只用于grid画图 channel=3
    clothes         = c.cuda() # target cloth channel=3
    # 直接从数据集中拿
    pre_clothes_mask = torch.FloatTensor((cloth_mask.detach().numpy() > 0.5) \
                                            .astype(np.float32)).cuda() # unwarped cloth
    
    # 应当是 person torch.Size([1, 3, 1024, 768]) c torch.Size([1, 3, 1024, 768]) cm torch.Size([1, 1, 1024, 768])
    #       person torch.Size([1, 3, 1024, 768]) c torch.Size([1, 3, 1024, 768]) cm torch.Size([1, 1, 1024, 768])
    # print(f'person {person.shape} c {c.shape} cm {cloth_mask.shape}')
    ############################## input_parse_agnostic #################################
    
    # prefix = '/data/lsj/OOTDiffusion/run'
    # 只要将cloth path处理的步骤放到外面即可
    # args = get_args()
    target_shape = (opt.fine_width , opt.fine_height)
    
    # cloth_img = Image.open(cloth_path).resize(target_shape)
    model_img = Image.open(model_path).resize(target_shape)
    # 开始数据处理.......
    openpose_model,parsing_model = get_pose_parse_model()
    
    # 使用模型得到openpose和parse
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))
    
    # print(type(model_parse))            # <class 'PIL.Image.Image'>
    # print(np.array(model_parse).shape)  # (512, 384)
    # print(np.unique(model_parse))       # [ 0  2  4  6 11 14 15 18]

    # 得到mask
    mask, mask_gray = get_mask_location(model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
    # mask (768, 1024) mask_gray (768, 1024)
    # print(f'mask {mask.size} mask_gray {mask_gray.size}')
        
    # mask.save(join_path(prefix,'images_output/mask-cloth.png'))
    # 使用parse 和 mask 获得 parse_agnostic
    # get_parse_agnostic(model_parse,mask)
    
    # 使用humanparsing的结果处理而来 13*h*w
    # img_cm 用于处理 后续的 agnostic
    parse_agnostic,img_cm  = get_parse_agnostic(model_parse,mask)
    save_parse_agnostic(parse_agnostic)
    parse_agnostic.unsqueeze_(0)
    input_parse_agnostic = parse_agnostic.cuda()
    
    # parse agnostic torch.Size([1, 13, 1024, 768]) img_cm (1024, 768)
    # print(f'parse agnostic {parse_agnostic.shape} img_cm {img_cm.shape}')
    ############################# densepose #############################
    
    # densepose_map = Image.open(densepose_path)
    
    # 使用densepose模型处理而来
    densepose_test       = get_densepose(model_path) # numpy 
    densepose_test = process_densepose(densepose_test)
    # origin densepose (1024, 768, 3)  0  191
    # print('origin densepose',densepose_test.shape,densepose_test.min(),densepose_test.max())
    densepose_test = Image.fromarray(densepose_test)
    
    # densepose_test = densepose_map
    densepose_test.save('save_densepose.jpg')
    densepose_map = transforms.Resize(opt.fine_width, interpolation=2)(densepose_test)
    densepose_map = transform(densepose_map)  # [-1,1]
    densepose     = densepose_map.cuda()
    
    densepose.unsqueeze_(0)
    # densepose torch.Size([1, 3, 1024, 768])
    # print(f'densepose {densepose.shape}')
    ############################ agnostic ################################
    person_copy[img_cm==255] = (127,127,127)
    # person_copy (1024, 768, 3) 0 247
    # print('person_copy',person_copy.shape,person_copy.min(),person_copy.max())
    # img_cm (1024, 768) 0 255
    # print('img_cm',img_cm.shape,img_cm.min(),img_cm.max())
    agnostic = person_copy
    agnostic = Image.fromarray(agnostic)
    agnostic.save('save_agnostic.jpg')
    agnostic = transforms.Resize(opt.fine_width, interpolation=2)(agnostic)
    agnostic = transform(agnostic).cuda()
    
    agnostic = agnostic.unsqueeze_(0)
    # agnostic torch.Size([1, 3, 1024, 768])
    # print(f'agnostic {agnostic.shape}')
    ############################## 综上 可以获得 #####################
    '''
    im clothes pre_clothes_mask
    input_parse_agnostic
    densepose
    
    额外有
    person_copy img_cm => agnoistic
    '''
    clothes_down            = F.interpolate(clothes, size=(256, 192), mode='bilinear')
    pre_clothes_mask_down   = F.interpolate(pre_clothes_mask, size=(256, 192), mode='nearest')
    input_parse_agnostic_down = F.interpolate(input_parse_agnostic, size=(256, 192), mode='nearest')
    densepose_down          = F.interpolate(densepose, size=(256, 192), mode='bilinear')
    
    # tocg 就不反传梯度了吧.......
    input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1) # 3 + 1
    input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1) # 13 + 3
    
    
    # conditon forward
    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired \
                                                = tocg(opt,input1, input2)
    warped_cm_onehot = \
        torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5) \
                                            .astype(np.float32)).cuda() 
    
    cloth_mask = torch.ones_like(fake_segmap)
    cloth_mask[:,3:4, :, :] = warped_clothmask_paired
    fake_segmap = fake_segmap * cloth_mask   

    fake_parse_gauss = gauss(F.interpolate(fake_segmap, 
                                            size=(opt.fine_height, opt.fine_width), 
                                            mode='bilinear'))
    fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]
    old_parse = torch.FloatTensor(fake_parse.size(0), 13, 
                                    opt.fine_height, 
                                    opt.fine_width).zero_().cuda()
    old_parse.scatter_(1, fake_parse, 1.0)
    
    labels = {
        0:  ['background',  [0]],
        1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
        2:  ['upper',       [3]],
        3:  ['hair',        [1]],
        4:  ['left_arm',    [5]],
        5:  ['right_arm',   [6]],
        6:  ['noise',       [12]]
    }
    parse = torch.FloatTensor(fake_parse.size(0), 7, 
                                opt.fine_height, 
                                opt.fine_width).zero_().cuda()
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse[:, i] += old_parse[:, label]
            
    N, _, iH, iW = clothes.shape
    flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
    flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
        
    grid = make_grid(N, iH, iW,opt)
    warped_grid = grid + flow_norm
    warped_cloth = F.grid_sample(clothes, warped_grid, padding_mode='border')
    warped_clothmask = F.grid_sample(pre_clothes_mask, warped_grid, padding_mode='border')
    # occlusion
    warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), 
                                        warped_clothmask)
    warped_cloth = warped_cloth * warped_clothmask \
                    + torch.ones_like(warped_cloth) * (1-warped_clothmask)
    # 将 warped_cloth[0] 的值从 [-1, 1] 范围转换为 [0, 1] 范围
    image_tensor = warped_cloth[0].cpu().detach() / 2 + 0.5
    # 将 PyTorch 张量转换为 numpy 数组，并且确保值在 [0, 1] 之间
    image_numpy = image_tensor.permute(1, 2, 0).numpy()  # 转换为 HxWxC 格式
    # 将浮点数 numpy 数组转换为 8位图像 (0-255)
    image_numpy = (image_numpy * 255).astype(np.uint8)
    # 使用 PIL 保存图像
    image_pil = Image.fromarray(image_numpy)
    image_pil.save("save_warped_cloth.jpg")

    # output  torch.Size([1, 3, 1024, 768]) 
    # tensor(-0.9987, device='cuda:0', grad_fn=<MinBackward1>) 
    # tensor(0.9556, device='cuda:0', grad_fn=<MaxBackward1>)
    # print('output ',output.shape,output.min(),output.max())
    
    # ouput就是latent 格式？
    # save
    
    ########################### 1 output 已经处于归一化 #####################
    output = generator(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse) # 3+3+3
    # 不同于ootd 这里只会返回一个 候选答案
    
    
    ########################### 2 tensor 还原rgb #####################
    tensor = (output[0].clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    
    
    ########################### 3 array 保存图片 #####################
        
    array = tensor.detach().numpy().astype('uint8')
    # array  (3, 1024, 768) 0 249
    # print('array ',array.shape,array.min(),array.max())
    array = array.swapaxes(0, 1).swapaxes(1, 2)
    # array  (1024, 768, 3) 0 248
    # print('array ',array.shape,array.min(),array.max())
    im = Image.fromarray(array)
    im.save('./output.jpg', format='JPEG')

    return output
        

def get_vton_model(gpu_id):
    model = OOTDiffusionHD(gpu_id)
    return model

# 这里只返回 预处理的东西，不返回预测结果
def get_vton_res(cloth_path,model_path,save=False,no_grad = False):
    prefix = '/data/lsj/OOTDiffusion/run'
    # 只要将cloth path处理的步骤放到外面即可
    args = get_args()
    # image_scale = args.scale
    # n_steps = args.step
    # n_samples = args.sample
    seed = args.seed
    
    # target_shape = (768, 1024)
    # target_shape = (768//2, 1024//2)
    # target_shape = (768//4, 1024//4)
    target_shape = (192 , 384)

    with torch.no_grad():
        openpose_model,parsing_model = get_pose_parse_model(args)
        # model = get_vton_model(args)

        cloth_img = Image.open(cloth_path).resize(target_shape)
        model_img = Image.open(model_path).resize(target_shape)
        keypoints = openpose_model(model_img.resize((768//2, 1024//2)))
        model_parse, _ = parsing_model(model_img.resize((768//2, 1024//2)))

        mask, mask_gray = get_mask_location('hd', 'upper_body', model_parse, keypoints)
        mask = mask.resize(target_shape, Image.NEAREST)
        mask_gray = mask_gray.resize(target_shape, Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, model_img, mask)
        print('保存图片 start',cloth_path,model_path)
        if save:
            masked_vton_img.save( join_path(prefix,'images_output/mask.jpg'))

    print('保存图片 end',cloth_path,model_path)
    # if save:
    #     images[0].save( join_path(prefix, 'images_output/out_' + 'hd'  + '.png' )   )

    return cloth_img,masked_vton_img,mask,model_img

def get_vton_res2(cloth_path,model_path,save=False,no_grad = False):
    opt = get_opt()
    ################# 模型 ###########################
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()
    tocg  = get_vton_model2_tocg()
    
    ###################### person + cloth + cloth-mask ##########################
    cpath_split = cloth_path.split('/')
    cpath_split[-2] = 'cloth-mask'
    cm_path = '/'.join(cpath_split)
    mpath_split = model_path.split('/')
    mpath_split[-2] = 'image-densepose'
    densepose_path = '/'.join(mpath_split)
    
    # person image
    person = Image.open(model_path)
    person_ori = np.array(person).copy()
    person_copy = np.array(person).copy()
    person = transforms.Resize(opt.fine_width, interpolation=2)(person)
    person = transform(person)
    # cloth
    c_ori = Image.open(cloth_path).convert('RGB')
    c_ori.save('save_cloth.jpg')
    c = transforms.Resize(opt.fine_width, interpolation=2)(c_ori)
    c = transform(c)
    # cloth-msk
    cloth_mask = Image.open(cm_path)
    cloth_mask.save('save_cloth_mask.jpg')
    cloth_mask = transforms.Resize(opt.fine_width, interpolation=2)(cloth_mask)
    cloth_mask = np.array(cloth_mask)
    cloth_mask = (cloth_mask >= 128).astype(np.float32)
    cloth_mask = torch.from_numpy(cloth_mask)  # [0,1]
    cloth_mask.unsqueeze_(0)
    
    person.unsqueeze_(0)
    c.unsqueeze_(0)
    cloth_mask.unsqueeze_(0)
    
    # 两个输入
    im              = person.cuda() # 没用 只用于grid画图 channel=3
    clothes         = c.cuda() # target cloth channel=3
    # 直接从数据集中拿
    pre_clothes_mask = torch.FloatTensor((cloth_mask.detach().numpy() > 0.5) \
                                            .astype(np.float32)).cuda() # unwarped cloth
    
    # 应当是 person torch.Size([1, 3, 1024, 768]) c torch.Size([1, 3, 1024, 768]) cm torch.Size([1, 1, 1024, 768])
    #       person torch.Size([1, 3, 1024, 768]) c torch.Size([1, 3, 1024, 768]) cm torch.Size([1, 1, 1024, 768])
    # print(f'person {person.shape} c {c.shape} cm {cloth_mask.shape}')
    ############################## input_parse_agnostic #################################
    
    # prefix = '/data/lsj/OOTDiffusion/run'
    # 只要将cloth path处理的步骤放到外面即可
    # args = get_args()
    target_shape = (opt.fine_width , opt.fine_height)
    
    # cloth_img = Image.open(cloth_path).resize(target_shape)
    model_img = Image.open(model_path).resize(target_shape)
    # 开始数据处理.......
    openpose_model,parsing_model = get_pose_parse_model()
    
    # 使用模型得到openpose和parse
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))
    
    # print(type(model_parse))            # <class 'PIL.Image.Image'>
    # print(np.array(model_parse).shape)  # (512, 384)
    # print(np.unique(model_parse))       # [ 0  2  4  6 11 14 15 18]

    # 得到mask
    mask, mask_gray = get_mask_location(model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
    # mask (768, 1024) mask_gray (768, 1024)
    # print(f'mask {mask.size} mask_gray {mask_gray.size}')
        
    # mask.save(join_path(prefix,'images_output/mask-cloth.png'))
    # 使用parse 和 mask 获得 parse_agnostic
    # get_parse_agnostic(model_parse,mask)
    
    # 使用humanparsing的结果处理而来 13*h*w
    # img_cm 用于处理 后续的 agnostic
    parse_agnostic,img_cm  = get_parse_agnostic(model_parse,mask)
    save_parse_agnostic(parse_agnostic)
    parse_agnostic.unsqueeze_(0)
    input_parse_agnostic = parse_agnostic.cuda()
    
    # parse agnostic torch.Size([1, 13, 1024, 768]) img_cm (1024, 768)
    # print(f'parse agnostic {parse_agnostic.shape} img_cm {img_cm.shape}')
    ############################# densepose #############################
    
    # densepose_map = Image.open(densepose_path)
    
    # 使用densepose模型处理而来
    densepose_ori       = get_densepose(model_path) # numpy 
    densepose_ori = process_densepose(densepose_ori)
    # origin densepose (1024, 768, 3)  0  191
    # print('origin densepose',densepose_test.shape,densepose_test.min(),densepose_test.max())
    densepose_test = Image.fromarray(densepose_ori)
    
    # densepose_test = densepose_map
    densepose_test.save('save_densepose.jpg')
    densepose_map = transforms.Resize(opt.fine_width, interpolation=2)(densepose_test)
    densepose_map = transform(densepose_map)  # [-1,1]
    densepose     = densepose_map.cuda()
    
    densepose.unsqueeze_(0)
    # densepose torch.Size([1, 3, 1024, 768])
    # print(f'densepose {densepose.shape}')
    ############################ agnostic ################################
    person_copy[img_cm==255] = (127,127,127)
    # person_copy (1024, 768, 3) 0 247
    # print('person_copy',person_copy.shape,person_copy.min(),person_copy.max())
    # img_cm (1024, 768) 0 255
    # print('img_cm',img_cm.shape,img_cm.min(),img_cm.max())
    agnostic_ori = person_copy
    
    # 将人物以外的部分都设置成白色
    # 将 densepose 的人物以外部分在 agnostic 中设置成白色
    # densepose_ori <class 'numpy.ndarray'> (1024, 768, 3) 0 255
    # agnostic_ori <class 'numpy.ndarray'> (1024, 768, 3) 0 255
    print(f'densepose_ori {type(densepose_ori)} {densepose_ori.shape} {densepose_ori.min()} {densepose_ori.max()}')
    print(f'agnostic_ori {type(agnostic_ori)} {agnostic_ori.shape} {agnostic_ori.min()} {agnostic_ori.max()}')
    
    # 假设 densepose 中背景是 (48, 0, 58)，这根据你之前的描述
    background_color = np.array([0, 0, 0])
    # 找出 densepose 中背景部分
    densepose_mask = np.all(densepose_ori == background_color, axis=-1)  # True 表示背景
    # densepose_background = np.all(densepose_ori != background_color, axis=-1)  # True 表示背景
    # 创建白色背景
    agnostic_white  = agnostic_ori.copy()
    agnostic_back   = person_ori.copy()
    agnostic_background  = np.zeros_like(agnostic_ori)
    agnostic_white[densepose_mask] = [255, 255, 255]  # 将背景区域设置为白色
    agnostic_background[densepose_mask] = agnostic_back[densepose_mask] # 生成出来的图片  要把背景再贴回去
    # agnostic_background (158425, 3)
    print('agnostic_background',agnostic_background.shape)
    # 保存或显示结果图像
    aaaa = Image.fromarray(agnostic_background)
    aaaa.save('save_agnostic_background.jpg')
    
    # agnostic = Image.fromarray(agnostic_ori)
    agnostic = Image.fromarray(agnostic_white)
    agnostic.save('save_agnostic.jpg')
    agnostic = transforms.Resize(opt.fine_width, interpolation=2)(agnostic)
    agnostic = transform(agnostic).cuda()
    
    agnostic = agnostic.unsqueeze_(0)
    # agnostic torch.Size([1, 3, 1024, 768])
    # print(f'agnostic {agnostic.shape}')
    ############################## 综上 可以获得 #####################
    '''
    im clothes pre_clothes_mask
    input_parse_agnostic
    densepose
    
    额外有
    person_copy img_cm => agnoistic
    '''
    clothes_down            = F.interpolate(clothes, size=(256, 192), mode='bilinear')
    pre_clothes_mask_down   = F.interpolate(pre_clothes_mask, size=(256, 192), mode='nearest')
    input_parse_agnostic_down = F.interpolate(input_parse_agnostic, size=(256, 192), mode='nearest')
    densepose_down          = F.interpolate(densepose, size=(256, 192), mode='bilinear')
    
    # tocg 就不反传梯度了吧.......
    input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1) # 3 + 1
    input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1) # 13 + 3
    
    # conditon forward
    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired \
                                                = tocg(opt,input1, input2)
    warped_cm_onehot = \
        torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5) \
                                            .astype(np.float32)).cuda() 
    
    cloth_mask = torch.ones_like(fake_segmap)
    cloth_mask[:,3:4, :, :] = warped_clothmask_paired
    fake_segmap = fake_segmap * cloth_mask   

    fake_parse_gauss = gauss(F.interpolate(fake_segmap, 
                                            size=(opt.fine_height, opt.fine_width), 
                                            mode='bilinear'))
    fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]
    old_parse = torch.FloatTensor(fake_parse.size(0), 13, 
                                    opt.fine_height, 
                                    opt.fine_width).zero_().cuda()
    old_parse.scatter_(1, fake_parse, 1.0)
    
    labels = {
        0:  ['background',  [0]],
        1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
        2:  ['upper',       [3]],
        3:  ['hair',        [1]],
        4:  ['left_arm',    [5]],
        5:  ['right_arm',   [6]],
        6:  ['noise',       [12]]
    }
    parse = torch.FloatTensor(fake_parse.size(0), 7, 
                                opt.fine_height, 
                                opt.fine_width).zero_().cuda()
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse[:, i] += old_parse[:, label]
            
    N, _, iH, iW = clothes.shape
    flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
    flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
        
    grid = make_grid(N, iH, iW,opt)
    warped_grid = grid + flow_norm
    warped_cloth = F.grid_sample(clothes, warped_grid, padding_mode='border')
    warped_clothmask = F.grid_sample(pre_clothes_mask, warped_grid, padding_mode='border')
    # occlusion
    warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), 
                                        warped_clothmask)
    warped_cloth = warped_cloth * warped_clothmask \
                    + torch.ones_like(warped_cloth) * (1-warped_clothmask)
    # 将 warped_cloth[0] 的值从 [-1, 1] 范围转换为 [0, 1] 范围
    image_tensor = warped_cloth[0].cpu().detach() / 2 + 0.5
    # 将 PyTorch 张量转换为 numpy 数组，并且确保值在 [0, 1] 之间
    image_numpy = image_tensor.permute(1, 2, 0).numpy()  # 转换为 HxWxC 格式
    # 将浮点数 numpy 数组转换为 8位图像 (0-255)
    image_numpy = (image_numpy * 255).astype(np.uint8)
    # 使用 PIL 保存图像
    image_pil = Image.fromarray(image_numpy)
    image_pil.save("save_warped_cloth.jpg")

    # output  torch.Size([1, 3, 1024, 768]) 
    # tensor(-0.9987, device='cuda:0', grad_fn=<MinBackward1>) 
    # tensor(0.9556, device='cuda:0', grad_fn=<MaxBackward1>)
    # print('output ',output.shape,output.min(),output.max())
    
    # ouput就是latent 格式？
    # save
    

    return c_ori,densepose_mask,person_ori,agnostic, densepose, warped_cloth,parse

def get_vton_res_by_img(cloth_img,model_img,save=False):
    prefix = ''
    # 只要将cloth path处理的步骤放到外面即可
    args = get_args()
    image_scale = args.scale
    n_steps = args.step
    n_samples = args.sample
    seed = args.seed

    openpose_model,parsing_model = get_pose_parse_model(args)
    model = get_vton_model(args)

    # cloth_img = Image.open(cloth_path).resize((768, 1024))
    # model_img = Image.open(model_path).resize((768, 1024))
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask, mask_gray = get_mask_location('hd', 'upperbody', model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    print('保存图片 start')
    if save:
        masked_vton_img.save('./images_output/mask.jpg')
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

    print('保存图片 end')
    if save:
        images[0].save( join_path(prefix, 'images_output/out_' + 'hd'  + '.png' )   )

    return images[0]


# cloth_dir = '/public/backup/test04/lsj/datasets/VITON-HD/train/cloth'
# cloth_names = os.listdir(cloth_dir)
# def get_random_cloth_path():
#     random.shuffle(cloth_names)
#     cloth_path = os.path.join(cloth_dir,cloth_names[0])
#     return cloth_path

def try_main_vcc():
    from SCNet.main_test_vcc import main_vcc
    
    top1,top5,top10,top20,mAP = main_vcc(get_vton_model,get_vton_by_model=get_vton_res)
    # 传入vton-model，cloth-path，以及get-vton-res的方法
    # top1,top5,top10,top20,mAP = 0,0,0,0,0
    # count = 1
    # while top20 == 0 or count < 100:
    #     # cloth_path = get_random_cloth_path()
    #     top1,top5,top10,top20,mAP = main_vcc(cloth_path=cloth_path,get_vton_by_model=get_vton_res)
    #     count += 1

    # if count == 100:
    #     print('不太行，100次也没找到')
    # else:
    #     print('这件衣服比较好detect')
    #     print(cloth_path)
    #     print(top1,top5,top10,top20,mAP)

def try_main_vcc_hr():
    # from SCNet.main_test_vcc import main_vcc
    from SCNet.main_test_vcc import new_main_vcc as main_vcc
    
    top1,top5,top10,top20,mAP = main_vcc(get_vton_model2_generator,get_vton_by_model=get_vton_res2)

if __name__ == '__main__':
   
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # try_main_vcc()
    try_main_vcc_hr()
    
    # main_ltcc()

    # img,optimizer = get_vton_res(cloth_path=cloth_path,model_path=model_path,save=True,no_grad = False)
    # # 定义损失函数
    # criterion = nn.CrossEntropyLoss()
    
    # # 计算损失
    # loss = criterion(output, targets)

    # # 反向传播并更新小模型的参数
    # optimizer.zero_grad()  # 清空梯度
    # loss.backward()  # 计算梯度
    # optimizer.step()  # 更新小模型的参数
    
    ################################## HR VITON + Densepose #####################################
    # get_vton_model2()
    # process_data(cloth_path,model_path)
    

    # 打印当前显存分配情况
    print(f"当前显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")

    # 打印运行过程中最大的显存分配情况
    print(f"最大显存占用: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB")


# 使用 torch.cuda.memory_allocated 和 torch.cuda.max_memory_allocated
# torch.cuda.memory_allocated(): 返回当前在 GPU 上分配的显存量。
# torch.cuda.max_memory_allocated(): 返回运行过程中在 GPU 上分配的最大显存量。