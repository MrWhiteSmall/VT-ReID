import copy
import os.path

import torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import data.img_transforms as T
import random
import numpy as np

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


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


def read_parsing_result(img_path):
    """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        super(ImageDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # clothes_id 是 将衣服映射到一个数值上
        # ltcc的图片名中 -c5 -c8，自带c标记，汇总后映射到id
        # prcc的图片名中 A，B是同一件衣服，C是另一件衣服 而ABC中都有sub dir
        #       比如092 048，将这些作为标记，映射到id
        # vccloth的图片名中包含pid和cloth，将pid+cloth得到唯一标记，
        #       汇总后再映射到id
        img_path, pid, camid, clothes_id = self.dataset[index]

        if 'LTCC_ReID' in img_path:
            parsing_result_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'processed',
                                               os.path.basename(img_path))

        elif 'PRCC' in img_path:
            # /public/home/yangzhe/ltt/lsj/ccdatasets/PRCC   /rgb               /298/A_cropped_rgb046.jpg
            # /public/home/yangzhe/ltt/lsj/ccdatasets/PRCC   /rgb/processed     /298/A_cropped_rgb046.png does not exist
            parsing_result_path = os.path.join('/'.join(img_path.split('/')[:-3]),
                                                'processed',
                                                img_path.split('/')[-2],
                                                img_path.split('/')[-1][:-4] + '.png'
                                               )

        elif 'VC-Clothes' in img_path:
            parsing_result_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'processed',
                                              img_path.split('/')[-1][:-4] + '.png')

        elif 'DeepChange' in img_path:
            parsing_result_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'processed',
                                               img_path.split('/')[-1][:-4] + '.png')


        # img_pah
        # 如果 img 在 _vt_res 中也存在，则替换img_b
        # id_clothid_camid_frame.png id_clothid_camid_frame_clothname.jpg.png
        # 这一操作，只会在train中生效
        ########################## 更换img_b ###################################
        img_b = None
        # train_vt_res_path = '/data/lsj/ccdatasets/LTCC_ReID/train_vt_res'
        # train_vt_res_path = '/data/lsj/ccdatasets/VC-Clothes/train_vt_res'
        # train_vt_res_names = os.listdir(train_vt_res_path)
        
        # imgname = img_path.split('/')[-1]
        # vtname = imgname.split('.')[0]
        # # print(vtname)
        # for n in train_vt_res_names:
        #     if n.startswith(vtname):
        #         img_b = read_image(os.path.join(train_vt_res_path,n))
        #         # print(vtname,'更换为',n)
        #         break
        #######################################################################
        img = read_image(img_path)
        parsing_result = read_parsing_result(parsing_result_path)

        p1 = random.randint(0, 1)
        p2 = random.randint(0, 1)
        p3 = random.randint(0, 1)

        transform = T.Compose([
            T.Resize((384, 192)),
            T.RandomCroping(p=p1),
            T.RandomHorizontalFlip(p=p2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(probability=p3)
        ])

        transform_b = T.Compose([
            T.Resize((384, 192)),
            T.RandomCroping(p=p1),
            T.RandomHorizontalFlip(p=p2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(probability=p3)
        ])

        transform_parsing = T.Compose([
            T.Resize((384, 192)),
            T.Convet_ToTensor(),
        ])

        parsing_result_copy = torch.tensor(np.asarray(parsing_result, dtype=np.uint8)).unsqueeze(0).repeat(3, 1, 1)
        # img_b = copy.deepcopy(img)
        # print(type(img)) # PIL.Image.Image类型
        img_arr = np.asarray(img,dtype=np.uint8)
        # 如果上面已经找到img_b的替代品了，就不需要这里的黑色了
        if img_b is None:
            img_b = img_arr.copy().transpose(2, 0, 1)
            # img_b = np.asarray(img_b, dtype=np.uint8)
            img_b[(parsing_result_copy == 2) | (parsing_result_copy == 3) | (parsing_result_copy == 4) | (
                    parsing_result_copy == 5) | (parsing_result_copy == 6) | (parsing_result_copy == 7) | (
                        parsing_result_copy == 10) | (
                        parsing_result_copy == 11)] = 0
            img_b = img_b.transpose(1, 2, 0)
            img_b = Image.fromarray(img_b, mode='RGB')

        img = transform(img)
        img_b = transform_b(img_b)
        parsing_result = transform_parsing(parsing_result)

        return parsing_result, img, img_b, pid, clothes_id


class ImageDataset_test(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        super(ImageDataset_test, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)
        img_test = np.array(img)
        # 0 255
        # print('数据集中 img',type(img_test),img_test.shape,img_test.min(),img_test.max())
        if self.transform is not None:
            img = self.transform(img)

        return img_path, img, pid, camid, clothes_id


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
