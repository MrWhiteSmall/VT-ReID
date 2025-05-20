import os
import numpy as np
import cv2
import torch
import torchvision  # noqa: F401

from visualizer import End2EndVisualizer



def get_densepose(input = '00006_00.jpg'):
    model = '/data/lsj/DensePose-TorchScript-main/exported/densepose_rcnn_R_101_FPN_DL_s1x_fp32.pt'

    visualizer = End2EndVisualizer(alpha=.7, keep_bg=False)
    predictor = torch.jit.load(model).eval()


    device = torch.device("cuda")
    predictor = predictor.cuda()
    predictor = predictor.float()


    save_path = "_pred".join(os.path.splitext(input))
    img = cv2.imread(input)
    tensor = torch.from_numpy(img)

    outputs = predictor(tensor)

    test_img = np.zeros_like(img)

    image_vis = visualizer.visualize(test_img, outputs)

    image_vis = cv2.cvtColor(image_vis,cv2.COLOR_BGR2RGB)
    # cv2.imwrite(save_path, image_vis)
    # print(f"Image saved to {save_path}")
    return image_vis

if __name__ == '__main__':
    densepose = get_densepose(input = '00006_00.jpg')
    '''
    densepose (1024, 768, 3) [  0  21  30  34  39  40  41  42  43  44  47  51  52  58  61  63  64  65
    77  78  79  86  94  95 102 107 109 113 116 121 128 133 134 140 145 149
    151 154 156 159 161 172 191]
    '''
    print(f'densepose {densepose.shape} {np.unique(densepose[:,:,0])}')
    print(f'densepose {densepose.shape} {np.unique(densepose[:,:,1])}')
    print(f'densepose {densepose.shape} {np.unique(densepose[:,:,2])}')