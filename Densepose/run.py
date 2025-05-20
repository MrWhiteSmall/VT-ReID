import argparse
import os
from itertools import count

import numpy as np
import cv2
import torch
import torchvision  # noqa: F401

import pickle

from visualizer import End2EndVisualizer

parser = argparse.ArgumentParser(description='Export DensePose model to TorchScript module')
parser.add_argument("model", type=str, help="Model file")
parser.add_argument("input", type=str, help="Input data")
parser.add_argument("--cpu", action="store_true", help="Only use CPU")
parser.add_argument("--fp32", action="store_true", help="Only use FP32")
args = parser.parse_args()
visualizer = End2EndVisualizer(alpha=.7, keep_bg=False)
predictor = torch.jit.load(args.model).eval()
# 从PKL文件中加载模型
# with open(args.model, 'rb') as f:
#     predictor = pickle.load(f)['model']
# predictor.eval()  # 设置模型为评估模式
    
# 打印预测器的类型和内容
'''
<class 'dict'>
dict_keys(['model', '__author__'])
'''
# print(type(predictor))  # 查看类型
# print(predictor.keys())  # 如果是字典，查看它包含哪些键


if torch.cuda.is_available() and not args.cpu:
    device = torch.device("cuda")
    predictor = predictor.cuda()
    if args.fp32:
        predictor = predictor.float()
    else:
        predictor = predictor.half()
else:
    device = torch.device("cpu")
    predictor = predictor.float()

save_path = "_pred".join(os.path.splitext(args.input))
if os.path.splitext(args.input)[1].lower() in [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
    img = cv2.imread(args.input)
    tensor = torch.from_numpy(img)

    outputs = predictor(tensor)
    # data = visualizer.extractor(outputs)
    # densepose_result, boxes_xywh = data
    
    # boxes_xywh = boxes_xywh.cpu().numpy()
    # for i, result in enumerate(densepose_result):
    #     iuv_array = torch.cat((result['labels'][None].type(torch.float32), result['uv'] * 255.0)).byte()
        
    #     matrix = _extract_i_from_iuvarr(iuv_arr)
    #     segm = _extract_i_from_iuvarr(iuv_arr)
    #     mask = np.zeros(matrix.shape, dtype=np.uint8)
    #     mask[segm > 0] = 1
    # '''
    # outputs <class 'dict'> dict_keys([
    #                         'image_size', 'pred_boxes', 'scores', 'pred_classes', 
    #                         'pred_densepose_coarse_segm', 'pred_densepose_fine_segm', 
    #                         'pred_densepose_u', 'pred_densepose_v'])
    # data <class 'tuple'> len=2
    # densepose_result <class 'list'> len=2
    # densepose_result[0] <class 'dict'> 
    # boxes_xywh <class 'torch.Tensor'> torch.Size([2, 4])
    # '''
    # print(f'''
    #       outputs {type(outputs)} {outputs.keys()}
    #       data {type(data)} {len(data)}
    #       densepose_result {type(densepose_result)} {len(densepose_result)}
    #       densepose_result[0]  {type(densepose_result[0])} {densepose_result[0].keys}
    #       boxes_xywh {type(boxes_xywh)} {boxes_xywh.shape}
    #       ''')
    test_img = np.zeros_like(img)
    
    image_vis = visualizer.visualize(test_img, outputs)

    cv2.imwrite(save_path, image_vis)
    print(f"Image saved to {save_path}")
else:
    cap = cv2.VideoCapture(args.input)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    try:
        for i in count():
            ret, frame = cap.read()
            if not ret:
                break
            tensor = torch.from_numpy(frame)
            outputs = predictor(tensor)
            image_vis = visualizer.visualize(frame, outputs)
            if writer is None:
                writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (image_vis.shape[1], image_vis.shape[0]))
            writer.write(image_vis)
            print(f"Frame {i + 1}/{n_frames} processed", end="\r")
    except KeyboardInterrupt:
        pass
    if writer is not None:
        writer.release()
        print(f"Video saved to {save_path}")
    else:
        print("No frames processed")
