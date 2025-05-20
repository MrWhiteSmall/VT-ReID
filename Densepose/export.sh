python run.py <model.pt> <input.[jpg|png|mp4|avi]>

# Legacy Models
python export.py configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x_legacy/164832157/model_final_d366fa.pkl

python export.py configs/densepose_rcnn_R_101_FPN_s1x_legacy.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x_legacy/164832182/model_final_10af0e.pkl

# Improved Baselines, Original Fully Convolutional Head
python export.py configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl

python export.py configs/densepose_rcnn_R_101_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl

# Improved Baselines, DeepLabV3 Head
python export.py configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/model_final_0ed407.pkl

python export.py configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml densepose_rcnn_R_101_FPN_DL_s1x.pkl