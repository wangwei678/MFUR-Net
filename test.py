import os
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from options import opt
from model.MFURNet_Res50 import MFURNet_R50
from dataLoader import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load the model
print('load model...')
if opt.backbone == 'R50':
    model = MFURNet_R50()
print('Use backbone ' + opt.backbone)
gpu_num = torch.cuda.device_count()
# load gpu
if gpu_num == 1:
    print("Use Single GPU-", opt.gpu_id)
elif gpu_num > 1:
    print("Use multiple GPUs-", opt.gpu_id)
    model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load('./MFUR-Net_cpts/' + opt.test_model, map_location='cpu'))


model.cuda()
model.eval()

test_datasets = ['SIP', 'DUT', 'NJU2K', 'STERE', 'NLPR', 'LFSD', 'DES', 'ReDWeb-S']

dataset_path = opt.test_path

for dataset in test_datasets:
    print("Testing {} ...".format(dataset))
    save_path = 'test_maps/' + opt.backbone + '/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    depth_root = dataset_path + dataset + '/depth/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, depth_root, gt_root, opt.testsize)
    mae_sum = 0
    for i in range(test_loader.size):
        image_s, depth_s, gt_s, name = test_loader.load_data()
        name = name.split('/')[-1]
        gt = np.asarray(gt_s, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image_s.cuda()
        depth = depth_s.cuda()
        _, _, pre = model(image, depth)
        pre_s = F.interpolate(pre, size=gt.shape, mode='bilinear', align_corners=False)
        pre = pre_s.sigmoid().data.cpu().numpy().squeeze()
        pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)
        cv2.imwrite(save_path + name, pre*255)
    print("Dataset:{} testing completed.".format(dataset))
print("Test Done!")
