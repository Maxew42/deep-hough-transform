import cv2
import argparse
import os
import random
import time
from os.path import isfile, join, split

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import numpy as np
import tqdm
import yaml
import cv2

from torch.optim import lr_scheduler
from logger import Logger

from dataloader import get_loader
from model.network import Net
from skimage.measure import label, regionprops
from utils import reverse_mapping, visulize_mapping, edge_align, get_boundary_point

parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
parser.add_argument('--config', default="./config.yml", help="path to config file")
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.load(open(args.config), Loader=yaml.Loader)

train_loader = get_loader(CONFIGS["DATA"]["DIR"], CONFIGS["DATA"]["LABEL_FILE"],
                          batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"],
                          split='train')

bar = tqdm.tqdm(train_loader)
iter_num = len(train_loader.dataset) // CONFIGS["DATA"]["BATCH_SIZE"]

total_loss_hough = 0
for i, data in enumerate(bar):

    images, hough_space_label, x_gt, names = data
    print(x_gt)
    # print(type(hough_space_label))
    # print(hough_space_label.shape)
    size = [[images.shape[2], images.shape[3]]]
    key_points = torch.sigmoid(hough_space_label)

    visualize_save_path = os.path.join("./resultMerde/reproduce", 'visualize_test')
    os.makedirs(visualize_save_path, exist_ok=True)

    binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
    kmap_label = label(binary_kmap, connectivity=1)
    props = regionprops(kmap_label)
    plist = []
    for prop in props:
        plist.append(prop.centroid)

    size = (size[0][0], size[0][1])
    b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"],
                               size=(400, 400))
    scale_w = size[1] / 400
    scale_h = size[0] / 400
    for i in range(len(b_points)):
        y1 = int(np.round(b_points[i][0] * scale_h))
        x1 = int(np.round(b_points[i][1] * scale_w))
        y2 = int(np.round(b_points[i][2] * scale_h))
        x2 = int(np.round(b_points[i][3] * scale_w))
        if x1 == x2:
            angle = -np.pi / 2
        else:
            angle = np.arctan((y1 - y2) / (x1 - x2))
        (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
        b_points[i] = (y1, x1, y2, x2)
    print(b_points)

    b_points = x_gt[0].tolist()
    print(b_points)
    vis = visulize_mapping(b_points, size[::-1], names[0])

    cv2.imwrite(join(visualize_save_path, names[0].split('/')[-1]), vis)
