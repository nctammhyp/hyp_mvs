from glob import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, '../')
from os.path import join
from ocamcamera import OcamCamera

import argparse
import random
from datetime import datetime
import json
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (8, 6)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision import transforms

from models import OmniMVS, SphericalSweeping
from dataloader import OmniStereoDataset
from dataloader import load_image, load_invdepth
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from utils import InvDepthConverter, evaluation_metrics

# ---------------------------
# Argument parser
# ---------------------------
parser = argparse.ArgumentParser(description='Training for OmniMVS',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('root_dir', metavar='DIR', help='path to dataset')
parser.add_argument('-t','--train-list', default='../datasets/omnithings/omnithings_train.txt',
                    type=str, help='Text file includes filenames for training')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='total epochs')
parser.add_argument('--pretrained', default=None, metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--min_depth', type=float, default=0.55, help='minimum depth in m')

# Lightweight model settings
parser.add_argument('--ndisp', type=int, default=64, help='number of disparity')
parser.add_argument('--input_width', type=int, default=500, help='input image width')
parser.add_argument('--input_height', type=int, default=480, help='input image height')
parser.add_argument('--output_width', type=int, default=512, help='output depth width')
parser.add_argument('--output_height', type=int, default=256, help='output depth height')

parser.add_argument('-j', '--workers', default=6, type=int, metavar='J', help='number of data loading workers')
parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum for sgd')
parser.add_argument('--arch', default='omni_small', type=str, help='architecture name for log folder')
parser.add_argument('--log-interval', type=int, default=1, metavar='L', help='tensorboard log interval')

# ---------------------------
# Main runtime
# ---------------------------
if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()  # required for Windows multiprocessing

    # Example dataset paths
    root_dir = r'F:\tmp\datasets\omnithings'
    file_list = '-t ./omnithings_train.txt'
    pretrained = "--pretrained ../checkpoints/pretrain/checkpoints_ndisp_48.pth"

    args = parser.parse_args(f'{root_dir} {file_list} {pretrained} --lr 1e-3 --ndisp 48'.split())

    # ---------------------------
    # Generate filename lists
    # ---------------------------
    with open('omnithings_train.txt', 'w') as f:
        for i in range(1, 4097):
            f.write(f'{i:05}.png\n')
        for i in range(5121, 8241):
            f.write(f'{i:05}.png\n')
    with open('omnithings_val.txt', 'w') as f:
        for i in range(8241, 10240+1):
            f.write(f'{i:05}.png\n')
    with open('omnihouse_val.txt', 'w') as f:
        for i in range(1, 2560+1):
            f.write(f'{i:04}.png\n')

    # ---------------------------
    # Device setup
    # ---------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cpu':
        cudnn.benchmark = True
    print("device:", device)

    # ---------------------------
    # Model setup
    # ---------------------------
    sweep = SphericalSweeping(args.root_dir, h=args.output_height, w=args.output_width)
    model = OmniMVS(sweep, args.ndisp, args.min_depth, h=args.output_height, w=args.output_width)
    invd_0 = model.inv_depths[0]
    invd_max = model.inv_depths[-1]

    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)
    model = model.to(device)
    start_epoch = 0

    # ---------------------------
    # Precompute grids
    # ---------------------------
    num_cam = 4
    pool = ThreadPoolExecutor(5)
    futures = []
    for i in range(num_cam):
        for d in model.depths[::2]:
            futures.append(pool.submit(sweep.get_grid, i, d))

    # ---------------------------
    # Optimizer and scheduler
    # ---------------------------
    print('=> setting optimizer')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print('=> setting scheduler')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ---------------------------
    # Load pretrained weights
    # ---------------------------
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        param_check = {
            'ndisp' : model.ndisp,
            'min_depth' : model.min_depth,
            'output_width' : model.w,
            'output_height' : model.h,
        }
        for key, val in param_check.items():
            if not checkpoint[key] == val:
                print(f'Error! Key:{key} is not the same as the checkpoints')

        print("=> using pre-trained weights")
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Resume training from epoch {}".format(start_epoch))

    # ---------------------------
    # Logging
    # ---------------------------
    timestamp = datetime.now().strftime("%m%d-%H%M")
    log_folder = join('checkpoints', f'{args.arch}_{timestamp}')
    print(f'=> create log folder: {log_folder}')
    os.makedirs(log_folder, exist_ok=True)
    with open(join(log_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    writer = SummaryWriter(log_dir=log_folder)

    # ---------------------------
    # Transforms
    # ---------------------------
    image_size = (args.input_width, args.input_height)
    depth_size = (args.output_width, args.output_height)
    ToPIL = lambda x: transforms.ToPILImage()(x.cpu())
    train_transform = transforms.Compose([Resize(image_size, depth_size), ToTensor(), Normalize()])

    # ---------------------------
    # Dataset and DataLoader
    # ---------------------------
    filename_txt = args.train_list
    root_dir = args.root_dir
    trainset = OmniStereoDataset(root_dir, filename_txt, transform=train_transform)
    print(f'{len(trainset)} samples were found.')

    train_loader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.workers)
    loader_iter = iter(train_loader)

    # ---------------------------
    # Test visualization
    # ---------------------------
    batch = next(loader_iter)
    tensor = batch['cam1'][0]
    plt.imshow(ToPIL(0.5 + 0.5*tensor))
    plt.show()
