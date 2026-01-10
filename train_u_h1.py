import os
import json
import argparse
from datetime import datetime
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from models import OmniMVS, SphericalSweeping
from dataloader.omnistereo_dataset import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from utils import InvDepthConverter

from tqdm import tqdm
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Training for OmniMVS')
    parser.add_argument('root_dir', nargs='?', default='/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/omnithings')
    parser.add_argument('-t', '--train-list', default='./dataloader/omnithings_train.txt')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--ndisp', type=int, default=48)
    parser.add_argument('--min_depth', type=float, default=0.55)
    parser.add_argument('--input_width', type=int, default=500)
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--output_width', type=int, default=512)
    parser.add_argument('--output_height', type=int, default=256)
    parser.add_argument('-j', '--workers', default=6, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--arch', default='omni_small', type=str)
    parser.add_argument('--save-interval', type=int, default=100)

    args = parser.parse_args(f"{parser.get_default('root_dir')} -t {parser.get_default('train_list')}".split())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cpu':
        cudnn.benchmark = True

    # Setup Model
    sweep = SphericalSweeping(args.root_dir, h=args.output_height, w=args.output_width)
    model = OmniMVS(sweep, args.ndisp, args.min_depth, h=args.output_height, w=args.output_width)
    
    invd_0 = model.inv_depths[0]
    invd_max = model.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)
    model = model.to(device)

    # Dataloader
    train_transform = transforms.Compose([
        Resize((args.input_width, args.input_height), (args.output_width, args.output_height)),
        ToTensor(),
        Normalize()
    ])
    
    full_dataset = OmniStereoDataset(args.root_dir, args.train_list, transform=train_transform)
    indices = np.random.choice(len(full_dataset), int(0.01 * len(full_dataset)), replace=False)
    train_loader = DataLoader(Subset(full_dataset, indices), args.batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    log_folder = os.path.join('checkpoints', f'{args.arch}_{datetime.now().strftime("%m%d-%H%M")}')
    img_dir = os.path.join(log_folder, 'images')
    os.makedirs(img_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader)
        for idx, batch in enumerate(pbar):
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            pred = model(batch) # Shape: [B, H, W]
            gt_idepth = batch['idepth']
            gt_invd_idx = converter.invdepth_to_index(gt_idepth)

            loss = nn.L1Loss()(pred, gt_invd_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # ============ FIX LOGIC LƯU ẢNH ============
            if idx % args.save_interval == 0:
                # 1. Kiểm tra giá trị thực tế (Debug logic)
                with torch.no_grad():
                    p_max, g_max = pred.max().item(), gt_invd_idx.max().item()
                    if g_max == 0:
                        print(f"\n[WARNING] GT index is ALL ZEROS. Check load_invdepth logic!")

                # 2. Lưu ảnh Pred/GT: Cần thêm dimension channel [1, H, W]
                # Chọn sample đầu tiên trong batch [0], sau đó unsqueeze(0) để có shape [1, H, W]
                res_pred = (pred[0:1] / args.ndisp).clamp(0, 1) 
                res_gt = (gt_invd_idx[0:1] / args.ndisp).clamp(0, 1)

                save_image(res_pred, os.path.join(img_dir, f"ep{epoch}_it{idx}_pred.png"))
                save_image(res_gt, os.path.join(img_dir, f"ep{epoch}_it{idx}_gt.png"))

                # 3. Lưu Input: Phải nghịch đảo Normalize (x * 0.5 + 0.5)
                input_img = batch['cam1'][0] * 0.5 + 0.5
                save_image(input_img, os.path.join(img_dir, f"ep{epoch}_it{idx}_input.png"))

    print('Training finished!')

if __name__ == '__main__':
    main()