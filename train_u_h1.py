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
import random
import numpy as np
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

def save_colored_depth(tensor, filename, vmin=0, vmax=48):
    """Chuyển đổi tensor chỉ số depth sang ảnh màu jet và lưu lại."""
    data = tensor.detach().cpu().numpy()
    if len(data.shape) == 3: # Nếu là (C, H, W)
        data = data[0]
    plt.imsave(filename, data, vmin=vmin, vmax=vmax, cmap='jet')

def main():
    # ===================== ARGS =====================
    parser = argparse.ArgumentParser(description='Training for OmniMVS',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('root_dir', nargs='?', 
                        default='/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/omnithings')
    parser.add_argument('-t', '--train-list', default='./dataloader/omnithings_train.txt')
    parser.add_argument('-v', '--val-list', default='./dataloader/omnithings_val.txt')
    
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--min_depth', type=float, default=0.55)
    parser.add_argument('--ndisp', type=int, default=48)
    parser.add_argument('--input_width', type=int, default=500)
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--output_width', type=int, default=512)
    parser.add_argument('--output_height', type=int, default=256)
    parser.add_argument('-j', '--workers', default=6, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--arch', default='omni_small', type=str)
    parser.add_argument('--save-interval', type=int, default=20) # Giảm interval để dễ quan sát

    # Giả lập tham số chạy từ notebook
    args = parser.parse_args(f"{parser.get_default('root_dir')} -t {parser.get_default('train_list')}".split())

    # ===================== DEVICE =====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cpu':
        cudnn.benchmark = True

    # ===================== MODEL & PRECOMPUTE =====================
    sweep = SphericalSweeping(args.root_dir, h=args.output_height, w=args.output_width)
    model = OmniMVS(sweep, args.ndisp, args.min_depth, h=args.output_height, w=args.output_width)
    
    invd_0 = model.inv_depths[0]
    invd_max = model.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    model = model.to(device)

    print('=> Precompute sweep grid')
    pool = ThreadPoolExecutor(5)
    for i in range(4):
        for d in model.depths[::2]:
            pool.submit(sweep.get_grid, i, d)
    pool.shutdown()

    # ===================== OPTIMIZER =====================
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ===================== LOGGING =====================
    timestamp = datetime.now().strftime('%m%d-%H%M')
    log_folder = os.path.join('checkpoints', f'{args.arch}_{timestamp}')
    img_dir = os.path.join(log_folder, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # ===================== DATALOADER =====================
    train_transform = transforms.Compose([
        Resize((args.input_width, args.input_height), (args.output_width, args.output_height)),
        ToTensor(),
        Normalize()
    ])

    full_trainset = OmniStereoDataset(args.root_dir, args.train_list, transform=train_transform)
    
    # Lấy subset 1% để debug nhanh
    percent = 0.02 
    subset_size = int(percent * len(full_trainset))
    indices = np.random.choice(len(full_trainset), subset_size, replace=False)
    trainset = Subset(full_trainset, indices)

    train_loader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.workers)
    print(f"Dataset: {len(full_trainset)} samples | Training on: {len(trainset)} samples")

    # ===================== TRAINING LOOP =====================
    for epoch in range(args.epochs):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for idx, batch in enumerate(pbar):
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            pred = model(batch) # Output shape: [B, H, W], values: [0, ndisp-1]
            gt_idepth = batch['idepth']
            gt_invd_idx = converter.invdepth_to_index(gt_idepth)

            loss = nn.L1Loss()(pred, gt_invd_idx)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # ============ SAVE IMAGES (FIXED) ============
            if idx % args.save_interval == 0:
                # 1. Lưu ảnh Input (De-normalize để nhìn rõ)
                imgs = []
                for cam in ['cam1', 'cam2', 'cam3', 'cam4']:
                    # Nghịch đảo Normalize([0.5], [0.5]): img * 0.5 + 0.5
                    imgs.append(batch[cam][0] * 0.5 + 0.5)
                
                img_grid = make_grid(imgs, nrow=2, padding=5, pad_value=1)
                save_image(img_grid, os.path.join(img_dir, f"ep{epoch:02d}_it{idx:03d}_input.png"))

                # 2. Lưu ảnh Pred và GT với Colormap
                save_colored_depth(pred[0], os.path.join(img_dir, f"ep{epoch:02d}_it{idx:03d}_pred.png"), vmax=args.ndisp)
                save_colored_depth(gt_invd_idx[0], os.path.join(img_dir, f"ep{epoch:02d}_it{idx:03d}_gt.png"), vmax=args.ndisp)
                
                # Debug giá trị
                # print(f"\nDebug values - Pred Max: {pred.max().item():.1f}, GT Max: {gt_invd_idx.max().item():.1f}")

        scheduler.step()
        print(f"Epoch {epoch} Average Loss: {sum(losses)/len(losses):.4f}")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ndisp': args.ndisp
        }, os.path.join(log_folder, f'checkpoint_latest.pth'))

    print('Training finished!')

if __name__ == '__main__':
    main()