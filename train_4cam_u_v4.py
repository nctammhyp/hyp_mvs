import argparse, os, json
from os.path import join
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from models import OmniMVS, SphericalSweeping
from utils import InvDepthConverter
from torch.utils.data import DataLoader, Subset

# ----------------------------
# Seed
# ----------------------------
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('root_dir', nargs='?', default='/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/omnithings')
parser.add_argument('-t', '--train-list', default='./dataloader/omnithings_train.txt')
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('-b', '--batch-size', default=2, type=int)
parser.add_argument('--min_depth', type=float, default=0.55)
parser.add_argument('--fov', type=float, default=220)
parser.add_argument('--ndisp', type=int, default=48)
parser.add_argument('--input_width', type=int, default=500)
parser.add_argument('--input_height', type=int, default=480)
parser.add_argument('--output_width', type=int, default=512)
parser.add_argument('--output_height', type=int, default=256)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--log-interval', default=10, type=int)

# ----------------------------
# Utils
# ----------------------------
def save_depth_as_colormap(depth, path, vmin=None, vmax=None):
    """Lưu depth map dưới dạng colormap"""
    depth = depth.detach().squeeze().cpu().numpy()
    if vmin is None: vmin = depth.min()
    if vmax is None: vmax = depth.max()
    normed = (depth - vmin) / (vmax - vmin + 1e-8)
    cmap = plt.get_cmap('jet')(normed)[:, :, :3]
    cmap = (cmap * 255).astype(np.uint8)
    plt.imsave(path, cmap)

# ----------------------------
# Train loop
# ----------------------------
def train_epoch(args, model, loader, optimizer, writer, epoch, device):
    model.train()
    invd_0 = model.module.inv_depths[0]
    invd_max = model.module.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    losses = []
    pbar = tqdm(loader)
    for i, batch in enumerate(pbar):
        for k in batch:
            batch[k] = batch[k].to(device)

        pred_idx = model(batch)
        gt_idx = converter.invdepth_to_index(batch['idepth'])

        # Mask invalid depth
        valid = batch['idepth'] > 0
        loss = (torch.abs(pred_idx - gt_idx) * valid).sum() / valid.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Log
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if i % args.log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), epoch * len(loader) + i)
            print(f"TRAIN pred_idx stats: min={pred_idx.min().item():.2f} "
                  f"max={pred_idx.max().item():.2f} mean={pred_idx.mean().item():.2f}")

    return sum(losses)/len(losses)

# ----------------------------
# Main
# ----------------------------
def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # ----------------------------
    # Model
    # ----------------------------
    sweep = SphericalSweeping(args.root_dir, h=args.output_height, w=args.output_width, fov=args.fov)
    model = OmniMVS(sweep, args.ndisp, args.min_depth, args.output_width, args.output_height)
    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------------
    # Dataset & Dataloader
    # ----------------------------
    transform = transforms.Compose([
        Resize((args.input_width, args.input_height), (args.output_width, args.output_height)),
        ToTensor(),
        Normalize()
    ])
    dataset = OmniStereoDataset(args.root_dir, args.train_list, transform=transform, fov=args.fov)
    # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Dataset đầy đủ
    dataset = OmniStereoDataset(args.root_dir, args.train_list, transform=transform, fov=args.fov)

    # Lấy 50 ảnh đầu tiên để test nhanh
    subset_size = 50
    subset_dataset = Subset(dataset, range(subset_size))

    # Dataloader
    loader = DataLoader(
        subset_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    # ----------------------------
    # Logger
    # ----------------------------
    logdir = join("checkpoints", datetime.now().strftime("%m%d-%H%M"))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    with open(join(logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ----------------------------
    # Training
    # ----------------------------
    for epoch in range(args.epochs):
        loss = train_epoch(args, model, loader, optimizer, writer, epoch, device)
        print(f"[Epoch {epoch}] train_loss={loss:.4f}")

        # Save checkpoint
        torch.save(model.module.state_dict(), join(logdir, f"model_epoch{epoch}.pth"))

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
