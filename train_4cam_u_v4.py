import argparse, os, json, math
from os.path import join
from collections import OrderedDict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from models import OmniMVS, SphericalSweeping
from utils import InvDepthConverter

# ----------------------------
# SEED
# ----------------------------
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ----------------------------
# ARGUMENTS
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
parser.add_argument('--log-interval', default=20, type=int)

# ----------------------------
# UTILS
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
# TRAIN LOOP
# ----------------------------
def train(args, model, loader, optimizer, writer, epoch, device):
    model.train()

    invd_0 = model.module.inv_depths[0]
    invd_max = model.module.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    losses = []
    pbar = tqdm(loader)

    for i, batch in enumerate(pbar):
        for k in batch:
            batch[k] = batch[k].to(device)

        # ----------------------------
        # Forward
        # ----------------------------
        pred_idx = model(batch)  # shape [B, C, H, W] (index)
        gt_idx = converter.invdepth_to_index(batch['idepth'])

        # Mask invalid depth
        valid = batch['idepth'] > 0
        loss = (torch.abs(pred_idx - gt_idx) * valid).sum() / valid.sum()

        # ----------------------------
        # Backward
        # ----------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # ----------------------------
        # Logging
        # ----------------------------
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if i % args.log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), epoch * len(loader) + i)
            print(
                f"TRAIN pred_idx: min={pred_idx.min().item():.2f} "
                f"max={pred_idx.max().item():.2f} "
                f"mean={pred_idx.mean().item():.2f}"
            )

    return sum(losses) / len(losses)

# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # ----------------------------
    # MODEL (H/W đúng)
    # ----------------------------
    sweep = SphericalSweeping(
        args.root_dir,
        h=args.output_height,
        w=args.output_width,
        fov=args.fov
    )

    model = OmniMVS(
        sweep,
        args.ndisp,
        args.min_depth,
        h=args.output_height,
        w=args.output_width
    )

    model = nn.DataParallel(model).to(device)

    # ----------------------------
    # OPTIMIZER (Adam chống collapse)
    # ----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------------
    # DATA
    # ----------------------------
    transform = transforms.Compose([
        Resize((args.input_width, args.input_height),
               (args.output_width, args.output_height)),
        ToTensor(),
        Normalize()
    ])

    dataset = OmniStereoDataset(
        args.root_dir,
        args.train_list,
        transform=transform,
        fov=args.fov
    )

    # Subset đủ lớn
    subset_size = len(dataset) // 20
    dataset = Subset(dataset, range(subset_size))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ----------------------------
    # LOG
    # ----------------------------
    logdir = join("checkpoints", datetime.now().strftime("%m%d-%H%M"))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # ----------------------------
    # TRAIN
    # ----------------------------
    for epoch in range(args.epochs):
        loss = train(args, model, loader, optimizer, writer, epoch, device)
        print(f"[Epoch {epoch}] train_loss = {loss:.4f}")

        # Save checkpoint
        torch.save(
            model.module.state_dict(),
            join(logdir, f"model_epoch{epoch}.pth")
        )

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
