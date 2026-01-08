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

from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from models import OmniMVS, SphericalSweeping
from utils import InvDepthConverter, evaluation_metrics
import random

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
parser.add_argument('-v', '--val-list', default='./dataloader/omnithings_val.txt')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch-size', default=2, type=int)
parser.add_argument('--min_depth', type=float, default=0.55)
parser.add_argument('--fov', type=float, default=220)
parser.add_argument('--ndisp', type=int, default=48)
parser.add_argument('--input_width', type=int, default=500)
parser.add_argument('--input_height', type=int, default=480)
parser.add_argument('--output_width', type=int, default=512)
parser.add_argument('--output_height', type=int, default=256)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--arch', default='omni_small')
parser.add_argument('--log-interval', default=20, type=int)

# ----------------------------
# UTILS
# ----------------------------
def save_depth_as_colormap(depth, path, vmin=0.5, vmax=10.0):
    depth = depth.detach().squeeze().cpu().numpy()
    normed = (depth - vmin) / (vmax - vmin + 1e-8)
    cmap = plt.get_cmap('jet')(np.clip(normed, 0, 1))[:, :, :3]
    plt.imsave(path, cmap)

def save_rgb_image(img_tensor, path):
    img_tensor = (img_tensor * 0.5 + 0.5).clamp(0, 1)
    save_image(img_tensor, path)

# ----------------------------
# TRAIN
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

        pred_idx = model(batch)
        gt_idx = converter.invdepth_to_index(batch['idepth'])

        valid = batch['idepth'] > 0
        loss = (torch.abs(pred_idx - gt_idx) * valid).sum() / valid.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if i % args.log_interval == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(loader) + i)

            print(
                "TRAIN pred_idx:",
                pred_idx.min().item(),
                pred_idx.max().item(),
                pred_idx.mean().item()
            )

    return sum(losses) / len(losses)

# ----------------------------
# VALIDATION
# ----------------------------
def validate(args, model, loader, writer, epoch, device):
    model.eval()

    invd_0 = model.module.inv_depths[0]
    invd_max = model.module.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    losses = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            for k in batch:
                batch[k] = batch[k].to(device)

            pred_idx = model(batch)
            gt_idx = converter.invdepth_to_index(batch['idepth'])

            valid = batch['idepth'] > 0
            loss = (torch.abs(pred_idx - gt_idx) * valid).sum() / valid.sum()
            losses.append(loss.item())

    ave = sum(losses) / len(losses)
    writer.add_scalar('val/loss', ave, epoch)
    return ave

# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    sweep = SphericalSweeping(args.root_dir, args.output_height, args.output_width, args.fov)
    model = OmniMVS(sweep, args.ndisp, args.min_depth, args.output_height, args.output_width)
    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    transform = transforms.Compose([
        Resize((args.input_width, args.input_height), (args.output_width, args.output_height)),
        ToTensor(),
        Normalize()
    ])

    dataset = OmniStereoDataset(args.root_dir, args.train_list, transform, args.fov)

    subset_size = len(dataset) // 20   # <<< QUAN TRỌNG
    dataset = Subset(dataset, range(subset_size))

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    logdir = join("checkpoints", datetime.now().strftime("%m%d-%H%M"))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    for epoch in range(args.epochs):
        train_loss = train(args, model, train_loader, optimizer, writer, epoch, device)
        val_loss = validate(args, model, val_loader, writer, epoch, device)
        scheduler.step()

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        torch.save(
            model.module.state_dict(),
            join(logdir, f"model_{epoch}.pth")
        )

    writer.close()

if __name__ == "__main__":
    main()
