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

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ----------------------------
# ARGUMENTS
# ----------------------------
parser = argparse.ArgumentParser()

# parser.add_argument('root_dir', nargs='?', default=r'F:\tmp\datasets\omnithings')
# parser.add_argument('-t', '--train-list', default=r'.\dataloader\omnithings_train.txt', type=str)
# parser.add_argument('-v', '--val-list', default=r'.\dataloader\omnithings_val.txt', type=str)

parser.add_argument(
    'root_dir',
    nargs='?',
    default='/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/omnithings'
)

parser.add_argument(
    '-t', '--train-list',
    default='./dataloader/omnithings_train.txt'
)

parser.add_argument(
    '-v', '--val-list',
    default='./dataloader/omnithings_val.txt'
)


parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--pretrained', default=None)
parser.add_argument('-b', '--batch-size', default=2, type=int)
parser.add_argument('--min_depth', type=float, default=0.55)
parser.add_argument('--fov', type=float, default=220)
parser.add_argument('--ndisp', type=int, default=48)
parser.add_argument('--input_width', type=int, default=500)
parser.add_argument('--input_height', type=int, default=480)
parser.add_argument('--output_width', type=int, default=512)
parser.add_argument('--output_height', type=int, default=256)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--arch', default='omni_small')
parser.add_argument('--log-interval', default=5, type=int)

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

def save_rgb_image(img_tensor, path):
    """Lưu ảnh RGB từ tensor"""
    img_tensor = img_tensor.detach()
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3,1,1)
    img_tensor = (img_tensor * 0.5 + 0.5).clamp(0,1)
    save_image(img_tensor, path)

# ----------------------------
# TRAINING LOOP
# ----------------------------
def train(args, model, train_loader, optimizer, writer, epoch, device):
    invd_0, invd_max = model.module.inv_depths[0], model.module.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)
    ndisp = model.module.ndisp

    model.train()
    losses = []
    pbar = tqdm(train_loader)
    for idx, batch in enumerate(pbar):
        for k in batch.keys():
            batch[k] = batch[k].to(device)

        pred = model(batch)

        # print(f"batchsize: {batch['cam1']}")
        # print(f"batchsize: {batch['cam1'].size()}")
        # print(f"pred: {pred.size()}")



        gt_idx = converter.invdepth_to_index(batch['idepth'])
        loss = nn.L1Loss()(pred, gt_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        pbar.set_postfix(OrderedDict(epoch=f"{epoch}", loss=f"{losses[-1]:.4f}"))

        niter = epoch * len(train_loader) + idx
        if idx % args.log_interval == 0:
            writer.add_scalar('train/loss', loss.item(), niter)
            # --- Lưu ảnh RGB và GT thực ---
            for cam in model.module.cam_list:
                save_rgb_image(batch[cam][0], f"pred_train/train_epoch{epoch}_idx{idx}_{cam}.png")
            save_depth_as_colormap(batch['idepth'][0:1], f"pred_train/train_epoch{epoch}_idx{idx}_gt.png", vmin=0.0, vmax=None)
            save_depth_as_colormap(pred[0:1], f"pred_train/train_epoch{epoch}_idx{idx}_pred.png", vmin=0, vmax=ndisp)

    ave_loss = sum(losses)/len(losses)
    writer.add_scalar('train/loss_ave', ave_loss, epoch)
    return ave_loss

# ----------------------------
# VALIDATION LOOP
# ----------------------------
def validation(args, model, val_loader, writer, epoch, device, save_dir='./val_results'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    invd_0, invd_max = model.module.inv_depths[0], model.module.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)
    ndisp = model.module.ndisp

    preds, gts, losses = [], [], []
    pbar = tqdm(val_loader)
    for idx, batch in enumerate(pbar):
        with torch.no_grad():
            for k in batch.keys():
                batch[k] = batch[k].to(device)

            pred = model(batch)
            gt_idx = converter.invdepth_to_index(batch['idepth'])
            loss = nn.L1Loss()(pred, gt_idx)
            losses.append(loss.item())
            preds.append(pred.cpu())
            gts.append(gt_idx.cpu())

            # Lưu ảnh RGB
            for cam in model.module.cam_list:
                save_rgb_image(batch[cam][0], join(save_dir, f'epoch{epoch}_idx{idx}_{cam}.png'))
            # Lưu ảnh depth
            save_depth_as_colormap(pred[0:1], join(save_dir, f'epoch{epoch}_idx{idx}_pred.png'), vmin=0, vmax=ndisp)
            # **Sửa lỗi GT: dùng batch['idepth'] thay vì gt_idx**
            save_depth_as_colormap(batch['idepth'][0:1], join(save_dir, f'epoch{epoch}_idx{idx}_gt.png'), vmin=0.0, vmax=None)

        pbar.set_postfix(OrderedDict(epoch=f"{epoch}", loss=f"{losses[-1]:.4f}"))
        niter = epoch * len(val_loader) + idx
        if idx % args.log_interval == 0:
            writer.add_scalar('val/loss', loss.item(), niter)

    # Metrics
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    errors, names = evaluation_metrics(preds, gts, args.ndisp)
    for name, val in zip(names, errors):
        writer.add_scalar(f'val_metrics/{name}', val, epoch)

    ave_loss = sum(losses)/len(losses)
    writer.add_scalar('val/loss_ave', ave_loss, epoch)
    return ave_loss

# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cpu': cudnn.benchmark = True

    # Model
    sweep = SphericalSweeping(args.root_dir, h=args.output_height, w=args.output_width, fov=args.fov)
    model = OmniMVS(sweep, args.ndisp, args.min_depth, h=args.output_height, w=args.output_width).to(device)

    # Precompute grids
    pool = ThreadPoolExecutor(5)
    for i in range(4):
        for d in model.depths[::2]:
            pool.submit(sweep.get_grid, i, d)
    pool.shutdown()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2*args.epochs//3, gamma=0.1)

    # Load pretrained
    start_epoch = 0
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    model = nn.DataParallel(model)

    # Logger
    timestamp = datetime.now().strftime("%m%d-%H%M")
    log_folder = join('checkpoints', f'{args.arch}_{timestamp}')
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)
    with open(join(log_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)

    # Datasets
    image_size = (args.input_width, args.input_height)
    depth_size = (args.output_width, args.output_height)
    transform = transforms.Compose([Resize(image_size, depth_size), ToTensor(), Normalize()])
    trainset = OmniStereoDataset(args.root_dir, args.train_list, transform=transform, fov=args.fov)

    # Subset demo
    subset_size = math.ceil(len(trainset)/300)
    train_subset = Subset(trainset, range(subset_size))
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(train_subset, batch_size=1, shuffle=False)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        ave_train = train(args, model, train_loader, optimizer, writer, epoch, device)
        ave_val = validation(args, model, val_loader, writer, epoch, device)
        print(f"Epoch {epoch}: Train={ave_train:.4f}, Val={ave_val:.4f}")
        scheduler.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch+1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ave_loss': ave_val,
            'ndisp': model.module.ndisp,
            'min_depth': model.module.min_depth,
            'output_width': model.module.w,
            'output_height': model.module.h,
        }, join(log_folder, f'checkpoints_{epoch}.pth'))

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()