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

# ----------------------------
# ARGUMENTS
# ----------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('root_dir', nargs='?', default=r'F:\tmp\datasets\omnithings')
# parser.add_argument('-t', '--train-list', default=r'.\dataloader\omnithings_train.txt')
# parser.add_argument('-v', '--val-list', default=r'.\dataloader\omnithings_val.txt')

parser.add_argument(
    'root_dir',
    nargs='?',
    default='tamnguyen/Desktop/depth_project/datasets/datasets/omnithings'
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
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('--min_depth', type=float, default=0.55)
parser.add_argument('--fov', type=float, default=220)
parser.add_argument('--ndisp', type=int, default=64)
parser.add_argument('--input_width', type=int, default=500)
parser.add_argument('--input_height', type=int, default=480)
parser.add_argument('--output_width', type=int, default=512)
parser.add_argument('--output_height', type=int, default=256)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--arch', default='omni_small')
parser.add_argument('--log-interval', default=5, type=int)

# ----------------------------
# UTILS
# ----------------------------
def save_depth_as_colormap(depth, path, vmin=None, vmax=None):
    depth = depth.detach().squeeze().cpu().numpy()
    vmin = depth.min() if vmin is None else vmin
    vmax = depth.max() if vmax is None else vmax
    normed = (depth - vmin) / (vmax - vmin + 1e-8)
    cmap = plt.get_cmap('jet')(normed)[:, :, :3]
    cmap = (cmap * 255).astype(np.uint8)
    plt.imsave(path, cmap)

def save_rgb_image(img, path):
    img = img.detach()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = (img * 0.5 + 0.5).clamp(0, 1)
    save_image(img, path)

# ----------------------------
# TRAIN
# ----------------------------
def train(args, model, loader, optimizer, writer, epoch, device):
    model.train()
    invd_0, invd_max = model.inv_depths[0], model.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    losses = []
    pbar = tqdm(loader)

    for idx, batch in enumerate(pbar):
        for k in batch:
            batch[k] = batch[k].to(device)

        pred = model(batch)
        gt_idx = converter.invdepth_to_index(batch['idepth'])
        loss = nn.L1Loss()(pred, gt_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}")

        niter = epoch * len(loader) + idx
        if idx % args.log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), niter)

            for cam in model.cam_list:
                save_rgb_image(batch[cam][0],
                               f"train_e{epoch}_i{idx}_{cam}.png")

            save_depth_as_colormap(batch['idepth'][0:1],
                                   f"train_e{epoch}_i{idx}_gt.png")
            save_depth_as_colormap(pred[0:1],
                                   f"train_e{epoch}_i{idx}_pred.png",
                                   vmin=0, vmax=args.ndisp)

    ave = sum(losses) / len(losses)
    writer.add_scalar("train/loss_ave", ave, epoch)
    return ave

# ----------------------------
# VALIDATION
# ----------------------------
def validation(args, model, loader, writer, epoch, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    invd_0, invd_max = model.inv_depths[0], model.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    preds, gts, losses = [], [], []
    pbar = tqdm(loader)

    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            for k in batch:
                batch[k] = batch[k].to(device)

            pred = model(batch)
            gt_idx = converter.invdepth_to_index(batch['idepth'])
            loss = nn.L1Loss()(pred, gt_idx)

            preds.append(pred.cpu())
            gts.append(gt_idx.cpu())
            losses.append(loss.item())

            for cam in model.cam_list:
                save_rgb_image(batch[cam][0],
                               join(save_dir, f"e{epoch}_i{idx}_{cam}.png"))

            save_depth_as_colormap(pred[0:1],
                                   join(save_dir, f"e{epoch}_i{idx}_pred.png"),
                                   vmin=0, vmax=args.ndisp)
            save_depth_as_colormap(batch['idepth'][0:1],
                                   join(save_dir, f"e{epoch}_i{idx}_gt.png"))

            pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}")

    preds = torch.cat(preds)
    gts = torch.cat(gts)
    errors, names = evaluation_metrics(preds, gts, args.ndisp)

    for n, v in zip(names, errors):
        writer.add_scalar(f"val_metrics/{n}", v, epoch)

    ave = sum(losses) / len(losses)
    writer.add_scalar("val/loss_ave", ave, epoch)
    return ave

# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    # Model
    sweep = SphericalSweeping(args.root_dir,
                              h=args.output_height,
                              w=args.output_width,
                              fov=args.fov)

    model = OmniMVS(sweep,
                    args.ndisp,
                    args.min_depth,
                    h=args.output_height,
                    w=args.output_width).to(device)

    # Precompute grids
    pool = ThreadPoolExecutor(4)
    for i in range(4):
        for d in model.depths[::2]:
            pool.submit(sweep.get_grid, i, d)
    pool.shutdown()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2 * args.epochs // 3, gamma=0.1
    )

    start_epoch = 0
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']

    # Logger
    log_dir = join("checkpoints",
                   f"{args.arch}_{datetime.now():%m%d-%H%M}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    with open(join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Dataset
    transform = transforms.Compose([
        Resize((args.input_width, args.input_height),
               (args.output_width, args.output_height)),
        ToTensor(),
        Normalize()
    ])

    trainset = OmniStereoDataset(args.root_dir,
                                 args.train_list,
                                 transform=transform,
                                 fov=args.fov)

    subset_size = math.ceil(len(trainset) / 300)
    subset = Subset(trainset, range(subset_size))

    train_loader = DataLoader(subset,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(subset,
                            batch_size=1,
                            shuffle=False)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        tr = train(args, model, train_loader,
                   optimizer, writer, epoch, device)
        va = validation(args, model, val_loader,
                        writer, epoch, device,
                        join(log_dir, "val"))

        print(f"[Epoch {epoch}] Train={tr:.4f}  Val={va:.4f}")
        scheduler.step()

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ave_loss': va,
        }, join(log_dir, f"checkpoint_{epoch}.pth"))

    writer.close()
    print("✅ Training finished")

if __name__ == "__main__":
    main()
