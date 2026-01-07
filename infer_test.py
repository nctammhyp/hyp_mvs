import os
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from models import OmniMVS, SphericalSweeping
from utils import InvDepthConverter

# ----------------------------
# DETERMINISTIC / SEED (GIỐNG TRAIN)
# ----------------------------
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ----------------------------
# UTILS
# ----------------------------
def save_depth_as_colormap(depth, path, vmin=None, vmax=None):
    depth = depth.detach().squeeze().cpu().numpy()
    if vmin is None: vmin = depth.min()
    if vmax is None: vmax = depth.max()
    depth = (depth - vmin) / (vmax - vmin + 1e-8)
    cmap = plt.get_cmap('jet')(depth)[:, :, :3]
    plt.imsave(path, (cmap * 255).astype(np.uint8))

def save_rgb_image(img, path):
    img = img.detach().cpu()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img_np = img.permute(1, 2, 0).numpy()
    plt.imsave(path, img_np)

# ----------------------------
# INFERENCE
# ----------------------------
@torch.no_grad()
def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- MODEL (GIỐNG TRAIN) ----
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
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(model)
    model.eval()

    # ---- INV DEPTH CONVERTER ----
    invd_0, invd_max = model.module.inv_depths[0], model.module.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    # ---- DATASET (GIỐNG TRAIN) ----
    image_size = (args.input_width, args.input_height)
    depth_size = (args.output_width, args.output_height)

    transform = transforms.Compose([
        Resize(image_size, depth_size),
        ToTensor(),
        Normalize()
    ])

    dataset = OmniStereoDataset(
        args.root_dir,
        args.list,
        transform=transform,
        fov=args.fov
    )

    # ---- SUBSET GIỐNG TRAIN ----
    subset_size = math.ceil(len(dataset) / 300)
    subset = Subset(dataset, range(subset_size))

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- LOOP ----
    for idx, batch in enumerate(tqdm(loader)):
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        pred_idx = model(batch)                       # (1, H, W)
        pred_idepth = converter.index_to_invdepth(pred_idx)

        # Save RGB
        for cam in model.module.cam_list:
            save_rgb_image(
                batch[cam][0],
                os.path.join(args.out_dir, f"{idx:04d}_{cam}.png")
            )

        # Save depth
        save_depth_as_colormap(
            pred_idepth[0],
            os.path.join(args.out_dir, f"{idx:04d}_pred.png")
        )

    print("✅ Inference finished.")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'root_dir',
        nargs='?',
        default=r'F:\tmp\datasets\omnithings'
    )

    parser.add_argument(
        '-v', '--list',
        default=r'.\dataloader\omnithings_train.txt'
    )

    parser.add_argument(
        '--checkpoint',
        default=r'F:\omnimvs_pytorch\checkpoints\pretrain\checkpoints_31.pth'
    )

    parser.add_argument('--out_dir', default='inference_results')

    parser.add_argument('--ndisp', type=int, default=48)
    parser.add_argument('--min_depth', type=float, default=0.55)
    parser.add_argument('--fov', type=float, default=220)

    parser.add_argument('--input_width', type=int, default=500)
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--output_width', type=int, default=512)
    parser.add_argument('--output_height', type=int, default=256)

    args = parser.parse_args()
    inference(args)
