import argparse
import os
from os.path import join
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from models import OmniMVS, SphericalSweeping
from utils import InvDepthConverter, evaluation_metrics


def parse_args():
    parser = argparse.ArgumentParser("Eval OmniMVS")

    parser.add_argument(
        'root_dir',
        nargs='?',
        default=r'F:/tmp/datasets/omnithings',
        help='path to dataset'
    )
    parser.add_argument(
        '--val-list',
        default=r'.\dataloader\omnithings_val.txt',
        help='validation txt'
    )
    parser.add_argument(
        '--checkpoint',
        default=r'F:\omnimvs_pytorch\checkpoints\hyp_v1\checkpoints_5.pth',
        help='trained model .pth'
    )

    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--ndisp', default=64, type=int)
    parser.add_argument('--min_depth', default=0.55, type=float)
    parser.add_argument('--fov', default=220, type=float)

    parser.add_argument('--input_width', default=500, type=int)
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--output_width', default=512, type=int)
    parser.add_argument('--output_height', default=256, type=int)

    parser.add_argument('--out-dir', default='eval_results', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print("Using device:", device)
    print("Saving results to:", args.out_dir)

    # =====================
    # Model
    # =====================
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

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    model = nn.DataParallel(model).to(device)
    model.eval()

    # =====================
    # Dataset
    # =====================
    image_size = (args.input_width, args.input_height)
    depth_size = (args.output_width, args.output_height)

    tfm = transforms.Compose([
        Resize(image_size, depth_size),
        ToTensor(),
        Normalize()
    ])

    valset = OmniStereoDataset(
        args.root_dir,
        args.val_list,
        transform=tfm,
        fov=args.fov
    )

    loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    print(f"Validation samples: {len(valset)}")

    # =====================
    # Depth converter
    # =====================
    invd_0 = model.module.inv_depths[0]
    invd_max = model.module.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    preds, gts = [], []

    # =====================
    # Inference
    # =====================
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            for k in batch:
                batch[k] = batch[k].to(device)

            pred = model(batch)               # (B,1,H_out,W_out)
            gt_idepth = batch['idepth']
            gt_idx = converter.invdepth_to_index(gt_idepth)

            preds.append(pred.cpu())
            gts.append(gt_idx.cpu())

            # ========= Visualization =========

            # input images (4 cams): (1,H,W)
            imgs = [
                0.5 * batch[cam][0].cpu() + 0.5
                for cam in model.module.cam_list
            ]

            # pred / gt -> (1,H_out,W_out)
            pred_vis = (pred[0:1] / args.ndisp).cpu()
            gt_vis = (gt_idx[0:1] / args.ndisp).cpu()

            # resize depth to input size
            pred_vis = F.interpolate(
                pred_vis.unsqueeze(0),
                size=(args.input_height, args.input_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            gt_vis = F.interpolate(
                gt_vis.unsqueeze(0),
                size=(args.input_height, args.input_width),
                mode='nearest'
            ).squeeze(0)

            # grid: cam0 cam1 cam2 / cam3 pred gt
            grid = make_grid(
                imgs + [pred_vis, gt_vis],
                nrow=3,
                padding=5,
                pad_value=1.0
            )

            save_image(
                grid,
                join(args.out_dir, f'{i:05d}.png')
            )

    # =====================
    # Metrics
    # =====================
    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)

    errors, names = evaluation_metrics(preds, gts, args.ndisp)

    print("\nEvaluation metrics:")
    print(OrderedDict(zip(names, errors)))


if __name__ == '__main__':
    main()
