# ================= FULL TRAIN.PY (WINDOWS SAFE VERSION) =================
# - Wrapped in main() for Windows multiprocessing
# - Hardcoded args
# - Cache sweep grids
# - Save input/pred/gt images
# - Save checkpoints
# ======================================================================

import os
import json
import argparse
from datetime import datetime
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from models import OmniMVS, SphericalSweeping
from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from utils import InvDepthConverter

from tqdm import tqdm

import random
import numpy as np
import torch
import numpy as np
import torch
from torch.utils.data import Subset




def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


def main():
    # ===================== ARGS =====================
    parser = argparse.ArgumentParser(description='Training for OmniMVS',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--pretrained', default=None)
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
    parser.add_argument('--save-interval', type=int, default=100)

    # ===== HARDCODED NOTEBOOK STYLE =====
    root_dir = '/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/omnithings'
    file_list = '-t dataloader/omnithings_train.txt'

    args = parser.parse_args(
        f'{root_dir} {file_list} --lr 1e-3 --ndisp 48'.split()
    )

    print(args)

    # ===================== DEVICE =====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type != 'cpu':
        cudnn.benchmark = True

    # ===================== MODEL =====================
    print('=> Setup model')
    sweep = SphericalSweeping(args.root_dir, h=args.output_height, w=args.output_width)
    model = OmniMVS(sweep, args.ndisp, args.min_depth,
                    h=args.output_height, w=args.output_width)

    invd_0 = model.inv_depths[0]
    invd_max = model.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    model = model.to(device)
    start_epoch = 0

    # ===================== CACHE =====================
    print('=> Precompute sweep grid')
    pool = ThreadPoolExecutor(5)
    futures = []
    for i in range(4):
        for d in model.depths[::2]:
            futures.append(pool.submit(sweep.get_grid, i, d))
    pool.shutdown()
    print('=> Done cache')

    # ===================== OPTIMIZER =====================
    print('=> setting optimizer')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print('=> setting scheduler')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ===================== LOG FOLDER =====================
    timestamp = datetime.now().strftime('%m%d-%H%M')
    log_folder = os.path.join('checkpoints', f'{args.arch}_{timestamp}')
    os.makedirs(log_folder, exist_ok=True)

    img_dir = os.path.join(log_folder, 'images')
    os.makedirs(img_dir, exist_ok=True)

    with open(os.path.join(log_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ===================== DATALOADER =====================
    image_size = (args.input_width, args.input_height)
    depth_size = (args.output_width, args.output_height)

    train_transform = transforms.Compose([
        Resize(image_size, depth_size),
        ToTensor(),
        Normalize()
    ])

    trainset = OmniStereoDataset(args.root_dir, args.train_list, transform=train_transform)
    print(f'{len(trainset)} samples were found.')

    # train_loader = DataLoader(
    #     trainset,
    #     args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True
    # )

    # ===== TỈ LỆ LẤY =====
    percent = 0.01  # ví dụ: lấy 20%
    num_samples = len(trainset)
    subset_size = int(percent * num_samples)

    # ===== RANDOM INDICES (NHƯNG CỐ ĐỊNH) =====
    indices = np.random.choice(num_samples, subset_size, replace=False)

    sub_trainset = Subset(trainset, indices)

    train_loader = DataLoader(
        sub_trainset,
        args.batch_size,
        shuffle=True,   # shuffle batch, không ảnh hưởng subset
    )

    print(f"Original: {len(trainset)} | Subset: {len(sub_trainset)}")


    # ===================== TRAINING =====================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []

        pbar = tqdm(train_loader)
        for idx, batch in enumerate(pbar):
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            pred = model(batch)
            gt_idepth = batch['idepth']
            gt_invd_idx = converter.invdepth_to_index(gt_idepth)

            loss = nn.L1Loss()(pred, gt_invd_idx)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(OrderedDict(epoch=f"{epoch:>2}", loss=f"{loss.item():.4f}"))

            # ============ SAVE IMAGES ============
            if idx % args.save_interval == 0:
                imgs = []
                for cam in model.cam_list:
                    imgs.append(0.5 * batch[cam][0] + 0.5)

                img_grid = make_grid(imgs, nrow=2, padding=5, pad_value=1)

                save_image(
                    img_grid,
                    os.path.join(img_dir, f"epoch{epoch:03d}_iter{idx:05d}_input.png")
                )

                save_image(
                    pred / args.ndisp,
                    os.path.join(img_dir, f"epoch{epoch:03d}_iter{idx:05d}_pred.png")
                )

                save_image(
                    gt_invd_idx / args.ndisp,
                    os.path.join(img_dir, f"epoch{epoch:03d}_iter{idx:05d}_gt.png")
                )

        scheduler.step()
        ave_loss = sum(losses) / len(losses)

        print(f"Epoch:{epoch}, Loss average:{ave_loss:.4f}")

        save_data = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ave_loss': ave_loss,
            'ndisp': model.ndisp,
            'min_depth': model.min_depth,
            'output_width': model.w,
            'output_height': model.h,
        }

        torch.save(save_data, os.path.join(log_folder, f'checkpoint_{epoch}.pth'))

    print('Training finished!')


if __name__ == '__main__':
    main()
