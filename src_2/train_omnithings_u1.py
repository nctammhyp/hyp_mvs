import os
import argparse as arg_std
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F # <--- THÊM DÒNG NÀY
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict as Edict
import numpy as np

# Imports custom modules
from custom_dataset import OmniThingsDataset
from module.network import OmniMVSNet
from module.loss_functions import *
from torch.utils.data import Subset


def main():
    parser = arg_std.ArgumentParser()
    # parser.add_argument('--root', type=str, default=r'F:\tmp\datasets\omnithings')
    # parser.add_argument('--train_list', type=str, default=r'F:\hyp_mvs_clean\dataloader\omnithings_train.txt')

    parser.add_argument(
    'root_dir',
    nargs='?',
    default='/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/omnithings'
    )

    parser.add_argument(
        '-t', '--train-list',
        default='../dataloader/omnithings_train.txt'
    )

    parser.add_argument(
        '-v', '--val-list',
        default='./dataloader/omnithings_val.txt'
    )
    
    # --- CẤU HÌNH ---
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ndisp', type=int, default=48)
    
    parser.add_argument('--h', type=int, default=256)
    parser.add_argument('--w', type=int, default=512)
    parser.add_argument('--fov', type=float, default=220.0)
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints/omni_v1')
    parser.add_argument('--resume', type=str, default='')

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    torch.cuda.empty_cache()

    print(f"=== Training OmniMVS ===")
    print(f"Batch: {args.batch_size}, Ndisp: {args.ndisp}, Device: {device}")

    # # 1. Dataset
    # train_dataset = OmniThingsDataset(
    #     root_dir=args.root,
    #     list_file=args.train_list,
    #     equirect_size=(args.h, args.w),
    #     num_invdepth=args.ndisp,
    #     fov=args.fov
    # )
    
    # train_loader = DataLoader(
    #     train_dataset, 
    #     batch_size=args.batch_size, 
    #     shuffle=True, 
    #     num_workers=2,
    #     pin_memory=True
    # )

        # 1. Dataset Initialization
    # Logic tạo Grid (LUT) sẽ được xử lý tự động bên trong dataset
    train_dataset = OmniThingsDataset(
        root_dir=args.root,
        list_file=args.train_list,
        equirect_size=(args.h, args.w),
        num_invdepth=args.ndisp,
        fov=args.fov
    )
    
    # train_loader = DataLoader(
    #     train_dataset, 
    #     batch_size=args.batch_size, 
    #     shuffle=True, 
    #     num_workers=4, 
    #     pin_memory=True
    # )

    total_size = len(train_dataset)
    subset_size = int(0.1 * total_size)

    indices = list(range(subset_size))
    train_subset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=4,
        # pin_memory=True
    )


    # 2. Model
    model_opts = Edict()
    model_opts.CH = 32
    model_opts.num_invdepth = args.ndisp
    model_opts.use_rgb = True 
    
    model = OmniMVSNet(model_opts).to(device)

    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.SmoothL1Loss(reduction='none')

    # Resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")

    # 4. Grids to GPU
    grids_gpu = [g.to(device) for g in train_dataset.sweep_grids]

    # 5. Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")

        for i, batch in enumerate(pbar):
            imgs = [img.to(device) for img in batch['imgs']]
            gt_idx = batch['gt'].to(device)

            optimizer.zero_grad()

            try:
                # Forward
                # pred_disp shape: [B, H/2, W/2] do stride trong network
                pred_disp = model(imgs, grids_gpu, out_cost=False)

                # --- FIX: Upsample pred_disp lên kích thước GT [B, H, W] ---
                if pred_disp.shape[-2:] != gt_idx.shape[-2:]:
                    # Thêm dim channel để interpolate: [B, H, W] -> [B, 1, H, W]
                    pred_disp = pred_disp.unsqueeze(1)
                    pred_disp = F.interpolate(pred_disp, size=(args.h, args.w), 
                                              mode='bilinear', align_corners=False)
                    pred_disp = pred_disp.squeeze(1)

                mask = (gt_idx >= 0) & (gt_idx < args.ndisp)
                if mask.sum() > 0:
                    loss = criterion(pred_disp[mask], gt_idx[mask]).mean()
                    loss.backward()
                    optimizer.step()
                    
                    val = loss.item()
                    total_loss += val
                    pbar.set_postfix({'loss': f"{val:.4f}"})
                else:
                    pbar.set_postfix({'loss': "0.00"})
            
            except torch.cuda.OutOfMemoryError:
                print("\n[ERR] CUDA OOM! Skipping batch.")
                torch.cuda.empty_cache()
                continue

        scheduler.step()
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")

        save_path = os.path.join(args.save_dir, f'checkpoint_ep{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
            'args': args
        }, save_path)

if __name__ == '__main__':
    main()