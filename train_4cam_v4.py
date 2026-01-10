import os, json, argparse, torch, random
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from models import OmniMVS, SphericalSweeping
from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from utils import InvDepthConverter

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', nargs='?', default='/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/omnithings')
    parser.add_argument('-t', '--train-list', default='./dataloader/omnithings_train.txt')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float) # AdamW dùng LR nhỏ hơn
    parser.add_argument('--ndisp', type=int, default=48)
    parser.add_argument('--save-interval', type=int, default=50)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    sweep = SphericalSweeping(args.root_dir, h=256, w=512)
    model = OmniMVS(sweep, args.ndisp, min_depth=0.55, h=256, w=512).to(device)
    
    converter = InvDepthConverter(args.ndisp, model.inv_depths[0], model.inv_depths[-1])

    # Optimizer & Loss (THAY ĐỔI QUAN TRỌNG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss() # Giúp hội tụ ổn định hơn L1

    # Dataloader
    transform = transforms.Compose([
        Resize((500, 480), (512, 256)),
        ToTensor(),
        Normalize() # Sẽ normalize 3 kênh RGB
    ])
    
    trainset = OmniStereoDataset(args.root_dir, args.train_list, transform=transform)
    
    # Tăng tỷ lệ subset lên để mô hình thấy đủ độ đa dạng
    indices = np.random.choice(len(trainset), int(0.1 * len(trainset)), replace=False)
    train_loader = DataLoader(Subset(trainset, indices), batch_size=args.batch_size, shuffle=True)

    log_folder = os.path.join('checkpoints', f'omni_{datetime.now().strftime("%m%d-%H%M")}')
    img_dir = os.path.join(log_folder, 'images')
    os.makedirs(img_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader)
        for idx, batch in enumerate(pbar):
            for k in batch: batch[k] = batch[k].to(device)

            pred = model(batch) # [B, H, W]
            gt_idepth = batch['idepth']
            gt_idx = converter.invdepth_to_index(gt_idepth).clamp(0, args.ndisp-1)

            loss = criterion(pred, gt_idx)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}")

            # ============ FIX LƯU ẢNH ============
            if idx % args.save_interval == 0:
                # Lấy tấm đầu tiên trong batch để visualize
                # Pred và GT cần .unsqueeze(1) để thành [1, 1, H, W] cho save_image
                p_vis = (pred[0:1] / args.ndisp).clamp(0, 1)
                g_vis = (gt_idx[0:1] / args.ndisp).clamp(0, 1)
                
                # Input image (nghịch đảo Normalize)
                input_vis = (batch['cam1'][0] * 0.5 + 0.5).clamp(0, 1)

                save_image(p_vis, os.path.join(img_dir, f"ep{epoch}_{idx}_pred.png"))
                save_image(g_vis, os.path.join(img_dir, f"ep{epoch}_{idx}_gt.png"))
                save_image(input_vis, os.path.join(img_dir, f"ep{epoch}_{idx}_input.png"))

        torch.save(model.state_dict(), os.path.join(log_folder, f'model_{epoch}.pth'))

if __name__ == '__main__':
    main()