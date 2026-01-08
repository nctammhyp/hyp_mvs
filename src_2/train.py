# train.py
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Dataset
# from module.network import OmniMVSNet
from module.network_fixed import OmniMVSNet

from utils.common import LOG_INFO
from torch.utils.data import DataLoader, Subset

# =========================
# CONFIG
# =========================
DB_ROOT = '/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets'
DB_NAME = 'omnithings'
BATCH_SIZE = 2
NUM_WORKERS = 0  # Windows safe
NUM_EPOCHS = 5
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# LOAD DATASET
# =========================
train_dataset = Dataset(DB_NAME, db_root=DB_ROOT, train=True)
# train_loader = DataLoader(train_dataset,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True,
#                           num_workers=NUM_WORKERS,
#                           pin_memory=True)

# Chọn subset: ví dụ dùng 100 frame đầu tiên
subset_indices = list(range(10))  # hoặc np.arange(100)
train_subset = Subset(train_dataset, subset_indices)

# DataLoader cho subset
train_loader = DataLoader(train_subset,
                          batch_size=BATCH_SIZE,
                        #   shuffle=True,
                        #   num_workers=4,
                        #   pin_memory=True
                       )

print("#### done dataloader  #####")

# =========================
# LOAD NETWORK
# =========================
net_opts = dict(num_invdepth=train_dataset.num_invdepth)
net = OmniMVSNet(net_opts).to(DEVICE)
net.train()

# =========================
# LOSS AND OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(net.parameters(), lr=LR)

# =========================
# TRAIN LOOP
# =========================
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time

scaler = GradScaler()  # cho mixed precision

for epoch in range(NUM_EPOCHS):
    LOG_INFO(f'===== Epoch {epoch+1}/{NUM_EPOCHS} =====')
    epoch_loss = 0.0
    tic_epoch = time.time()

    # tqdm để thấy tiến trình batch
    for batch_idx, (imgs, gt, valid, *rest) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        # Move to device
        imgs = [img.to(DEVICE).float() for img in imgs]
        gt = gt.to(DEVICE).long()
        valid = valid.to(DEVICE)

        optimizer.zero_grad()

        with autocast():  # mixed precision
            invdepth_idx, prob, _ = net(imgs, train_dataset.grids, out_cost=True)

            mask = valid
            prob_flat = prob.permute(0,2,3,1)[mask]  # [N, D]
            gt_flat = gt[mask]  # [N]

            if gt_flat.numel() == 0:
                continue

            loss = criterion(prob_flat, gt_flat)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        if batch_idx % 5 == 0:
            LOG_INFO(f'Batch {batch_idx} | Loss: {loss.item():.4f}')

    toc_epoch = time.time() - tic_epoch
    avg_loss = epoch_loss / len(train_loader)
    LOG_INFO(f'Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f} | Time: {toc_epoch:.2f}s')


# =========================
# SAVE MODEL
# =========================
os.makedirs('checkpoints', exist_ok=True)
save_path = os.path.join('checkpoints', f'omnimvs_epoch{NUM_EPOCHS}.pt')
torch.save({
    'net_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': NUM_EPOCHS,
}, save_path)
LOG_INFO(f'Model saved to {save_path}')
