import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.common import EPS
from ocamcamera import OcamCamera


class OmniThingsDataset(Dataset):
    def __init__(
        self,
        root_dir,
        list_file,
        equirect_size=(160, 640),
        num_invdepth=192,
        min_depth=0.5
    ):
        self.root = root_dir
        self.equirect_size = equirect_size
        self.num_invdepth = num_invdepth
        self.min_depth = min_depth
        self.max_depth = 1.0 / EPS

        # ---- filenames ----
        with open(list_file, 'r') as f:
            self.filenames = [l.strip() for l in f if l.strip()]

        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.depth_dir = 'depth_train_640'

        # ---- ocam (NO invalid_mask) ----
        self.ocams = []
        self.valid_masks = []
        H, W = equirect_size
        for i in range(1, 5):
            self.ocams.append(OcamCamera(os.path.join(root_dir, f'ocam{i}.txt')))
            self.valid_masks.append(np.ones((H, W), dtype=bool))

        # ---- inverse depth sampling ----
        self.min_inv = 1.0 / self.max_depth
        self.max_inv = 1.0 / self.min_depth
        self.step_inv = (self.max_inv - self.min_inv) / (num_invdepth - 1)

        print(f"[INFO] OmniThingsDataset loaded: {len(self.filenames)} samples")

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, path, valid_mask):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img[~valid_mask] = 0.0
        return img

    def _load_invdepth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = depth / 100.0  # cm → m
        invdepth = 1.0 / np.maximum(depth, 1e-6)
        return invdepth

    def _invdepth_to_index(self, invdepth):
        return (invdepth - self.min_inv) / self.step_inv

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        imgs = []
        for cam, mask in zip(self.cam_list, self.valid_masks):
            path = os.path.join(self.root, cam, fname)
            img = self._load_image(path, mask)
            imgs.append(torch.from_numpy(img).unsqueeze(0))  # [1,H,W]

        depth_path = os.path.join(self.root, self.depth_dir, fname)
        invdepth = self._load_invdepth(depth_path)
        gt_idx = self._invdepth_to_index(invdepth)

        valid = np.isfinite(gt_idx) & (gt_idx >= 0) & (gt_idx < self.num_invdepth)

        return {
            'imgs': imgs,
            'gt_idx': torch.from_numpy(gt_idx).float(),
            'valid': torch.from_numpy(valid)
        }
