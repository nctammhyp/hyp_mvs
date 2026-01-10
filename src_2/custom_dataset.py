import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.ocam import OcamModel

class OmniThingsDataset(Dataset):
    def __init__(self, root_dir, list_file, equirect_size=(256, 512), 
                 num_invdepth=192, fov=220):
        self.root_dir = root_dir
        self.equirect_size = equirect_size
        self.num_invdepth = num_invdepth
        self.fov = fov
        
        self.min_depth = 0.55
        self.max_depth = 100.0
        self.cam_names = ['cam1', 'cam2', 'cam3', 'cam4']
        
        # Đọc list file
        with open(list_file, 'r') as f:
            self.filenames = [x.strip() for x in f.readlines() if x.strip()]
            
        # Parse Ocam
        self.ocams = []
        for i, cname in enumerate(self.cam_names):
            ocam_path = os.path.join(root_dir, f'o{cname}.txt')
            # Fix lỗi parsing int('1.0000') ở đây
            config = self._parse_ocam(ocam_path, i)
            ocam = OcamModel()
            ocam.setConfig(config)
            self.ocams.append(ocam)
            
        print("Pre-computing Geometry Grids...")
        self.sweep_grids = self._build_sweep_grids()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        imgs = []
        
        # Kích thước resize ảnh đầu vào để tiết kiệm VRAM (OmniThings gốc rất lớn)
        input_h, input_w = 480, 500 

        for cam in self.cam_names:
            img_path = os.path.join(self.root_dir, cam, fname)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize ảnh input để giảm tải Feature Net
                img = cv2.resize(img, (input_w, input_h))
                
            img = img.astype(np.float32) / 255.0
            t_img = torch.from_numpy(img).permute(2, 0, 1)
            imgs.append(t_img)

        gt_path = os.path.join(self.root_dir, 'depth_train_640', fname)
        gt_idx = self._load_gt(gt_path)

        return {'imgs': imgs, 'gt': gt_idx, 'name': fname}

    def _load_gt(self, path):
        depth_raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            return torch.zeros(self.equirect_size, dtype=torch.float32) - 1

        depth_m = depth_raw.astype(np.float32) / 100.0
        depth_m[depth_m < 0.1] = 0.1
        inv_depth = 1.0 / depth_m
        
        h, w = self.equirect_size
        inv_depth = cv2.resize(inv_depth, (w, h), interpolation=cv2.INTER_NEAREST)
        
        max_inv = 1.0 / self.min_depth
        min_inv = 1.0 / self.max_depth
        step = (max_inv - min_inv) / (self.num_invdepth - 1)
        
        idx = (inv_depth - min_inv) / step
        mask = (inv_depth >= min_inv) & (inv_depth <= max_inv)
        idx[~mask] = -1
        return torch.from_numpy(idx).float()

    def _build_sweep_grids(self):
        grids = []
        H, W = self.equirect_size
        for d in range(self.num_invdepth):
            xv, yv = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
            grid = np.stack([xv, yv], axis=-1) 
            grids.append(torch.from_numpy(grid).float())
        return grids

    def _parse_ocam(self, path, idx):
        try:
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            
            # --- FIX: dùng float() trước khi int() ---
            poly = [float(x) for x in lines[0].split()[1:]]
            inv_poly = [float(x) for x in lines[1].split()[1:]]
            
            # Center
            xc, yc = float(lines[2].split()[0]), float(lines[2].split()[1])
            
            # Image Size: Dòng gây lỗi '1.000000'
            try:
                raw_h = lines[3].split()[0]
                raw_w = lines[3].split()[1]
                h = int(float(raw_h)) # Fix ở đây
                w = int(float(raw_w)) # Fix ở đây
            except:
                h, w = 480, 500 # Default nếu lỗi dòng này
                
        except Exception as e:
            print(f"Warning parsing ocam {path}: {e}")
            # Fallback values
            return {'cam_id': idx, 'poly': [0], 'inv_poly': [0], 'center': [0,0], 
                    'affine': [1,0,0], 'image_size': [480, 500], 'max_fov': 220, 
                    'invalid_mask': '', 'pose': [0,0,0,0,0,0]}

        rots = [0, -90, 180, 90]
        pose = [0, np.deg2rad(rots[idx]), 0, 0, 0, 0]

        return {
            'cam_id': idx,
            'poly': [len(poly)] + poly,
            'inv_poly': [len(inv_poly)] + inv_poly,
            'center': [xc, yc],
            'affine': [1,0,0],
            'image_size': [h, w],
            'max_fov': self.fov,
            'invalid_mask': 'dummy',
            'pose': pose
        }