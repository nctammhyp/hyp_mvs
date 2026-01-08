# dataset.py
import os
import os.path as osp
import numpy as np
import cv2
import torch
import torch.utils.data
from easydict import EasyDict as Edict

from utils.common import *
from utils.image import *
import utils.dbhelper
from utils.log import LOG_INFO

def makeSphericalRays(equirect_size: (int, int), phi_deg: float, phi2_deg=-1.0):
    h, w = equirect_size
    xs, ys = np.meshgrid(range(w), range(h))
    w_2, h_2 = w / 2.0, (h - 1) / 2.0
    xs = (xs - w_2) / w_2 * np.pi + (np.pi / 2.0)
    if phi2_deg > 0.0:
        med = np.deg2rad(sum(phi2_deg - phi_deg) / 2.0)
        med2 = np.deg2rad((phi2_deg + phi_deg) / 2.0)
        ys = (ys - h_2) / h_2 * med2 - med
    else:
        ys = (ys - h_2) / h_2 * np.deg2rad(phi_deg)

    X = -np.cos(ys) * np.cos(xs)
    Y = np.sin(ys)
    Z = np.cos(ys) * np.sin(xs)
    rays = np.concatenate((np.reshape(X, [1, -1]),
                           np.reshape(Y, [1, -1]),
                           np.reshape(Z, [1, -1]))).astype(np.float64)
    return rays

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dbname: str, db_opts=None, load_lut=True, train=True, db_root='../data'):
        super().__init__()
        self.dbname = dbname.lower()
        self.db_path = osp.join(db_root, self.dbname)

        # default opts
        opts = Edict()
        opts.img_fmt = 'cam%d/%05d.png'
        self.cam_offset = 1
        opts.lut_fmt = 'lt_(%d,%d,%d).hwd'
        opts.gt_depth_fmt = 'omnidepth_gt_%d/%05d.tiff'
        opts.equirect_size, opts.num_invdepth = [160, 640], 192
        opts.phi_deg, opts.phi2_deg = 45, -1.0
        opts.min_depth = 0.5
        opts.max_depth = 1000.0
        opts.max_fov = 220.0
        opts.read_input_image = True
        opts.upsample_output = False
        opts.start, opts.step, opts.end = 1, 1, 1000
        opts.train_idx, opts.test_idx = [], []
        opts.gt_phi = 0.0
        opts.dtype = 'gt'
        opts.use_rgb = False

        opts, self.ocams = utils.dbhelper.loadDBConfigs(self.dbname, self.db_path, opts)
        opts = argparse(opts, db_opts)

        self.opts = opts
        self.img_fmt, self.lut_fmt = opts.img_fmt, opts.lut_fmt
        self.gt_depth_fmt = opts.gt_depth_fmt
        self.frame_idx = list(range(opts.start, opts.end + opts.step, opts.step))
        self.train_idx, self.test_idx = opts.train_idx, opts.test_idx
        self.gt_phi = opts.gt_phi
        self.dtype = opts.dtype
        self.use_rgb = opts.use_rgb

        self.equirect_size = opts.equirect_size
        self.min_depth, self.max_depth = opts.min_depth, opts.max_depth
        self.max_theta = np.deg2rad(opts.max_fov) / 2.0
        self.phi_deg, self.phi2_deg = opts.phi_deg, opts.phi2_deg
        self.num_invdepth = opts.num_invdepth
        self.read_input_image = opts.read_input_image
        self.upsample_output = opts.upsample_output
        self.data_size = len(self.frame_idx)
        self.train_size = len(self.train_idx)
        self.test_size = len(self.test_idx)
        self.train = train

        self.__initSweep(load_lut)

    # ===========================
    # Load images cho 1 frame
    # ===========================
    def loadImages(self, fidx, read_raw=True, use_rgb=False):
        imgs, raw_imgs = [], []
        num_cams = len(self.ocams)
        for cam_id in range(num_cams):
            img_path = osp.join(self.db_path, self.img_fmt % (cam_id + self.cam_offset, fidx))
            img = readImage(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            if not use_rgb and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            im_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
            if im_tensor.ndim == 2:
                im_tensor = im_tensor.unsqueeze(0)
            else:
                im_tensor = im_tensor.permute(2, 0, 1)
            imgs.append(im_tensor)

            if read_raw:
                raw_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
                if raw_tensor.ndim == 2:
                    raw_tensor = raw_tensor.unsqueeze(0)
                else:
                    raw_tensor = raw_tensor.permute(2, 0, 1)
                raw_imgs.append(raw_tensor)

        return imgs, raw_imgs

    # ===========================
    # Build spherical sweep lookup table
    # ===========================
    def buildLookupTable(self, shape):
        H, W = shape
        grids = []
        for d in range(self.num_invdepth):
            xs = np.linspace(-1, 1, W, dtype=np.float32)
            ys = np.linspace(-1, 1, H, dtype=np.float32)
            xv, yv = np.meshgrid(xs, ys)
            grid = np.stack([xv, yv], axis=-1)  # H x W x 2
            grids.append(torch.from_numpy(grid))  # Tensor
        return grids

    def __initSweep(self, load_lut=True):
        h, w = self.equirect_size
        self.rays = makeSphericalRays(self.equirect_size, self.phi_deg, self.phi2_deg)
        self.min_invdepth = 1.0 / self.max_depth
        self.max_invdepth = 1.0 / self.min_depth
        self.sample_step_invdepth = (self.max_invdepth - self.min_invdepth) / (self.num_invdepth - 1.0)
        self.invdepths = np.arange(self.min_invdepth, self.max_invdepth + self.sample_step_invdepth,
                                   self.sample_step_invdepth, dtype=np.float64)
        if load_lut:
            self.__loadOrBuildLookupTable()

    def __loadOrBuildLookupTable(self):
        h, w = self.equirect_size
        if self.upsample_output:
            h, w = int(h / 2), int(w / 2)
        path = osp.join(self.db_path, self.lut_fmt % (h, w, self.num_invdepth))
        if not osp.exists(path):
            LOG_INFO(f'Lookup table not found: "{path}"')
            LOG_INFO('Build lookup table...')
            self.grids = self.buildLookupTable((h, w))
            np.concatenate([g.unsqueeze(0).numpy() for g in self.grids], axis=0).tofile(path)
            LOG_INFO(f'Lookup table saved: "{path}"')
        else:
            LOG_INFO(f'Load lookup table: "{path}"')
            grids = np.fromfile(path, dtype=np.float32).reshape([self.num_invdepth, h, w, 2])
            self.grids = [torch.from_numpy(grids[i, ...]) for i in range(self.num_invdepth)]

    # ===========================
    # Dataset length & getitem
    # ===========================
    def __len__(self):
        return len(self.train_idx) if self.train else len(self.test_idx)

    def __getitem__(self, i):
        return self.loadTrainSample(i) if self.train else self.loadTestSample(i, self.read_input_image)

    # ===========================
    # Read invdepth
    # ===========================
    def readInvdepth(self, path: str) -> np.ndarray:
        _, ext = osp.splitext(path)
        if ext == '.png':
            step_invdepth = (self.max_invdepth - self.min_invdepth) / 65500.0
            quantized_inv_index = readImage(path).astype(np.float32)
            invdepth = self.min_invdepth + quantized_inv_index * step_invdepth
            return invdepth
        elif ext in ['.tif', '.tiff']:
            gt = readImageFloat(path)
            if isinstance(gt, tuple):
                gt = gt[0]
            return np.array(gt, dtype=np.float32)
        else:
            return np.fromfile(path, dtype=np.float32)

    # ===========================
    # Load GT và sample
    # ===========================
    def loadGTInvdepthIndex(self, fidx, remove_gt_noise=True, morph_win_size=5):
        h, w = self.equirect_size
        gt_depth_file = osp.join(self.db_path, self.gt_depth_fmt % (w, fidx))
        gt = self.readInvdepth(gt_depth_file)
        if gt is None or len(gt) == 0:
            return np.array([])

        gt_h = gt.shape[0]
        if h < gt_h:
            sh = int(round((gt_h - h) / 2.0))
            gt = gt[sh:sh + h, :]

        gt_idx = self.invdepthToIndex(gt)

        if not remove_gt_noise:
            return gt_idx

        finite_depth = (gt >= 1e-3).astype(np.uint8)
        kernel = np.ones((morph_win_size, morph_win_size), np.uint8)
        closed = cv2.morphologyEx(finite_depth, cv2.MORPH_CLOSE, kernel)
        infinite_hole = np.logical_and(finite_depth == 0, closed > 0)
        gt_idx[infinite_hole] = -1

        return gt_idx

    def loadSample(self, fidx: int, read_input_image=True, varargin=None):
        opts = Edict()
        opts.remove_gt_noise = True
        opts.morph_win_size = 5
        opts = argparse(opts, varargin)

        imgs, raw_imgs = [], []
        if read_input_image:
            imgs, raw_imgs = self.loadImages(fidx, True, use_rgb=self.use_rgb)

        gt, valid = [], []
        if self.dtype == 'gt':
            gt = self.loadGTInvdepthIndex(fidx, opts.remove_gt_noise, opts.morph_win_size)
            valid = np.logical_and(gt >= 0, gt <= self.num_invdepth).astype(bool)

        return imgs, gt, valid, raw_imgs

    # ===========================
    # Train/Test helpers
    # ===========================
    def loadTrainSample(self, i, read_input_image=True, varargin=None):
        return self.loadSample(self.train_idx[i], read_input_image, varargin)

    def loadTestSample(self, i, read_input_image=True, varargin=None):
        return self.loadSample(self.test_idx[i], read_input_image, varargin)

    # ===========================
    # Helper convert index <-> invdepth
    # ===========================
    def indexToInvdepth(self, idx, start_index=0):
        return self.min_invdepth + (idx - start_index) * self.sample_step_invdepth

    def invdepthToIndex(self, inv_depth, start_index=0):
        return (inv_depth - self.min_invdepth) / self.sample_step_invdepth + start_index
