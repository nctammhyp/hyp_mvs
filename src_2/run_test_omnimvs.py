# run_test_omnimvs.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
# Modified: use fixed default arguments (no CLI required)
#
import os
import sys
import time
import multiprocessing
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Torch libs
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Internal modules
from dataset import Dataset, MultiDataset
from utils.common import *
from utils.image import *
from module.network import OmniMVSNet
from module.loss_functions import *

# =========================
# GPU setup
# =========================
GPU_ID = 0
os.putenv('CUDA_VISIBLE_DEVICES', str(GPU_ID))
torch.backends.cudnn.benchmark = True
torch.backends.cuda.benchmark = True

# =========================
# Options (DEFAULT ONLY)
# =========================
opts = Edict()

# ---- Model checkpoint (DEFAULT)
opts.snapshot_path = r'F:\omnimvs_pytorch\checkpoints\omnimvs_default\tiny_plus_ft.pt'

# ---- Dataset (DEFAULT)
opts.dbname = 'itbt_sample'
opts.db_root = r'F:\omnimvs_pytorch\src_2\data'

# ---- Dataset options
opts.data_opts = Edict()
opts.data_opts.phi_deg = 45.0
opts.data_opts.num_invdepth = 192
opts.data_opts.equirect_size = (160, 640)

# ---- Network options
opts.net_opts = Edict()
opts.net_opts.num_invdepth = opts.data_opts.num_invdepth

# ---- Output / visualization
opts.vis = True
opts.save_result = True
opts.save_misc = True

snapshot_name = osp.splitext(osp.basename(opts.snapshot_path))[0]
opts.result_dir = osp.join('./results', opts.dbname, snapshot_name)
opts.out_invdepth_fmt = osp.join(opts.result_dir, '%05d.tiff')
opts.out_entropy_fmt = osp.join(opts.result_dir, '%05d_entropy.tiff')
opts.out_misc_fmt = osp.join(opts.result_dir, '%05d.png')

# ---- Visualization window
if opts.vis:
    fig = plt.figure(frameon=False, figsize=(25, 10), dpi=40)
    plt.ion()
    plt.show()


# =========================
# Main
# =========================

def main():
    # Dataset
    data = Dataset(opts.dbname, opts.data_opts, db_root=opts.db_root)
    dbloader = torch.utils.data.DataLoader(data, shuffle=False)

    # Load snapshot
    if not osp.exists(opts.snapshot_path):
        sys.exit('%s does not exist' % opts.snapshot_path)

    snapshot = torch.load(opts.snapshot_path)

    # Network
    opts.net_opts.CH = snapshot['CH']
    net = OmniMVSNet(opts.net_opts).cuda()
    net.load_state_dict(snapshot['net_state_dict'])
    net.eval()

    # Precomputed grids
    grids = [torch.tensor(grid, requires_grad=False).cuda()
             for grid in data.grids]

    # Output directory
    if not osp.exists(opts.result_dir):
        os.makedirs(opts.result_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % opts.result_dir)

    errors = np.zeros((data.data_size, 5))

    # =========================
    # Inference loop
    # =========================
    for d in range(data.data_size):
        fidx = data.frame_idx[d]
        imgs, gt, valid, raw_imgs = data.loadSample(fidx)

        tic = time.time()
        imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]

        with torch.no_grad():
            invdepth_idx, prob, _ = net(imgs, grids, out_cost=True)

        invdepth_idx = toNumpy(invdepth_idx)
        invdepth = data.indexToInvdepth(invdepth_idx)
        entropy = toNumpy(torch.sum(-torch.log(prob + EPS) * prob, 0))
        toc = time.time() - tic

        # ---- Error metrics
        if len(gt) > 0:
            errors[d, :] = data.evalError(invdepth_idx, gt, valid)

        # ---- Visualization
        toc2 = 0
        if opts.vis or opts.save_misc:
            tic2 = time.time()
            vis_img = data.makeVisImage(raw_imgs, invdepth, entropy, gt)

            if opts.vis:
                fig.clf()
                plt.imshow(vis_img)
                plt.axis('off')
                plt.tight_layout()
                plt.draw()
                plt.pause(0.5)

            if opts.save_misc:
                writeImage(vis_img, opts.out_misc_fmt % fidx)

            toc2 += time.time() - tic2

        # ---- Save results
        if opts.save_result:
            tic2 = time.time()
            data.writeInvdepth(invdepth, opts.out_invdepth_fmt % fidx)
            data.writeEntropy(entropy, opts.out_entropy_fmt % fidx)
            toc2 += time.time() - tic2

        LOG_INFO('Process %d/%d | MAE: %.3f | net: %.3f s | io+vis: %.3f s' %
                 (d + 1, data.data_size, errors[d, 3], toc, toc2))


if __name__ == '__main__':
    main()
