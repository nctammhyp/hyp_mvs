import torch
import torch.nn as nn
import torch.nn.functional as F
from module.basic import Conv2D, Conv3D, DeConv3D
from easydict import EasyDict as Edict

# -----------------------------
# Feature extractor
# -----------------------------
class FeatureLayers(nn.Module):
    def __init__(self, CH=32, use_rgb=False):
        super().__init__()
        layers = []
        in_channel = 3 if use_rgb else 1
        layers.append(Conv2D(in_channel, CH, 5, 2, 2))  # conv[1]
        layers += [Conv2D(CH, CH, 3, 1, 1) for _ in range(10)]  # conv[2-11]
        for d in range(2, 5):
            layers += [Conv2D(CH, CH, 3, 1, d, dilation=d) for _ in range(2)]
        layers.append(Conv2D(CH, CH, 3, 1, 1, bn=False, relu=False))  # conv[18]
        self.layers = nn.ModuleList(layers)

    def forward(self, im):
        x = self.layers[0](im)
        for i in range(1, 17, 2):
            x_ = self.layers[i](x)
            x = self.layers[i + 1](x_, residual=x)
        x = self.layers[17](x)
        return x

# -----------------------------
# Spherical Sweep
# -----------------------------
class SphericalSweep(nn.Module):
    def __init__(self, CH=32):
        super().__init__()
        self.transfer_conv = Conv2D(CH, CH, 3, 2, 1, bn=False, relu=False)

    def forward(self, feature, grids):
        B, C, H, W = feature.shape
        D = len(grids)
        sweep_list = []

        for d in range(D):
            g = grids[d]
            if not isinstance(g, torch.Tensor):
                g = torch.from_numpy(g).float()
            g = g.to(feature.device)
            if g.ndim == 3:
                g = g.unsqueeze(0).repeat(B, 1, 1, 1)
            sweep_list.append(F.grid_sample(feature, g, align_corners=True))

        sweep = torch.stack(sweep_list, dim=1)  # [B, D, C, H, W]
        # reshape để Conv2D nhận
        B, D, C, H, W = sweep.shape
        sweep_reshape = sweep.view(B * D, C, H, W)
        out = self.transfer_conv(sweep_reshape)
        _, C_out, H_out, W_out = out.shape
        out = out.view(B, D, C_out, H_out, W_out)
        # permute sang [B, C, D, H, W] cho Conv3D
        out = out.permute(0, 2, 1, 3, 4)
        return out

# -----------------------------
# Cost volume computation
# -----------------------------
class CostCompute(nn.Module):
    def __init__(self, CH=32):
        super().__init__()
        CH2 = CH * 2
        self.fusion = Conv3D(CH, CH, 3, 1, 1)
        convs = []
        convs += [Conv3D(CH, CH, 3, 1, 1) for _ in range(3)]
        convs += [Conv3D(CH, CH2, 3, 2, 1),
                  Conv3D(CH2, CH2, 3, 1, 1),
                  Conv3D(CH2, CH2, 3, 1, 1)]
        convs += [Conv3D(CH2, CH2, 3, 2, 1),
                  Conv3D(CH2, CH2, 3, 1, 1),
                  Conv3D(CH2, CH2, 3, 1, 1)]
        self.convs = nn.ModuleList(convs)
        self.deconv1 = DeConv3D(CH2, CH2, 3, 2, 1, out_pad=1)
        self.deconv2 = DeConv3D(CH2, CH2, 3, 2, 1, out_pad=1)
        self.deconv3 = DeConv3D(CH2, CH, 3, 2, 1, out_pad=1)
        self.deconv4 = DeConv3D(CH, 1, 3, 2, 1, out_pad=1, bn=False, relu=False)

    def forward(self, feats):
        c = self.fusion(feats)
        for i in range(0, len(self.convs), 3):
            c = self.convs[i](c)
            c = self.convs[i + 1](c)
            c = self.convs[i + 2](c)
        c = self.deconv1(c)
        c = self.deconv2(c)
        c = self.deconv3(c)
        costs = self.deconv4(c)
        return costs

# -----------------------------
# OmniMVSNet
# -----------------------------
class OmniMVSNet(nn.Module):
    def __init__(self, varargin=None):
        super().__init__()
        self.opts = Edict(varargin) if varargin else Edict()
        self.opts.CH = getattr(self.opts, 'CH', 32)
        self.opts.num_invdepth = getattr(self.opts, 'num_invdepth', 192)
        self.opts.use_rgb = getattr(self.opts, 'use_rgb', False)

        self.feature_layers = FeatureLayers(self.opts.CH, self.opts.use_rgb)
        self.spherical_sweep = SphericalSweep(self.opts.CH)
        self.cost_computes = CostCompute(self.opts.CH)
        self.register_buffer("disps",
                             torch.arange(0, self.opts.num_invdepth).view(1, -1, 1, 1).float())

    def forward(self, imgs, grids, upsample=False, out_cost=False):
        feats = [self.feature_layers(x) for x in imgs]
        spherical_feats_list = [self.spherical_sweep(feats[i], grids) for i in range(len(imgs))]
        spherical_feats = torch.cat(spherical_feats_list, dim=0)  # batch B*C,D,H,W
        costs = self.cost_computes(spherical_feats)

        if upsample:
            costs = F.interpolate(costs.squeeze(1), scale_factor=2, mode='bilinear', align_corners=True)
        else:
            costs = costs.squeeze(1)

        prob = F.softmax(costs, dim=1)
        disp = torch.sum(prob * self.disps, dim=1)

        if out_cost:
            return disp, prob, costs
        else:
            return disp
