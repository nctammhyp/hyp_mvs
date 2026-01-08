import torch
import torch.nn as nn
import torch.nn.functional as F
from module.basic import Conv2D, Conv3D, DeConv3D
from easydict import EasyDict as Edict

# -----------------------------
# Feature extractor
# -----------------------------
class FeatureLayers(nn.Module):
    def __init__(self, CH=8, use_rgb=False):
        super().__init__()
        layers = []
        in_channel = 3 if use_rgb else 1
        layers.append(Conv2D(in_channel, CH, 5, 2, 2))  # conv[1]
        layers += [Conv2D(CH, CH, 3, 1, 1) for _ in range(5)]  # conv[2-6] giảm từ 10 → 5
        for d in range(2, 4):  # giảm từ 2-5
            layers += [Conv2D(CH, CH, 3, 1, d, dilation=d) for _ in range(2)]
        layers.append(Conv2D(CH, CH, 3, 1, 1, bn=False, relu=False))  # conv cuối
        self.layers = nn.ModuleList(layers)

    def forward(self, im):
        x = self.layers[0](im)
        for i in range(1, len(self.layers) - 1, 2):
            x_ = self.layers[i](x)
            x = self.layers[i + 1](x_, residual=x)
        x = self.layers[-1](x)
        return x

# -----------------------------
# Spherical Sweep
# -----------------------------
class SphericalSweep(nn.Module):
    def __init__(self, CH=8):
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
        B, D, C, H, W = sweep.shape
        sweep_reshape = sweep.view(B * D, C, H, W)
        out = self.transfer_conv(sweep_reshape)
        _, C_out, H_out, W_out = out.shape
        out = out.view(B, D, C_out, H_out, W_out)
        out = out.permute(0, 2, 1, 3, 4)  # [B, C, D, H, W]
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
        convs += [Conv3D(CH, CH, 3, 1, 1) for _ in range(2)]  # giảm layers
        convs += [Conv3D(CH, CH2, 3, 2, 1),
                  Conv3D(CH2, CH2, 3, 1, 1)]
        self.convs = nn.ModuleList(convs)
        self.deconv1 = DeConv3D(CH2, CH, 3, 2, 1, out_pad=1)
        self.deconv2 = DeConv3D(CH, 1, 3, 2, 1, out_pad=1, bn=False, relu=False)

    def forward(self, feats):
        c = self.fusion(feats)
        for conv in self.convs:
            c = conv(c)
        c = self.deconv1(c)
        costs = self.deconv2(c)
        return costs

# -----------------------------
# OmniMVSNet
# -----------------------------
class OmniMVSNet(nn.Module):
    def __init__(self, varargin=None, num_views=4):
        super().__init__()
        self.num_views = num_views
        # convert dict -> EasyDict
        if isinstance(varargin, dict):
            self.opts = Edict(varargin)
        else:
            self.opts = Edict() if varargin is None else varargin

        # set defaults
        ch = getattr(self.opts, 'CH', 8)
        self.opts.CH = ch
        self.opts.num_invdepth = getattr(self.opts, 'num_invdepth', 16)
        self.opts.use_rgb = getattr(self.opts, 'use_rgb', False)

        # build network
        self.feature_layers = FeatureLayers(self.opts.CH, self.opts.use_rgb)
        self.spherical_sweep = SphericalSweep(self.opts.CH)
        self.cost_computes = CostCompute(ch * self.num_views)  # fix channel mismatch

        self.register_buffer(
            "disps",
            torch.arange(0, self.opts.num_invdepth).view(1, -1, 1, 1).float()
        )

    def forward(self, imgs, grids, upsample=False, out_cost=False):
        feats = [self.feature_layers(x) for x in imgs]
        spherical_feats_list = [self.spherical_sweep(feats[i], grids) for i in range(len(imgs))]
        # concat theo channel → CH * num_views
        spherical_feats = torch.cat(spherical_feats_list, dim=1)
        costs = self.cost_computes(spherical_feats)

        if upsample:
            costs = F.interpolate(costs.squeeze(1), scale_factor=2, mode='bilinear', align_corners=True)
        else:
            costs = costs.squeeze(1)

        prob = F.softmax(costs, dim=1)
        # resize disps nếu cần match với CH
        if prob.shape[1] != self.disps.shape[1]:
            disps = F.interpolate(self.disps, size=prob.shape[1], mode='nearest')
        else:
            disps = self.disps

        disp = torch.sum(prob * disps, dim=1)

        if out_cost:
            return disp, prob, costs
        else:
            return disp
