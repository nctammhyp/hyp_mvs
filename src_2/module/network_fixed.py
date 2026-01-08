# network_fixed.py
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
        for d in range(2, 5):  # conv[12-17]
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
        """
        feature: [B, C, H, W]
        grids: list of D tensors, mỗi tensor [H, W, 2] hoặc [B, H, W, 2]
        """
        B, C, H, W = feature.shape
        D = len(grids)
        sweep_list = []

        for d in range(D):
            g = grids[d]
            if not isinstance(g, torch.Tensor):
                g = torch.from_numpy(g).float()
            g = g.to(feature.device)
            if g.ndim == 3:  # [H,W,2] -> [B,H,W,2]
                g = g.unsqueeze(0).repeat(B, 1, 1, 1)
            sweep_list.append(F.grid_sample(feature, g, align_corners=True))  # [B,C,H,W]

        # stack theo depth -> [B,D,C,H,W]
        sweep = torch.stack(sweep_list, dim=1)

        # reshape để Conv2d nhận 4D input
        B, D, C, H, W = sweep.shape
        sweep_reshape = sweep.reshape(B * D, C, H, W)  # [B*D, C, H, W]

        out = self.transfer_conv(sweep_reshape)  # [B*D, C, H_out, W_out]

        # reshape trở lại [B, D, C_out, H_out, W_out]
        _, C_out, H_out, W_out = out.shape
        out = out.view(B, D, C_out, H_out, W_out)
        return out

# -----------------------------
# Cost volume computation
# -----------------------------
class CostCompute(nn.Module):
    def __init__(self, CH=32, num_views=2):
        super().__init__()
        self.CH = CH
        self.num_views = num_views
        CH_total = CH * num_views
        CH2 = CH

        # input channels = CH_total (sau khi concat views)
        self.fusion = Conv3D(CH_total, CH, 3, 1, 1)

        convs = []
        convs += [Conv3D(CH, CH, 3, 1, 1),
                  Conv3D(CH, CH, 3, 1, 1),
                  Conv3D(CH, CH, 3, 1, 1)]
        convs += [Conv3D(CH, CH2, 3, 2, 1),
                  Conv3D(CH2, CH2, 3, 1, 1),
                  Conv3D(CH2, CH2, 3, 1, 1)]
        convs += [Conv3D(CH2, CH2, 3, 2, 1),
                  Conv3D(CH2, CH2, 3, 1, 1),
                  Conv3D(CH2, CH2, 3, 1, 1)]
        convs += [Conv3D(CH2, CH2, 3, 2, 1),
                  Conv3D(CH2, CH2, 3, 1, 1),
                  Conv3D(CH2, CH2, 3, 1, 1)]
        convs += [Conv3D(CH2, CH2*2, 3, 2, 1),
                  Conv3D(CH2*2, CH2*2, 3, 1, 1),
                  Conv3D(CH2*2, CH2*2, 3, 1, 1)]
        self.convs = nn.ModuleList(convs)

        self.deconv1 = DeConv3D(CH2*2, CH2, 3, 2, 1, out_pad=1)
        self.deconv2 = DeConv3D(CH2, CH2, 3, 2, 1, out_pad=1)
        self.deconv3 = DeConv3D(CH2, CH2, 3, 2, 1, out_pad=1)
        self.deconv4 = DeConv3D(CH2, CH, 3, 2, 1, out_pad=1)
        self.deconv5 = DeConv3D(CH, 1, 3, 2, 1, out_pad=1, bn=False, relu=False)

    def forward(self, feats):
        c = self.fusion(feats)
        for i in range(0, len(self.convs), 3):
            c = self.convs[i](c)
            c = self.convs[i+1](c)
            c = self.convs[i+2](c)
        c = self.deconv1(c)
        c = self.deconv2(c)
        c = self.deconv3(c)
        c = self.deconv4(c)
        costs = self.deconv5(c)
        return costs

# -----------------------------
# OmniMVSNet
# -----------------------------
class OmniMVSNet(nn.Module):
    def __init__(self, varargin=None, num_views=2):
        super().__init__()

        # tạo opts mặc định
        if varargin is None:
            self.opts = Edict()
        elif isinstance(varargin, dict):
            self.opts = Edict(varargin)
        elif isinstance(varargin, Edict):
            self.opts = varargin
        else:
            raise TypeError("varargin must be dict or EasyDict or None")

        # set default values
        self.opts.setdefault('CH', 32)
        self.opts.setdefault('num_invdepth', 192)
        self.opts.setdefault('use_rgb', False)
        self.num_views = num_views

        self.feature_layers = FeatureLayers(self.opts.CH, self.opts.use_rgb)
        self.spherical_sweep = SphericalSweep(self.opts.CH)
        self.cost_computes = CostCompute(self.opts.CH, num_views=self.num_views)

        self.register_buffer(
            "disps",
            torch.arange(0, self.opts.num_invdepth).view(1, -1, 1, 1).float()
        )

    def forward(self, imgs, grids, upsample=False, out_cost=False):
        """
        imgs: list of [B,C,H,W] với length = num_views
        grids: list of D tensors
        """
        feats = [self.feature_layers(x) for x in imgs]

        # sweep từng view
        spherical_feats_list = [self.spherical_sweep(feats[i], grids) for i in range(len(imgs))]

        # concat các view theo channel -> [B,D,C*num_views,H,W]
        spherical_feats = torch.cat(spherical_feats_list, dim=2)

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
