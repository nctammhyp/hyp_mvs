# network.py
# Architecture submitted to PAMI
# Author: Changhee Won (changhee.1.won@gmail.com)
#
import torch
import torch.nn.functional as F
from torch import nn
from module.basic import *

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

class SphericalSweep(nn.Module):
    def __init__(self, CH=32):
        super().__init__()
        self.transfer_conv = Conv2D(CH, CH, 3, 2, 1, bn=False, relu=False)  # stride=2 halves H/W

    def forward(self, feature, grids):
        """
        feature: [B, C, H, W]
        grids: list of num_invdepth tensors, mỗi tensor [H, W, 2] hoặc [B, H, W, 2]
        """
        B, C, H, W = feature.shape
        D = len(grids)
        sweep_list = []

        for d in range(D):
            g = grids[d]
            if isinstance(g, np.ndarray):
                g = torch.from_numpy(g).float()
            g = g.to(feature.device)
            if g.ndim == 3:  # [H,W,2]
                g = g.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]
            sweep_list.append(F.grid_sample(feature, g, align_corners=True))

        # stack theo depth -> [B, D, C, H, W]
        sweep = torch.stack(sweep_list, dim=1)

        # merge batch và depth để Conv2d nhận input 4D
        sweep_reshape = sweep.view(B * D, C, H, W)  # [B*D, C, H, W]

        # apply Conv2D
        out = self.transfer_conv(sweep_reshape)  # [B*D, C, H_out, W_out]

        # reshape về [B, D, C, H_out, W_out]
        _, C_out, H_out, W_out = out.shape
        out = out.view(B, D, C_out, H_out, W_out)
        return out

class CostCompute(nn.Module):
    def __init__(self, CH=32):
        super().__init__()
        CH2 = 2 * CH
        self.fusion = Conv3D(CH2, CH, 3, 1, 1)
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
        for i in range(3):
            c = self.convs[i](c)
        c1 = c.clone()
        for i in range(3, 6):
            c = self.convs[i](c)
        c2 = c.clone()
        for i in range(6, 9):
            c = self.convs[i](c)
        c3 = c.clone()
        for i in range(9, 12):
            c = self.convs[i](c)
        c4 = c.clone()
        for i in range(12, 15):
            c = self.convs[i](c)
        c5 = c.clone()
        c = self.deconv1(c5, residual=c4)
        c = self.deconv2(c, residual=c3)
        c = self.deconv3(c, residual=c2)
        c = self.deconv4(c, residual=c1)
        costs = self.deconv5(c)
        return costs

class OmniMVSNet(nn.Module):
    def __init__(self, varargin=None):
        super().__init__()
        opts = Edict()
        opts.CH = 32
        opts.num_invdepth = 192
        opts.use_rgb = False
        self.opts = argparse(opts, varargin)

        self.feature_layers = FeatureLayers(self.opts.CH, self.opts.use_rgb)
        self.spherical_sweep = SphericalSweep(self.opts.CH)
        self.cost_computes = CostCompute(self.opts.CH)

        self.disps = torch.arange(0, self.opts.num_invdepth,
                                  requires_grad=False).view(1, -1, 1, 1).float().cuda()

    def forward(self, imgs, grids, upsample=False, out_cost=False):
        """
        imgs: list of BxCxHxW images
        grids: list of num_invdepth grids [H,W,2]
        """
        feats = [self.feature_layers(x) for x in imgs]  # list of [B,C,H,W]

        # Spherical sweep cho mỗi view
        spherical_feats_list = [self.spherical_sweep(feats[i], grids) for i in range(len(imgs))]

        # concat theo view -> [B, D, C, H_out, W_out]
        spherical_feats = torch.cat(spherical_feats_list, dim=1)

        # đưa vào CostCompute
        costs = self.cost_computes(spherical_feats)

        if upsample:
            costs = F.interpolate(costs.squeeze(1), scale_factor=2,
                                  mode='bilinear', align_corners=True)
        else:
            costs = costs.squeeze(1)

        prob = F.softmax(costs, 1)
        disp = torch.mul(prob, self.disps)
        disp = torch.sum(disp, 1)

        if out_cost:
            return disp, prob.squeeze(), costs.squeeze()
        else:
            return disp
