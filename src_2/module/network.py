import torch
import torch.nn.functional as F
from module.basic import *
from easydict import EasyDict as Edict

class FeatureLayers(torch.nn.Module):
    def __init__(self, CH=32, use_rgb=False):
        super().__init__()
        layers = []
        in_channel = 3 if use_rgb else 1
        layers.append(Conv2D(in_channel,CH,5,2,2))
        layers += [Conv2D(CH,CH,3,1,1) for _ in range(10)]
        for d in range(2,5):
            layers += [Conv2D(CH,CH,3,1,d,dilation=d) for _ in range(2)]
        layers.append(Conv2D(CH,CH,3,1,1,bn=False,relu=False))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, im):
        x = self.layers[0](im)
        for i in range(1,17,2):
            x_ = self.layers[i](x)
            x = self.layers[i+1](x_,residual=x)
        x = self.layers[17](x)
        return x

class SphericalSweep(torch.nn.Module):
    def __init__(self, CH=32):
        super().__init__()
        self.transfer_conv = Conv2D(CH,CH,3,2,1,bn=False,relu=False)

    def forward(self, feature, grids):
        """
        feature: [B,C,H,W]
        grids: list of num_invdepth tensors, each [H,W,2] or [B,H,W,2]
        """
        B,C,H,W = feature.shape
        D = len(grids)
        sweep_list = []

        for d in range(D):
            g = grids[d]
            if isinstance(g, np.ndarray):
                g = torch.from_numpy(g).float()
            g = g.to(feature.device)
            if g.ndim == 3:  # [H,W,2] -> add batch
                g = g.unsqueeze(0).repeat(B,1,1,1)
            elif g.ndim == 4 and g.shape[0] != B:
                g = g[:B]
            sweep_list.append(F.grid_sample(feature, g, align_corners=True))
        
        sweep = torch.stack(sweep_list, dim=1)  # [B,D,C,H,W]
        sweep = sweep.view(B*D,C,H,W)  # Conv2d needs 4D
        out = self.transfer_conv(sweep)
        out = out.view(B,D,C,H,W)
        return out

class CostCompute(torch.nn.Module):
    def __init__(self,CH=32):
        super().__init__()
        CH2 = CH*2
        self.fusion = Conv3D(2*CH,CH,3,1,1)
        convs = [Conv3D(CH,CH,3,1,1) for _ in range(3)]
        convs += [Conv3D(CH,CH,3,2,1), Conv3D(CH,CH,3,1,1), Conv3D(CH,CH,3,1,1)]
        self.convs = torch.nn.ModuleList(convs)
        self.deconv1 = DeConv3D(CH,CH,3,2,1,out_pad=1)
        self.deconv2 = DeConv3D(CH,1,3,2,1,out_pad=1,bn=False,relu=False)

    def forward(self, feats):
        c = self.fusion(feats)
        for conv in self.convs:
            c = conv(c)
        c = self.deconv1(c,residual=c)
        costs = self.deconv2(c)
        return costs

class OmniMVSNet(torch.nn.Module):
    def __init__(self,varargin=None):
        super().__init__()
        opts = Edict()
        opts.CH = 32
        opts.num_invdepth = 192
        opts.use_rgb = False
        self.opts = argparse(opts,varargin)

        self.feature_layers = FeatureLayers(self.opts.CH,self.opts.use_rgb)
        self.spherical_sweep = SphericalSweep(self.opts.CH)
        self.cost_computes = CostCompute(self.opts.CH)
        self.disps = torch.arange(0,self.opts.num_invdepth,
            requires_grad=False).view((1,-1,1,1)).float().cuda()

    def forward(self, imgs, grids, upsample=False, out_cost=False):
        feats = [self.feature_layers(x) for x in imgs]
        spherical_feats_list = [self.spherical_sweep(feats[i], grids) for i in range(len(imgs))]
        spherical_feats = torch.stack(spherical_feats_list, dim=1)  # [B,N,C,H,W]
        B,N,C,H,W = spherical_feats.shape
        spherical_feats = spherical_feats.view(B*N,C,H,W)
        costs = self.cost_computes(spherical_feats)
        costs = costs.view(B,self.opts.num_invdepth,C,H)  # adjust as needed
        prob = F.softmax(costs,1)
        disp = torch.sum(prob*self.disps,1)
        if out_cost:
            return disp, prob, costs
        return disp
