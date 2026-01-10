import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1. Feature Extractor (CNN Encoder)
# -----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x  # [B, C, H, W]

# -----------------------------
# 2. Warp features to reference view (simplified)
# -----------------------------
def warp_features(features, depth_hypo, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics):
    """
    Simplified warping placeholder.
    In real MVSNet, this uses differentiable homography.
    Here, we'll just stack features for demonstration.
    """
    # Normally, we would project src pixels to ref view at each depth
    # For simplicity: return features as-is
    return features

# -----------------------------
# 3. Build cost volume
# -----------------------------
def build_cost_volume(ref_feat, src_feats, depth_hypo, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics):
    B, C, H, W = ref_feat.shape
    D = len(depth_hypo)
    cost_volume = ref_feat.unsqueeze(2).repeat(1,1,D,1,1)  # [B, C, D, H, W]
    
    for src_feat in src_feats:
        warped = warp_features(src_feat, depth_hypo, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics)
        cost_volume += warped.unsqueeze(2)  # simple aggregation by sum
    cost_volume /= (len(src_feats)+1)
    return cost_volume  # [B, C, D, H, W]

# -----------------------------
# 4. 3D CNN to regularize cost volume
# -----------------------------
class CostRegularization(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.conv3d_2 = nn.Conv3d(in_channels, 1, 3, padding=1)  # output single cost per voxel
    
    def forward(self, cost_volume):
        x = F.relu(self.conv3d_1(cost_volume))
        x = self.conv3d_2(x)  # [B, 1, D, H, W]
        return x.squeeze(1)  # [B, D, H, W]

# -----------------------------
# 5. Depth regression using soft-argmin
# -----------------------------
def depth_regression(prob_volume, depth_hypo):
    """
    prob_volume: [B, D, H, W]
    depth_hypo: [D]
    """
    depth_hypo = depth_hypo.view(1, -1, 1, 1).to(prob_volume.device)
    depth_map = torch.sum(prob_volume * depth_hypo, dim=1)
    return depth_map  # [B, H, W]

# -----------------------------
# 6. Full pipeline example
# -----------------------------
class SimpleMVSNet(nn.Module):
    def __init__(self, feature_channels=32):
        super().__init__()
        self.feature_extractor = FeatureExtractor(out_channels=feature_channels)
        self.cost_regularization = CostRegularization(in_channels=feature_channels)
    
    def forward(self, imgs, depth_hypo, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics):
        # imgs: list of 3 images, first one is reference
        ref_img = imgs[0]
        src_imgs = imgs[1:]
        
        # 1. Feature extraction
        ref_feat = self.feature_extractor(ref_img)
        src_feats = [self.feature_extractor(im) for im in src_imgs]
        
        # 2. Build cost volume
        cost_volume = build_cost_volume(ref_feat, src_feats, depth_hypo, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics)
        
        # 3. 3D CNN
        cost_volume = self.cost_regularization(cost_volume)
        
        # 4. Convert to probability
        prob_volume = F.softmax(-cost_volume, dim=1)  # softmax over depth
        
        # 5. Depth regression
        depth_map = depth_regression(prob_volume, depth_hypo)
        return depth_map

# -----------------------------
# Example usage
# -----------------------------
B, C, H, W = 1, 3, 128, 128
D = 32  # number of depth hypotheses
depth_hypo = torch.linspace(1.0, 10.0, D)  # hypothetical depth range

# fake inputs
imgs = [torch.rand(B,C,H,W) for _ in range(3)]
ref_intrinsics = torch.eye(3).unsqueeze(0)
src_intrinsics = torch.eye(3).unsqueeze(0)
ref_extrinsics = torch.eye(4).unsqueeze(0)
src_extrinsics = torch.eye(4).unsqueeze(0)

model = SimpleMVSNet()
depth_map = model(imgs, depth_hypo, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics)
print(depth_map.shape)  # [B, H, W]
