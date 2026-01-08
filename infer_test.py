import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from models import OmniMVS, SphericalSweeping
from utils import InvDepthConverter

# ----------------------------
# CONFIG
# ----------------------------
ROOT_DIR = r"F:\tmp\datasets\omnithings"
LIST_FILE = r".\dataloader\omnithings_val.txt"
CHECKPOINT = r"F:\omnimvs_pytorch\checkpoints\pretrain\checkpoint_60.pth"

FOV = 220
NDISP = 48
MIN_DEPTH = 0.55

INPUT_W, INPUT_H = 500, 480
OUT_W, OUT_H = 512, 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model():
    sweep = SphericalSweeping(
        ROOT_DIR,
        h=OUT_H,
        w=OUT_W,
        fov=FOV
    )

    model = OmniMVS(
        sweep,
        ndisp=NDISP,
        min_depth=MIN_DEPTH,
        h=OUT_H,
        w=OUT_W
    )

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    model = model.to(DEVICE)
    model.eval()
    return model


# ----------------------------
# VISUALIZATION
# ----------------------------
def visualize(batch, pred_depth, cam_list):
    plt.figure(figsize=(15, 4))

    # show input images
    for i, cam in enumerate(cam_list):
        img = batch[cam][0, 0].cpu().numpy()  # [1,1,H,W] -> [H,W]
        plt.subplot(1, len(cam_list) + 1, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(cam)
        plt.axis("off")

    # show predicted depth
    plt.subplot(1, len(cam_list) + 1, len(cam_list) + 1)
    plt.imshow(
        pred_depth[0].cpu().numpy(),
        cmap="jet",
        vmin=0.5,
        vmax=10.0
    )
    plt.title("Pred Depth (meters)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
def main():
    # same transform as training
    transform = transforms.Compose([
        Resize((INPUT_W, INPUT_H), (OUT_W, OUT_H)),
        ToTensor(),
        Normalize()
    ])

    dataset = OmniStereoDataset(
        root_dir=ROOT_DIR,
        filename_txt=LIST_FILE,
        transform=transform,
        fov=FOV
    )

    # pick one sample
    idx = 100
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=[idx]
    )

    batch = next(iter(loader))

    # move to device
    for k in batch:
        batch[k] = batch[k].to(DEVICE)

    # load model
    model = load_model()

    # ---------------- INFERENCE ----------------
    with torch.no_grad():
        pred_idx = model(batch)   # [B, H, W] disparity index

    # -------- index -> depth --------
    invd_0 = model.inv_depths[0]
    invd_max = model.inv_depths[-1]

    converter = InvDepthConverter(
        ndisp=NDISP,
        invd_0=invd_0,
        invd_max=invd_max
    )

    pred_invdepth = converter.index_to_invdepth(pred_idx)
    pred_depth = 1.0 / (pred_invdepth + 1e-8)

    # debug: check collapse
    print(
        "pred_idx stats:",
        pred_idx.min().item(),
        pred_idx.max().item(),
        pred_idx.mean().item()
    )

    # visualize
    visualize(batch, pred_depth, dataset.cam_list)


if __name__ == "__main__":
    main()
