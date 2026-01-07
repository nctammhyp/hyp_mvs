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
CHECKPOINT = r"F:\omnimvs_pytorch\checkpoints\pretrain\checkpoints_23.pth"  # đổi path

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
def visualize(batch, pred, cam_list):
    plt.figure(figsize=(15, 4))

    for i, cam in enumerate(cam_list):
        img = batch[cam][0, 0].cpu().numpy()  # <-- FIX
        plt.subplot(1, len(cam_list) + 1, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(cam)
        plt.axis("off")

    plt.subplot(1, len(cam_list) + 1, len(cam_list) + 1)
    plt.imshow(pred[0].cpu().numpy(), cmap="jet")
    plt.title("Pred InvDepth (index)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



# ----------------------------
# MAIN
# ----------------------------
def main():
    # transform giống train
    transform = transforms.Compose([
        Resize((INPUT_W, INPUT_H), (OUT_W, OUT_H)),
        ToTensor(),
        Normalize()
    ])

    # dataset + dataloader
    dataset = OmniStereoDataset(
        root_dir=ROOT_DIR,
        filename_txt=LIST_FILE,
        transform=transform,
        fov=FOV
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    model = load_model()

    # ---- LẤY IDX ----
    idx = 100
    batch = next(iter(torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=[idx]
    )))

    # move to device
    for k in batch:
        batch[k] = batch[k].to(DEVICE)

    # inference
    with torch.no_grad():
        pred = model(batch)

    visualize(batch, pred, dataset.cam_list)


if __name__ == "__main__":
    main()
