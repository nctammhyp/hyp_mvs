import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from torchvision import transforms

from dataloader import load_image
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from models import OmniMVS, SphericalSweeping
from utils import InvDepthConverter

# ----------------------------
# CONFIG
# ----------------------------
ROOT_DIR = r"F:\tmp\datasets\omnithings"
IMG_NAME = "00015.png"
CAM_LIST = ["cam1", "cam2", "cam3", "cam4"]

CHECKPOINT = r"F:\omnimvs_pytorch\checkpoints\pretrain\checkpoints_16.pth"  # đổi path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FOV = 220
NDISP = 48
MIN_DEPTH = 0.55

INPUT_W, INPUT_H = 500, 480
OUT_W, OUT_H = 512, 256


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
# READ IMAGES MANUALLY
# ----------------------------
def read_multicam_images():
    sample = {}

    for cam in CAM_LIST:
        img_path = join(ROOT_DIR, cam, IMG_NAME)
        img = load_image(img_path, gray=True)  # giống dataset
        sample[cam] = img  # numpy (H, W)

    return sample


# ----------------------------
# TRANSFORM (GIỐNG TRAIN)
# ----------------------------
def apply_transform(sample):
    transform = transforms.Compose([
        Resize((INPUT_W, INPUT_H), (OUT_W, OUT_H)),
        ToTensor(),
        Normalize()
    ])
    return transform(sample)


# ----------------------------
# INFERENCE
# ----------------------------
@torch.no_grad()
def inference(model, sample):
    batch = {}

    for k, v in sample.items():
        batch[k] = v.unsqueeze(0).to(DEVICE)  # (1,1,H,W)

    pred = model(batch)  # (1,H,W)
    return pred


# ----------------------------
# VISUALIZE
# ----------------------------
def visualize(sample, pred):
    plt.figure(figsize=(15, 4))

    for i, cam in enumerate(CAM_LIST):
        img = sample[cam][0].cpu().numpy()
        plt.subplot(1, 5, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(cam)
        plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(pred[0].cpu().numpy(), cmap="jet")
    plt.title("Pred InvDepth (index)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
def main():
    model = load_model()

    # read raw images
    sample = read_multicam_images()

    # apply SAME transform as training
    sample = apply_transform(sample)

    # inference
    pred = inference(model, sample)

    visualize(sample, pred)


if __name__ == "__main__":
    main()
