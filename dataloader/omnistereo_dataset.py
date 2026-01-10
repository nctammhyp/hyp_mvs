# from os.path import join

# import cv2
# import numpy as np
# from ocamcamera import OcamCamera
# from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
# import numpy as np
# from torch.utils.data import DataLoader


# class OmniStereoDataset(Dataset):
#     """Omnidirectional Stereo Dataset.
#     http://cvlab.hanyang.ac.kr/project/omnistereo/
#     """

#     def __init__(self, root_dir, filename_txt, transform=None, fov=220):
#         self.root_dir = root_dir
#         self.transform = transform

#         # load filenames
#         with open(filename_txt) as f:
#             data = f.read()
#         self.filenames = data.strip().split('\n')

#         # folder name
#         # self.cam_list = ['cam1', 'cam2', 'cam3']
#         self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']

#         self.depth_folder = 'depth_train_640'
#         # self.depth_folder = 'depth_train_omni'
#         print(f"Using depth folder: {self.depth_folder}")

#         # load ocam calibration data and generate valid image
#         self.ocams = []
#         self.valids = []
#         for cam in self.cam_list:
#             ocam_file = join(root_dir, f'o{cam}.txt')
#             self.ocams.append(OcamCamera(ocam_file, fov, show_flag=False))
#             self.valids.append(self.ocams[-1].valid_area())

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
#         sample = {}

#         filename = self.filenames[idx]
#         # load images
#         for i, cam in enumerate(self.cam_list):
#             img_path = join(self.root_dir, cam, filename)
#             sample[cam] = load_image(img_path, valid=self.valids[i])
#         # load inverse depth
#         depth_path = join(self.root_dir, self.depth_folder, filename)
#         sample['idepth'] = load_invdepth(depth_path)

#         if self.transform:
#             sample = self.transform(sample)

#         return sample


# def load_invdepth(filename, min_depth=55):
#     '''
#     min_depth in [cm]
#     '''
#     # print(f"Loading inverse depth from: {filename}")
#     invd_value = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
#     invdepth = (invd_value / 100.0) / (min_depth * 655) + np.finfo(np.float32).eps
#     invdepth *= 100  # unit conversion from cm to m
#     return invdepth


# def load_image(filename, gray=True, valid=None):
#     img = cv2.imread(filename)
#     if gray:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     if not valid is None:
#         img[valid == 0] = 0
#     return img


# if __name__ == "__main__":
#     # folder và file list
#     root_dir = r"F:\tmp\datasets\omnithings"  # đổi path theo dataset
#     train_list = r".\dataloader\omnithings_train.txt"

#     # tạo dataset
#     dataset = OmniStereoDataset(root_dir=root_dir, filename_txt=train_list, fov=220)
#     print(f"Dataset size: {len(dataset)}")

#     # tạo DataLoader
#     loader = DataLoader(dataset, batch_size=1, shuffle=True)

#     # lấy 1 sample
#     sample = next(iter(loader))
#     cam_list = dataset.cam_list

#     # show images
#     plt.figure(figsize=(12, 4))
#     for i, cam in enumerate(cam_list):
#         img = sample[cam][0].numpy()  # shape (H, W)
#         plt.subplot(1, len(cam_list)+1, i+1)
#         plt.imshow(img, cmap='gray')
#         plt.title(cam)
#         plt.axis('off')

#     # show inverse depth (gt)
#     gt_idepth = sample['idepth'][0].numpy()
#     plt.subplot(1, len(cam_list)+1, len(cam_list)+1)
#     plt.imshow(gt_idepth, cmap='jet')
#     plt.title('GT Inverse Depth')
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()


import cv2
import numpy as np
from os.path import join
from torch.utils.data import Dataset
from ocamcamera import OcamCamera

class OmniStereoDataset(Dataset):
    def __init__(self, root_dir, filename_txt, transform=None, fov=220):
        self.root_dir = root_dir
        self.transform = transform
        with open(filename_txt) as f:
            self.filenames = f.read().strip().split('\n')

        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.depth_folder = 'depth_train_640'
        
        # Load Ocam
        self.ocams = []
        self.valids = []
        for cam in self.cam_list:
            ocam_file = join(root_dir, f'o{cam}.txt')
            self.ocams.append(OcamCamera(ocam_file, fov, show_flag=False))
            self.valids.append(self.ocams[-1].valid_area())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample = {}
        filename = self.filenames[idx]
        
        # 1. Load Images (Chuyển sang RGB để Feature Extractor học tốt hơn)
        for i, cam in enumerate(self.cam_list):
            img_path = join(self.root_dir, cam, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Sửa từ Gray sang RGB
            
            # Apply valid mask
            if self.valids[i] is not None:
                img[self.valids[i] == 0] = 0
            sample[cam] = img

        # 2. Load Depth
        depth_path = join(self.root_dir, self.depth_folder, filename)
        sample['idepth'] = self.load_invdepth(depth_path)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_invdepth(self, filename, min_depth=0.55):
        # min_depth đơn vị mét (0.55m = 55cm)
        invd_value = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        if invd_value is None:
            return np.zeros((256, 512), dtype=np.float32)
        
        invd_value = invd_value.astype(np.float32)
        # Công thức: Inverse Depth = pixel_val / (65535 * min_depth)
        # 655.35 là hệ số scale thông dụng cho ảnh 16-bit
        invdepth = invd_value / (655.35 * min_depth) + 1e-6
        return invdepth