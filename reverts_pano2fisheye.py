import cv2
import numpy as np
import os

def panoramic_to_fisheye(panoramic_image):
    panoramic_height, panoramic_width = panoramic_image.shape[:2]

    fisheye_height = min(panoramic_height, panoramic_width)
    fisheye_width = fisheye_height * 2

    fisheye_image = np.zeros((fisheye_height, fisheye_width, 3), dtype=np.uint8)

    center_x = fisheye_width // 2
    center_y = fisheye_height // 2
    radius = min(center_x, center_y)

    mask = np.zeros((fisheye_height, fisheye_width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    for y in range(fisheye_height):
        for x in range(fisheye_width):
            if mask[y, x] == 255:
                theta = np.arctan2(y - center_y, x - center_x)
                rho = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                panoramic_x = int((theta / np.pi + 1) * (panoramic_width / 2))
                panoramic_y = int((rho / radius) * panoramic_height)

                if panoramic_x >= panoramic_width:
                    panoramic_x = panoramic_width - 1
                if panoramic_y >= panoramic_height:
                    panoramic_y = panoramic_height - 1

                fisheye_image[y, x] = panoramic_image[panoramic_y, panoramic_x]

    return fisheye_image

# Thư mục nguồn và đích
source_dir = r'F:\omnimvs_pytorch\datasets\omnithings\depth_train_640'
target_dir = r'F:\omnimvs_pytorch\datasets\omnithings\depth_train_omni'

os.makedirs(target_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

# Lặp qua tất cả các file trong thư mục nguồn
for filename in os.listdir(source_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        src_path = os.path.join(source_dir, filename)
        tgt_path = os.path.join(target_dir, filename)

        panoramic_img = cv2.imread(src_path)
        if panoramic_img is None:
            print(f"Không đọc được ảnh {src_path}, bỏ qua.")
            continue

        fisheye_img = panoramic_to_fisheye(panoramic_img)
        cv2.imwrite(tgt_path, fisheye_img)
        print(f"Đã lưu fisheye: {tgt_path}")

print("Hoàn tất chuyển đổi tất cả ảnh!")
