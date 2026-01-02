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

                panoramic_x = min(panoramic_x, panoramic_width - 1)
                panoramic_y = min(panoramic_y, panoramic_height - 1)
                fisheye_image[y, x] = panoramic_image[panoramic_y, panoramic_x]

    return fisheye_image

def fisheye_to_panoramic(fisheye_image, panoramic_height, panoramic_width):
    fisheye_height, fisheye_width = fisheye_image.shape[:2]
    panoramic_image = np.zeros((panoramic_height, panoramic_width, 3), dtype=np.uint8)

    center_x = fisheye_width // 2
    center_y = fisheye_height // 2
    radius = min(center_x, center_y)

    for y in range(panoramic_height):
        for x in range(panoramic_width):
            theta = (x / (panoramic_width / 2) - 1) * np.pi
            rho = (y / panoramic_height) * radius

            fisheye_x = int(center_x + rho * np.cos(theta))
            fisheye_y = int(center_y + rho * np.sin(theta))

            if 0 <= fisheye_x < fisheye_width and 0 <= fisheye_y < fisheye_height:
                panoramic_image[y, x] = fisheye_image[fisheye_y, fisheye_x]

    return panoramic_image

# Đường dẫn ảnh test
test_image_path = r'D:\ubuntu\test_algorithm\deep_learning\hyp_dataset\hyp_outdoor_3\all_outdoor_3\IMG_0058\frame_01323.jpg'

# Thư mục lưu kết quả test
output_dir = r'F:\omnimvs_pytorch'
os.makedirs(output_dir, exist_ok=True)

panoramic_img = cv2.imread(test_image_path)
if panoramic_img is None:
    raise ValueError(f"Không đọc được ảnh {test_image_path}")

# Chuyển sang fisheye
fisheye_img = panoramic_to_fisheye(panoramic_img)
fisheye_path = os.path.join(output_dir, 'fisheye_test.png')
cv2.imwrite(fisheye_path, fisheye_img)

# Phục hồi lại panoramic
recovered_img = fisheye_to_panoramic(fisheye_img, panoramic_img.shape[0], panoramic_img.shape[1])
recovered_path = os.path.join(output_dir, 'recovered_test.png')
cv2.imwrite(recovered_path, recovered_img)

print("Test hoàn tất! Kết quả lưu tại:", output_dir)
