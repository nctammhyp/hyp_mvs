import cv2
import numpy as np
import os

def panoramic_to_fisheye(pano, pano_fov_deg=70, fisheye_fov_deg=220):
    pano_h, pano_w = pano.shape[:2]

    # fisheye output size
    fisheye_size = min(pano_h, pano_w)
    fisheye = np.zeros((fisheye_size, fisheye_size, 3), dtype=np.uint8)

    cx = cy = fisheye_size // 2
    radius = fisheye_size // 2

    pano_half_fov = np.deg2rad(pano_fov_deg / 2)
    fisheye_half_fov = np.deg2rad(fisheye_fov_deg / 2)

    for y in range(fisheye_size):
        for x in range(fisheye_size):
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx * dx + dy * dy)

            if r > radius:
                continue

            # góc trong fisheye (±110°)
            theta = np.arctan2(dy, dx)

            # chỉ map vùng pano 70°
            if theta < -pano_half_fov or theta > pano_half_fov:
                continue

            # map theta → pano X
            pano_x = int(((theta + pano_half_fov) / (2 * pano_half_fov)) * pano_w)

            # map radius → pano Y
            pano_y = int((r / radius) * pano_h)

            pano_x = np.clip(pano_x, 0, pano_w - 1)
            pano_y = np.clip(pano_y, 0, pano_h - 1)

            fisheye[y, x] = pano[pano_y, pano_x]

    return fisheye


def main():
    input_path = r"F:\omnimvs_pytorch\frame_00122.jpg"   # pano 70°
    output_path = "fisheye_220.jpg"

    if not os.path.exists(input_path):
        print("❌ Không tìm thấy panorama.jpg")
        return

    pano = cv2.imread(input_path)
    if pano is None:
        print("❌ Không đọc được ảnh")
        return

    fisheye = panoramic_to_fisheye(
        pano,
        pano_fov_deg=360,
        fisheye_fov_deg=70
    )

    cv2.imwrite(output_path, fisheye)
    print("✅ Xuất fisheye 220°:", output_path)


if __name__ == "__main__":
    main()
