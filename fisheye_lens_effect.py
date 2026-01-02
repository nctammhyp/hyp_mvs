import cv2
import numpy as np


def circular_fisheye_cut_only(
    img,
    alpha=2.0,
    cut_radius=0.85,          # 🔥 chỉ chỉnh giá trị này
    bg_color=(255, 255, 255)
):
    # ======================
    # 1. Make square
    # ======================
    h, w = img.shape[:2]
    size = max(h, w)
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255

    y0 = (size - h) // 2
    x0 = (size - w) // 2
    canvas[y0:y0+h, x0:x0+w] = img
    img = canvas

    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    L = min(cx, cy)

    # ======================
    # 2. Coordinate grid
    # ======================
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    xn = (x - cx) / L
    yn = (y - cy) / L

    R = np.sqrt(xn**2 + yn**2)
    theta = np.arctan2(yn, xn)

    # ======================
    # 3. ORIGINAL effect (UNCHANGED)
    # ======================
    R_new = R * (1 + alpha * R**2)
    R_new = np.minimum(R_new, 1.0)

    # ======================
    # 4. Remap
    # ======================
    map_x = L * R_new * np.cos(theta) + cx
    map_y = L * R_new * np.sin(theta) + cy

    output = cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_color
    )

    # ======================
    # 5. HARD circular cut (ONLY FIX)
    # ======================
    mask = (R <= cut_radius).astype(np.uint8)
    mask = cv2.merge([mask, mask, mask])

    output = output * mask + np.array(bg_color, dtype=np.uint8) * (1 - mask)

    return output


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    img = cv2.imread(r"F:\omnimvs_pytorch\frame_00122 - Copy.jpg")
    if img is None:
        raise IOError("Cannot read input image")

    result = circular_fisheye_cut_only(
        img,
        alpha=2.0,
        cut_radius=0.6  # 👈 giảm nếu còn tia
    )

    cv2.imshow("Circular Fisheye (Cut Only)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("output_cut_only.jpg", result)
