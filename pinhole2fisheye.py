import cv2
import numpy as np

# Đọc ảnh fisheye
img = cv2.imread(r'D:\ubuntu\test_algorithm\deep_learning\hyp_dataset\hyp_outdoor_3\all_outdoor_3\IMG_0058\frame_01098.jpg')
h, w = img.shape[:2]

# ===== THAM SỐ CHÍNH - ĐIỀU CHỈNH ĐỂ GIỐNG ẢNH MỤC TIÊU =====
STRENGTH = 1.2        # Độ mạnh correction (0.8-1.5), càng cao càng thẳng
K1 = -0.15            # Hệ số distortion chính (-0.3 đến 0)
K2 = 0.05             # Hệ số distortion phụ (0 đến 0.1)
ZOOM = 0.75           # Zoom level (0.6-0.9), giảm để thấy rộng hơn
# =============================================================

# Tìm tâm và bán kính
cx, cy = w // 2, h // 2
radius = min(h, w) // 2

# Kích thước output
out_size = int(radius * 2)
out_cx, out_cy = out_size // 2, out_size // 2

# Tạo lưới tọa độ
X, Y = np.meshgrid(np.arange(out_size), np.arange(out_size))
dx = (X - out_cx).astype(np.float32)
dy = (Y - out_cy).astype(np.float32)
r_out = np.sqrt(dx**2 + dy**2)
theta = np.arctan2(dy, dx)

# Mask hình tròn với viền mượt
mask = r_out <= out_cx
edge_fade = 20
edge_region = (r_out > out_cx - edge_fade) & mask
fade_alpha = (out_cx - r_out[edge_region]) / edge_fade

# Chuẩn hóa bán kính
r_norm = r_out / out_cx
r_norm = np.clip(r_norm, 0, 1)

# ==== MÔ HÌNH DISTORTION CẢI TIẾN ====
# Sử dụng Brown-Conrady distortion model
# r_distorted = r * (1 + k1*r^2 + k2*r^4)
r_distorted = r_norm * (1 + K1 * r_norm**2 + K2 * r_norm**4)

# Áp dụng strength và zoom
r_fish = r_distorted * radius * STRENGTH * ZOOM

# Clipping để tránh vượt biên
r_fish = np.clip(r_fish, 0, radius * 0.98)

# Map về tọa độ ảnh gốc
map_x = cx + r_fish * np.cos(theta)
map_y = cy + r_fish * np.sin(theta)

# Remap với LANCZOS4 cho chất lượng cao nhất
result = cv2.remap(
    img, 
    map_x.astype(np.float32), 
    map_y.astype(np.float32),
    interpolation=cv2.INTER_LANCZOS4,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)

# Áp dụng alpha mask mượt
alpha = np.ones((out_size, out_size), dtype=np.float32)
alpha[edge_region] = fade_alpha
alpha[~mask] = 0

# Blend với alpha channel
for i in range(3):
    result[:, :, i] = (result[:, :, i] * alpha).astype(np.uint8)

# ==== HẬU KỲ XỬ LÝ ====
# 1. Làm sắc nét nhẹ
sharpen_kernel = np.array([
    [0, -0.3, 0],
    [-0.3, 2.2, -0.3],
    [0, -0.3, 0]
])
result = cv2.filter2D(result, -1, sharpen_kernel)

# 2. Tăng contrast nhẹ để rõ nét hơn
result = cv2.convertScaleAbs(result, alpha=1.05, beta=5)

# Lưu kết quả
cv2.imwrite('fisheye_dewarped.png', result)
print("=" * 50)
print("✓ Đã dewarp xong!")
print(f"  STRENGTH = {STRENGTH}  (tăng để thẳng hơn)")
print(f"  K1 = {K1}  (càng âm càng correction mạnh)")
print(f"  ZOOM = {ZOOM}  (giảm để thấy rộng hơn)")
print("=" * 50)

# Hiển thị để kiểm tra (optional)
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()