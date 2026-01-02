import os
from PIL import Image

# Thư mục gốc chứa các file .tiff
input_folder = r"F:\omnimvs_pytorch\datasets\omnithings\depth_train_640_tiff"

# Thư mục mới để lưu các file .png
output_folder = os.path.join(input_folder, "depth_train_640")
os.makedirs(output_folder, exist_ok=True)

# Lặp qua tất cả file trong thư mục gốc
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".tiff"):
        tiff_path = os.path.join(input_folder, filename)
        png_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
        
        # Mở file .tiff và lưu thành .png
        with Image.open(tiff_path) as img:
            img.save(png_path)
        
        print(f"Converted: {filename} -> {os.path.basename(png_path)}")

print("Hoàn tất chuyển đổi tất cả file .tiff sang .png!")
