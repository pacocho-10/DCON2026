from PIL import Image
import os

jpg_path = "/Users/pacocho/SENDAI_AI/Lecture/U-net/arange_data/"
output_dir = "/Users/pacocho/SENDAI_AI/Lecture/U-net/dataset/images"
os.makedirs(output_dir, exist_ok=True)

# ディレクトリ内のすべてのJPGファイルを処理
for filename in os.listdir(jpg_path):
    if filename.lower().endswith(".jpg"):
        jpg_file_path = os.path.join(jpg_path, filename)
        with Image.open(jpg_file_path) as img:
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(output_dir, f"{base_name}.png")
            img.save(out_path, "PNG")
            print(f"✅ 保存しました: {out_path}")