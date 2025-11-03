import cv2, os

src_dir = '/Users/pacocho/SENDAI_AI/Lecture/U-net/dataset/images'
dst_dir = '/Users/pacocho/SENDAI_AI/Lecture/U-net/dataset/masks'
os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if fname.endswith('.png'):
        path = os.path.join(src_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ 画像を読み取れませんでした: {path}")
            continue
        mask = cv2.bitwise_not(img)
        cv2.imwrite(os.path.join(dst_dir, fname), mask)