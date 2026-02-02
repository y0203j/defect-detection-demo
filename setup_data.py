import os
import shutil
from PIL import Image


# --- configuration ---
raw_data_folder = "data/images"
train_img_dir = "data/train_images"
train_mask_dir = "data/train_masks"

if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)
if not os.path.exists(train_mask_dir):
    os.makedirs(train_mask_dir)

print("scanning for 'b' (after laser) images...")

#filter and process
files = [f for f in os.listdir(raw_data_folder) if f.endswith('b.PNG')]
print(f"found {len(files)} relevant images. processing...")


for filename in files:
    src_path = os.path.join(raw_data_folder, filename)
    dst_path_img = os.path.join(train_img_dir, filename)
    shutil.copy(src_path, dst_path_img)


#generate the synthetic mask
with Image.open(src_path) as img:
    gray_img = img.convert('L')
    mask = gray_img.point(lambda p: 255 if p > 150 else 0)
    dst_path_mask = os.path.join(train_mask_dir, filename)
    mask.save(dst_path_mask)

print("\nSUCCESS!")
print(f"moved {len(files)} images to '{train_img_dir}'")
print(f"generated {len(files)} masks in '{train_mask_dir}'")