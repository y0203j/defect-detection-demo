import os
import shutil
from PIL import Image
import cv2
import numpy as np

#mask generation using OpenCV
# These parameters should be tuned based on the findings from the EDA notebook
# A lower threshold is more sensitive to dark pixels.
INTENSITY_THRESH = 40 
# The smallest area (in pixels) to be considered a defect.
MIN_AREA = 25       

def generate_opencv_mask(img_a, img_b, img_c, intensity_thresh, min_area):
    """
    Generates a synthetic defect mask using a multi-angle, OpenCV-based pipeline.
    
    Args:
        img_a, img_b, img_c: 16-bit PIL Image objects for the three angles.
        intensity_thresh: Pixel intensity below which is considered a potential defect (0-255).
        min_area: Minimum pixel area to be considered a valid defect.

    Returns:
        A PIL Image object for the final binary mask.
    """
    imgs_16bit = [np.array(img) for img in [img_a, img_b, img_c]]

    #normalize from 16-bit (0-65535) to 8-bit (0-255)
    imgs_8bit = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for img in imgs_16bit]

    #stack into a single (H, W, 3) numpy array
    stack = np.stack(imgs_8bit, axis=-1)

    #a pixel is a defect candidate if it's dark in ALL three channels (angles)
    #defect pixel is dark in at least 2 of 3 angles
    dark_votes = (stack < intensity_thresh).sum(axis=-1)
    dark_mask = dark_votes >= 2

    #convert boolean mask to a uint8 binary image (0 or 255)
    mask = (dark_mask * 255).astype(np.uint8)

    #morphological cleanup to remove salt-and-pepper noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    #remove small, noisy regions using connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned_mask = np.zeros_like(mask)

    #start from 1 to ignore the background label (0)
    for i in range(1, num_labels):
        #if the component's area is greater than the minimum, keep it
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255
            
    return Image.fromarray(cleaned_mask, mode='L')
# ---------------------------------------------

raw_data_folder = "data/images"
train_img_dir = "data/train_images"
train_mask_dir = "data/train_masks"

if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)
if not os.path.exists(train_mask_dir):
    os.makedirs(train_mask_dir)

print("Scanning for complete image sets (Angles A, B, C)...")

#use 'b' (Post-Melt) as the anchor file to find the others
files = [f for f in os.listdir(raw_data_folder) if f.endswith('b.PNG')]
print(f"Found {len(files)} potential sets. Verifying completeness...")

count = 0

for filename_b in files:
    #construct expected filenames for Angle A and Angle C
    base_name = filename_b.replace('b.PNG', '') 
    filename_a = base_name + 'a.PNG'
    filename_c = base_name + 'c.PNG'

    src_a = os.path.join(raw_data_folder, filename_a)
    src_b = os.path.join(raw_data_folder, filename_b)
    src_c = os.path.join(raw_data_folder, filename_c)

    #only process if ALL three angles exist
    if os.path.exists(src_a) and os.path.exists(src_c):
        shutil.copy(src_a, os.path.join(train_img_dir, filename_a))
        shutil.copy(src_b, os.path.join(train_img_dir, filename_b))
        shutil.copy(src_c, os.path.join(train_img_dir, filename_c))
        
        #generate the synthetic mask using the advanced OpenCV pipeline
        with Image.open(src_a) as img_a, Image.open(src_b) as img_b, Image.open(src_c) as img_c:
            mask = generate_opencv_mask(img_a, img_b, img_c, INTENSITY_THRESH, MIN_AREA)
            
            #save the final mask, using the 'b' filename for consistency.
            dst_path_mask = os.path.join(train_mask_dir, filename_b)
            mask.save(dst_path_mask)
        
        count += 1

print("\nSUCCESS!")
print(f"Processed {count} complete layer sets.")
print(f"Total images moved: {count * 3} (Angles A, B, C)")
print(f"Generated {count} masks in '{train_mask_dir}'")