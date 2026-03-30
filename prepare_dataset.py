import os
import cv2
import numpy as np
import shutil

src_dir = 'd:/555/images'
images_dir = 'd:/555/dataset/images/train'
masks_dir = 'd:/555/dataset/masks/train'

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

for file in os.listdir(src_dir):
    if file.endswith('_input.jpg'):
        idx = file.split('_')[0]
        input_path = os.path.join(src_dir, file)
        output_path = os.path.join(src_dir, f"{idx}_output.jpg")
        
        if not os.path.exists(output_path):
            continue
            
        img_in = cv2.imread(input_path)
        img_out = cv2.imread(output_path)
        
        diff = cv2.absdiff(img_in, img_out)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, binary_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
        
        # Save image and mask
        shutil.copy(input_path, os.path.join(images_dir, f"{idx}.jpg"))
        cv2.imwrite(os.path.join(masks_dir, f"{idx}.png"), binary_mask)
        print(f"Processed {idx}")

print("Dataset prepared!")
