import cv2
import glob
import os
import numpy as np
from ultralytics import YOLO

# Load model
model_path = r"d:\555\runs\crack_seg3\weights\best.pt"
model = YOLO(model_path)

input_imgs = glob.glob(r"d:\555\dataset\images\train\*.jpg")
os.makedirs(r"d:\555\output", exist_ok=True)

# Using exact color discovered analyzing 1_output.jpg
crack_color = np.array([48, 44, 150], dtype=np.uint8) # BGR
print(f"Applying exact output color {crack_color} to cracks")

for img_p in input_imgs:
    fname = os.path.basename(img_p)
    img = cv2.imread(img_p)
    if img is None:
        continue
    
    # Check original image for exact dimension
    H, W = img.shape[:2]
    
    # Predict
    results = model.predict(img, imgsz=1024, device=[1], verbose=False)
    
    # Extract
    result = results[0]
    if result.masks is not None:
        # data is tensor map, typically resized to original if we pass numpy array `img` directly
        masks = result.masks.data.cpu().numpy()
        
        # if masks were generated at model size instead of orig_shape, 
        # Ultralytics sometimes makes masks shape depending on retina_masks or orig_img
        # Let's ensure it matches orig_img:
        if masks.shape[1:] != (H, W):
            masks_resized = []
            for m in masks:
                masks_resized.append(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST))
            masks = np.array(masks_resized)
            
        merged_mask = np.max(masks, axis=0)
        
        # Apply exact color painting
        img[merged_mask > 0.5] = crack_color
    
    out_p = os.path.join(r"d:\555\output", fname)
    cv2.imwrite(out_p, img)
    print(f"Saved {out_p}")

print("Inference completed!")
