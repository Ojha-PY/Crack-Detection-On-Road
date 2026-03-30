import os
import glob
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    )
    model.load_state_dict(torch.load(r"d:\555\unet_crack.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    crack_color = np.array([48, 44, 150], dtype=np.uint8) # BGR
    out_dir = r"d:\555\output"
    os.makedirs(out_dir, exist_ok=True)

    input_imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        input_imgs.extend(glob.glob(os.path.join(r"d:\555\test", ext)))
        
    for img_p in input_imgs:
        fname = os.path.basename(img_p)
        
        orig_img = cv2.imread(img_p)
        if orig_img is None: continue
        
        H, W = orig_img.shape[:2]
        
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (1216, 1216))
        
        tensor_img = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor_img = normalize(tensor_img).unsqueeze(0)
        tensor_img = tensor_img.to(device)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            pred = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
        # STRICT THRESHOLD FOR HIGH PRECISION
        # Only accept pixels the model is >= 90% confident are cracks.
        pred_mask = (cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST) > 0.9).astype(np.uint8) * 255
        
        # --- Noise Removal via Connected Components Filtering ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
        filtered_mask = np.zeros_like(pred_mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # AGGRESSIVELY remove small fragmented lines to ensure zero false positives scattered around
            if area < 500: 
                continue
                
            # Filter out non-linear massive blobs using Solidity (Area / Convex Hull Area)
            x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
            comp_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    # A crack is severely branching/linear (very low solidity). 
                    # If solidity > 0.4, it's increasingly blocky/square.
                    if solidity > 0.4:
                        continue
                        
            # If it passes the structure checks, it is a valid visual crack
            filtered_mask[labels == i] = 255
            
        # Optional: Skeletonize to thin out the blobs into clean 1px lines
        try:
            from skimage.morphology import skeletonize
            skel = skeletonize(filtered_mask > 0)
            final_mask = skel.astype(np.uint8) * 255
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            final_mask = cv2.dilate(final_mask, dilate_kernel, iterations=1)
        except ImportError:
            final_mask = filtered_mask

        # Apply pure red coloring only on the validated crack mask using alpha blending
        crack_color = np.array([0, 0, 255], dtype=np.uint8) # Pure Red BGR
        red_overlay = np.zeros_like(orig_img)
        red_overlay[final_mask > 0] = crack_color
        
        alpha = 0.5
        for c in range(3):
            orig_img[:, :, c] = np.where(
                final_mask > 0,
                (alpha * orig_img[:, :, c] + (1 - alpha) * red_overlay[:, :, c]).astype(np.uint8),
                orig_img[:, :, c]
            )
        
        out_p = os.path.join(out_dir, fname)
        cv2.imwrite(out_p, orig_img)
        print(f"Saved {out_p}")

if __name__ == '__main__':
    main()
