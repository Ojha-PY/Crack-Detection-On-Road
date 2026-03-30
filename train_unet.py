import os
import glob
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # We resize to a fixed multiple of 32 for UNet (e.g. 1024x1024 or 1216x1216)
        # To maintain the ultra fine detail, we use a large resolution.
        img = cv2.resize(img, (1216, 1216))
        # Important: use nearest neighbor for binary masks to preserve thin lines
        mask = cv2.resize(mask, (1216, 1216), interpolation=cv2.INTER_NEAREST)
        
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        
        # ImageNet Normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)
        
        # mask is 0 or 255 initially
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        return img, mask

def main():
    # UNet is heavily used for thin lines/medical crack segmentation
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    )
    model.to(device)

    dataset = CrackDataset(r"d:\555\dataset\images\train", r"d:\555\dataset\masks\train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Using Adam Optimizer with a lower LR for finetuning pre-trained weights
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    # Adding a learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)
    
    # We use a combined BCE with Dice loss for highly imbalanced thin structures
    # smp provides convenient losses, but we can write a simple one
    bce = torch.nn.BCEWithLogitsLoss()
    def compute_loss(pred, target):
        loss_bce = bce(pred, target)
        pred_sig = torch.sigmoid(pred)
        # Dice Loss
        smooth = 1e-6
        intersection = (pred_sig * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_sig.sum() + target.sum() + smooth)
        return loss_bce + dice_loss

    print("Training UNet...")
    for epoch in range(250):
        model.train()
        epoch_loss = 0
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = compute_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
        scheduler.step()

    torch.save(model.state_dict(), r"d:\555\unet_crack.pth")
    print("Training finished!")

if __name__ == '__main__':
    main()
