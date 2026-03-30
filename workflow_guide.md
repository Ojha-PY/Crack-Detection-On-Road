# 🚀 Crack Detection Pipeline: Step-by-Step Guide

Welcome! Here is the complete workflow to take your new sample images, incorporate them into the dataset, retrain the model to make it smarter, and finally let it automatically draw the cracks for your team.

## Step 1: Prepare Your Environment

Whenever you start working on this project, ensure you open your PowerShell or command prompt and activate your dedicated Python environment:

```powershell
conda activate als
```
*(Alternatively, simply run the scripts universally using your direct Anaconda Python path: `C:\Users\Muktikanta_Ojha\anaconda3\envs\als\python.exe`)*

## Step 2: Add Your New Samples to the Dataset

The U-Net model `train_unet.py` expects your images and manually colored ground-truth masks to be placed securely inside specific folders:

* **Raw Images:** Place all your new `.jpg` concrete images exactly here:
  `d:\555\dataset\images\train`
* **Mask Images:** Ensure your team's matching ground-truth mask outlines (where cracks equal pure white `255`, and everything else is `0` black) are named identically and placed fully here:
  `d:\555\dataset\masks\train`

*(💡 Note: You can use your `prepare_dataset.py` script if you normally use that to automatically sort the raw outputs from your team into these directories).*

## Step 3: Run the Training Process

Once all your new images are nestled securely in their respective train folders, you retrain the model so it learns from them! The `train_unet.py` we built will automatically use its internal ImageNet normalization arrays, Cosine Annealing learning rates, and your `GPU 0` seamlessly.

Run this simple command inside `d:\555`:
```powershell
python train_unet.py
```

* **What Happens Next**: The process will quietly launch. It will complete 250 Epochs automatically over all your images, minimizing the loss down cleanly to ~0.10. 
* **The Result**: It will aggressively overwrite your old model weights with a massive new payload located in: `d:\555\unet_crack.pth`.

## Step 4: Run Real-World Inference on Unseen Images

Now that the new "brain" is fully serialized (`unet_crack.pth`), let's release it onto the real-world concrete:

1. Drop any completely new, unlabeled raw `.jpg` or `.png` camera images securely into your testing folder:
   `d:\555\test`
2. Once they are all inside, execute the clean inference script smoothly via:
   ```powershell
   python infer_unet.py
   ```
3. Your newly annotated, pure-red crack images—where noise has been structurally eliminated—will flawlessly pop out into your output directory:
   `d:\555\output`

## Tips on Tweaking the Pipeline Output

If you notice the model is missing some tiny cracks (False Negatives), or it randomly colors weird shadows as cracks (False Positives), you can easily open `infer_unet.py` in your code editor and change our two strict thresholds:

* **Probability Limit (Line 53)**: `pred > 0.9` -> Change this to `0.7` to make the model colorize more loosely (increasing Recall), or `0.95` to make it incredibly stubborn and color strictly on perfectly obvious cracks (maximizing Precision).
* **Area Filtering Limit (Line 60)**: `area < 500` -> Change to `200` to allow tiny dots/segments of lines to be saved, or `800` to aggressively blast away small features and retain ONLY enormous contiguous sprawling cracks!
