# Automated Concrete Crack Detection (U-Net)

A highly robust, precision-focused automated system for detecting and annotating concrete cracks in images. This tool utilizes a PyTorch **U-Net architecture (ResNet34 backbone)** optimized to prioritize high-confidence annotations, heavily minimizing false positives using aggressive post-processing structural filters.

## Features
- **State-of-the-Art Segmentation:** Powered by U-Net with pre-trained ResNet34 weights.
- **Auto Dataset Preparation:** Scripts to extract purely binary masks from input-output differencing.
- **Robust Training Pipeline:** Implements Cosine Annealing learning rate schedules and a combined BCE+Dice Loss designed to handle highly imbalanced thin structural details.
- **Aggressive Post-Processing:** Strict probability thresholding, continuous area (blob) filtering, and shape-solidity checking to guarantee zero random visual noise.
- **Automated Mask Overlay:** Flawlessly superimposes a sheer bright red overlay onto detected cracks across massive images.

---

## 💻 1. Environment & Installation Setup

We recommend using Anaconda or Miniconda. 

1. **Activate or create your Python environment**
```powershell
conda create -n als python=3.10
conda activate als
```

2. **Install required dependencies**
```powershell
pip install -r requirements.txt
```
*(Dependencies required: `torch`, `torchvision`, `opencv-python`, `numpy`, `segmentation-models-pytorch`, `scikit-image`)*

---

## 🚀 2. Preparing Your Dataset

To train the model, you need to provide original raw images and their corresponding truth masks.

### Automated Preparation Method
If your data comes as raw input images (e.g., `1_input.jpg`) and manually overlayed "colored" ground truth images (e.g., `1_output.jpg`), place them inside the `images/` directory.

Run the dataset generator script:
```powershell
python prepare_dataset.py
```
This script will automatically compare the "input" and "output" images, extract the differences as a clean black-and-white binary `.png` truth mask, and organize them into:
* `dataset/images/train/` (Your raw JPEG inputs)
* `dataset/masks/train/` (Your ground truth PNG masks)

### Manual Preparation Method
Simply place:
* All raw `.jpg` concrete images perfectly into `dataset/images/train/`
* All ground-truth mask outlines (where cracks = pure white `255`, and everything else is black `0`) heavily matched by identical names into `dataset/masks/train/`

---

## 🧠 3. Model Training

Once the images are sitting securely inside their respective train folders, you retrain the U-Net parameters so it learns from them!

Execute training:
```powershell
python train_unet.py
```

* **The Process**: The script automatically maps dataset dimensions, handles ImageNet normalizations, utilizes your GPU (CUDA), and runs through exactly **250 epochs**.
* **The Output**: Your highly optimized model weights will be aggressively serialized and saved as a large payload precisely at the root dictionary: `unet_crack.pth`.

---

## 🔍 4. Inference & Real-World Prediction

Now that your new "brain" (`unet_crack.pth`) is loaded, you can run inference against completely unseen concrete data!

1. **Add Test Images**: Drop any completely new, unlabeled raw `.jpg`, `.jpeg`, or `.png` camera images securely into the `test/` folder.
2. **Run Inference Workflow**:
```powershell
python infer_unet.py
```
3. **Review Results**: The accurately predicted annotations (colored beautifully in semi-transparent Pure Red) will be perfectly exported to the `output/` directory.

---

## ⚙️ 5. Advanced Configuration & Tweaking

If the model behaves too strictly or too loosely on production imagery, open `infer_unet.py` to directly manipulate to the filtering mechanisms:

| Feature | Code Line | Current Setting | Effect on Adjusting |
| :--- | :--- | :--- | :--- |
| **Probability Limit** | `pred > 0.9` | `0.9` (90% Confident) | Lower to `0.7` to make the model colorize more loosely (increasing Recall). Raise to `0.95` to make it incredibly stubborn (maximizing Precision). |
| **Area Check Limit** | `area < 500` | `500` pixels | Lower to `200` to allow tiny detached crack segments to survive. Raise to `800` to aggressively blast away small features and retain ONLY enormous contiguous sprawling cracks! |
| **Solidity Filter** | `solidity > 0.4`| `0.4` | Cracks naturally have heavily branching structures resulting in extremely low solidity. If set higher than `0.4`, it may allow non-linear, blocky pseudo-artifacts to sneak into annotations. |

---
**Maintained by the ALS Processing Team**
