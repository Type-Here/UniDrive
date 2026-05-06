# Lane Segmentation Pipeline

Fine-tuning pipeline for SegFormer on a custom lane marking dataset.

## Project structure

```
pipeline/
    config.yaml               central configuration (edit this first)
    1_prepare_dataset.py      JSON -> masks, crop, resize, train/val/test split
    2_dataset.py              PyTorch Dataset + DataLoader + augmentation
    3_train.py                fine-tuning loop with early stopping
    4_evaluate.py             metrics, confusion matrix, prediction visualisation
    5_export.py               ONNX export + optional simplification + verification

data/
    raw/
        images/               original JPGs from LabelMe
        labels/               LabelMe JSON files (same stem as image)
    dataset/
        train/images/         preprocessed training images (640x256)
        train/masks/          segmentation masks (uint8, values 0-4)
        val/
        test/
        previews/             visual check images from prepare step

checkpoints/
    best.pth                  best val mIoU checkpoint
    last.pth                  latest epoch checkpoint

logs/
    train_log.csv             per-epoch metrics
    train_log.png             loss + mIoU curves
    confusion_test.png        confusion matrix on test split
    iou_test.png              per-class IoU bar chart
    results_test.txt          all metrics in plain text
    predictions/              grid: original / GT / pred / errors

exports/
    model.onnx                exported model
    model_simplified.onnx     simplified model (if --simplify used)
```

## Classes

| ID | Name         | Render order | Notes                        |
|----|--------------|--------------|------------------------------|
|  0 | background   | first        |                              |
|  1 | road         | second       | large polygon, less precise  |
|  2 | lane_marking | third        | solid boundary lines         |
|  3 | lane_dashed  | fourth       | dashed lane dividers         |
|  4 | zebra        | last         | individual crosswalk strips  |

Render order matters: later classes overwrite earlier ones where polygons overlap.

## Setup

```bash
pip install torch torchvision transformers albumentations \
            pyyaml opencv-python matplotlib

# Optional for export verification and simplification
pip install onnxruntime onnx onnxsim
```

## Step by step

### 1. Configure

Edit `config.yaml`. The key parameters to check:

```yaml
paths:
  raw_images: "data/raw/images"
  raw_labels: "data/raw/labels"

image:
  crop_top_frac: 0.45    # fraction of image height to remove from top
  model_h: 256           # must be multiple of 32
  model_w: 640

training:
  epochs: 100
  batch_size: 8
  lr: 6.0e-5
  class_weights: [0.5, 1.0, 3.0, 4.0, 5.0]   # update from step 1 output

model:
  name: "nvidia/mit-b1"  # or "nvidia/mit-b0" for lighter model
```

### 2. Prepare dataset

```bash
python3 1_prepare_dataset.py --config config.yaml --preview 10
```

Outputs:
- Preprocessed images and masks in `data/dataset/`
- Class frequency stats + suggested `class_weights` for config.yaml
- Preview images in `data/dataset/previews/`

**Relevant Operations:**
- Crop top 45% of image to remove sky/noise and focus on road
- Resize to 640x256 (must be multiple of 32 for SegFormer)
- Convert LabelMe JSON annotations to single-channel masks with class IDs
- Train/val/test split (80/10/10 by default)
- Calculate class frequencies and suggest `class_weights` to handle imbalance

**Notes:**
The top 45% of the image is pure noise (wall, ceiling, people) so we remove it with a numpy slice.  
The critical point is the different interpolation modes for resize:
- INTER_LINEAR for images: bilinear interpolation, smoothly blends neighboring pixels, gives visually good results for photos.
- INTER_NEAREST for masks: nearest neighbor, no blending at all. 

If used INTER_LINEAR on a mask, pixel values like 1, 2, 3, 4 would get blended into fractional values like 1.7 or 2.3 which are meaningless class IDs. 
Nearest neighbor snaps each output pixel to the exact value of the closest input pixel, preserving integer class IDs exactly.

### 3. Check dataset loading

```bash
python3 2_dataset.py --config config.yaml
```

Outputs a batch preview at `data/dataset/batch_preview.jpg`.
Verify that augmentation looks reasonable and masks are correct.

**Notes:**
The geometric transforms (flip, affine) are applied to both image and mask simultaneously by Albumentations.  
This is the main reason to use Albumentations rather than plain torchvision transforms, which only operate on images.

Photometric transforms (brightness, hue, blur, noise, shadow) are applied only to the image, not the mask.

#### Normalization
```
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```
These are the ImageNet mean and standard deviation across its 1.2 million images. 
The formula applied per channel is:
```
    pixel_out = (pixel_in / 255.0 - mean) / std
```
This because SegFormer's backbone (MixTransformer, mit-b1) was pretrained on ImageNet.

### 4. Train

```bash
# B1 model (default)
python3 3_train.py --config config.yaml

# B0 model (lighter, faster on Jetson Nano)
python3 3_train.py --config config.yaml --model nvidia/mit-b0

# Resume after interruption
python3 3_train.py --config config.yaml --resume checkpoints/last.pth
```

Monitor progress:
- Console prints one line per epoch with loss, mIoU, LR, time
- Per-class IoU is printed every 10 epochs
- `logs/train_log.csv` has all metrics
- `logs/train_log.png` plots loss and mIoU curves

#### Loss function
**CrossEntropyLoss** with class weights
```
    L = - sum(w_c * y_true_c * log(y_pred_c)) / N
```
Where `w_c` is the class weight for class c, `y_true_c` is the binary indicator (0 or 1) if class c 
is the correct class for the pixel, and `y_pred_c` is the predicted probability for class c.  
The weights help to balance classes that are underrepresented in the dataset.

#### Optimizer
**AdamW** with learning rate 6e-5 and weight decay 0.01, 
which is a common choice for fine-tuning transformer-based models.

#### Scheduler
**CosineAnnealingLR** with `T_max` equal to the number of epochs, which gradually reduces the learning rate 
following a cosine curve, allowing for better convergence.  
A warmup phase is implemented in the first 5 epochs where the learning rate starts from a small value 
and increases to the initial learning rate, which can help stabilize training in the early stages.

**Mixed precision**: `GradScaler` + `autocast`

**Gradient clipping**: `Grad Norm [model.parameters(), 1.0]`  
Clips the global gradient norm to 1.0. 
Transformers can occasionally produce very large gradient spikes (exploding gradients), 
especially early in training when the decoder is uninitialised. 

Clipping prevents a single bad batch from taking a catastrophically large parameter update.

### 5. Evaluate

```bash
# Evaluate best checkpoint on test split
python3 4_evaluate.py --checkpoint checkpoints/best.pth

# Evaluate on val split with more prediction images
python3 4_evaluate.py --checkpoint checkpoints/best.pth \
                      --split val --save-preds 20
```
IoU (Intersection over Union) is the main metric for segmentation. It is calculated per class as:
```
IoU_c = TP / (TP + FP + FN)
```
This is also called the Jaccard index. 
Geometrically it is the intersection of predicted and ground truth regions divided by their union. 

`Precision_c = TP / (TP + FP)`   # of all pixels predicted as c, how many are actually c  
`Recall_c    = TP / (TP + FN)`   # of all pixels that are c, how many did we find  
`F1_c        = 2 * P * R / (P + R)`  
`px_acc = np.diag(mat).sum() / mat.sum()`  

Outputs:
- `logs/results_test.txt` -- mIoU, pixel accuracy, per-class IoU/F1
- `logs/confusion_test.png` -- normalized confusion matrix
- `logs/iou_test.png` -- per-class IoU bar chart
- `logs/predictions/*.jpg` -- original | GT | prediction | error map

Prediction error map legend:
- Red   -- false positive (predicted as class but should be background/other)
- Blue  -- false negative (missed, should be class but predicted as other)

### 6. Export to ONNX

```bash
# Basic export
python3 5_export.py --checkpoint checkpoints/best.pth

# Export + simplify + verify
python3 5_export.py --checkpoint checkpoints/best.pth \
                    --simplify --verify
```

### 7. Convert to TensorRT on Jetson Nano

```bash
# Copy ONNX to Jetson, then:
trtexec --onnx=exports/model.onnx \
        --saveEngine=exports/model.trt \
        --fp16

# Or with explicit workspace size (512 MB)
trtexec --onnx=exports/model.onnx \
        --saveEngine=exports/model.trt \
        --fp16 \
        --workspace=512
```

## Tips

**Dataset too small (< 100 images):**
Increase augmentation probability, especially brightness/contrast variation
to simulate the three lighting conditions (normal, bright, dark).

**Class imbalance:**
Run `1_prepare_dataset.py` and copy the suggested `class_weights` into
`config.yaml` before training.

**Out of memory on training machine:**
Reduce `batch_size` in config or pass `--batch 4` on the command line.
With B0 and batch 4 you need about 6 GB VRAM.

**B0 vs B1:**
Train B1 first for best accuracy. If it does not meet the FPS requirement
on the Jetson, retrain with B0 using the same config.