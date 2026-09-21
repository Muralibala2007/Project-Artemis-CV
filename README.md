# 🌑 The Pareidolia Paradox

A computer vision solution for classifying **lunar surface features as Depth or Rise** from 256×256 grayscale lunar surface images.

## 🎯 Problem

The challenge is to classify lunar surface crops into two categories:

* **Class 0 — Depth:** Craters, holes, and surface depressions
* **Class 1 — Rise:** Mounds, hills, rocks, and boulders

A major challenge is that the appearance of a lunar feature changes depending on the direction of sunlight. The same physical structure can produce very different shadow patterns under different **sun azimuth angles**.

The dataset contains:

* **7,854** labeled training images
* **2,000** test images
* `train_metadata.csv`
* `test_metadata.csv`

The evaluation metric is **Balanced Accuracy**, calculated as the average recall across both classes.

## 🧠 Approach

The solution combines visual features with illumination information:

* **ResNet34** backbone for extracting image features.
* **FiLM conditioning** to incorporate the sun azimuth information into the visual representation.
* **Late-fusion azimuth conditioning** to provide additional lighting context.
* **Azimuth-aligned Sobel gradients** to capture surface changes along the direction of illumination.
* **5-Fold Stratified Cross-Validation** to train an ensemble of models.
* **Test-Time Augmentation (TTA)** during inference for more robust predictions.

The directional gradient features are particularly useful for distinguishing **convex structures such as rocks and mounds** from **concave structures such as craters and holes**, where shadow geometry plays an important role.

## 📊 Results

| Metric                       |                Result |
| ---------------------------- | --------------------: |
| Validation Balanced Accuracy |            **0.6826** |
| Macro F1-Score               |              **0.64** |
| Cross-Validation             | **5-Fold Stratified** |
| Epochs per Fold              |                 **3** |
| GPU                          |         **NVIDIA T4** |
| Test Predictions             |             **2,000** |

The validation score is based on the **out-of-fold predictions** from the 5-fold ensemble.

## 📁 Repository Structure

```text
├── train.py
├── inference.py
├── submission.csv
└── README.md
```

* `train.py` — Trains the 5-fold ensemble.
* `inference.py` — Runs TTA, ensembles the fold predictions, and generates `submission.csv`.

## 🚀 Usage

### Training

```bash
python train.py
```

### Inference

```bash
python inference.py
```

The final submission is saved as:

```text
submission.csv
```

with the required columns:

```text
image_id,label
```

## 🛠️ Tech Stack

**Python · PyTorch · torchvision · OpenCV · NumPy · scikit-learn**
