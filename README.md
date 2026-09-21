# 🌑 The Pareidolia Paradox

A computer vision solution for classifying **lunar surface features as Depth or Rise** from 256×256 grayscale lunar surface images.

## 🎯 Problem

The challenge is to classify lunar surface crops into two categories:

* **Class 0 — Depth:** Craters, holes, and surface depressions
* **Class 1 — Rise:** Mounds, hills, rocks, and boulders

The main challenge is the effect of **sun azimuth** on the appearance of lunar terrain. The same physical feature can produce different shadow patterns depending on the direction of illumination.

The dataset consists of **7,854 labeled training images** and **2,000 test images**, along with metadata containing the corresponding sun azimuth angles.

The evaluation metric is **Balanced Accuracy**, calculated as the average recall across both classes.

## 🧠 Approach

Our solution combines image features with illumination information:

* **ResNet34** backbone for extracting visual features.
* **FiLM conditioning** to incorporate sun azimuth information.
* **Late-fusion azimuth conditioning** to provide additional lighting context.
* **Azimuth-aligned Sobel gradients** to capture surface changes along the illumination direction.
* **5-Fold Stratified Cross-Validation** for an ensemble of models.
* **Test-Time Augmentation (TTA)** during inference.

The directional gradient features help the model distinguish **convex structures such as rocks and mounds** from **concave structures such as craters and holes** based on their illumination and shadow patterns.

## 📊 Results

| Metric                       |                Result |
| ---------------------------- | --------------------: |
| Validation Balanced Accuracy |            **0.6826** |
| Macro F1-Score               |              **0.64** |
| Cross-Validation             | **5-Fold Stratified** |
| Epochs per Fold              |                 **3** |
| GPU                          |         **NVIDIA T4** |
| Test Predictions             |             **2,000** |

## 📁 Repository Structure

| File               | Description                                    |
| ------------------ | ---------------------------------------------- |
| `train.py`         | Training pipeline with 5-Fold Cross-Validation |
| `inference.py`     | Model inference, TTA, and ensemble predictions |
| `requirements.txt` | Python dependencies                            |
| `submission.csv`   | Final predictions for the 2,000 test images    |

## 🚀 Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

### Inference

```bash
python inference.py
```

The final predictions are generated in `submission.csv` with:

```text
image_id,label
```

## 🛠️ Tech Stack

**Python · PyTorch · torchvision · OpenCV · NumPy · scikit-learn**
