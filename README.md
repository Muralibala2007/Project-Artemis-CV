# 🌑 The Pareidolia Paradox

A computer vision solution for classifying **lunar surface features as Depth or Rise** from 256×256 grayscale lunar surface images.

## 🎯 Problem

The challenge is to classify lunar surface crops into two categories:

* **Class 0 — Depth:** Craters, holes, and surface depressions
* **Class 1 — Rise:** Mounds, hills, rocks, and boulders

The main challenge is the effect of **sun azimuth** on the appearance of lunar terrain. The same physical feature can produce very different shadow patterns depending on the direction of illumination.

The dataset consists of **7,854 labeled training images** and **2,000 test images**, along with metadata containing the corresponding sun azimuth angles.

The evaluation metric is **Balanced Accuracy**, calculated as the average recall across both classes.

## 🧠 Approach

Our solution combines visual features with physics-informed illumination information:

* **ResNet34 Backbone:** Used to extract visual features from the lunar surface images.

* **Azimuth-Aligned Sobel Gradients:** Multi-scale Sobel gradients are projected along the sun's lighting direction to capture illumination-dependent surface changes. These gradient channels are combined with the normalized grayscale image to form a **3-channel input**.

* **Sin/Cos Azimuth Encoding:** The `sun_azimuth_angle` is represented as **`(sin θ, cos θ)`** rather than using the raw angle. Since azimuth is circular, this avoids the artificial discontinuity between **0° and 360°** and provides the model with a continuous representation of the lighting direction.

* **FiLM Conditioning:** The `(sin θ, cos θ)` features are used to condition the ResNet through **Feature-wise Linear Modulation (FiLM)** after Layer 2.

* **Late Fusion:** The azimuth features are also incorporated at the classification head, allowing the final prediction to consider both visual features and lighting direction.

* **5-Fold Stratified Cross-Validation:** Five models are trained using stratified folds and combined as an ensemble.

* **Test-Time Augmentation (TTA):** During inference, augmented images are evaluated and the azimuth representation is adjusted consistently with horizontal flips.

This combination helps distinguish **convex structures such as rocks and mounds** from **concave structures such as craters and holes**, even when their appearance changes with illumination.

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
