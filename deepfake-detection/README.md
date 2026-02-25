# Deepfake Detection System

**Authors:** Harsh, Caleb, Ricky, Anand  

A robust deepfake detection pipeline combining Convolutional Neural Networks (CNNs) and Variational Autoencoders (VAEs) to detect manipulated facial videos with high accuracy and low false negative rate.

---

## 🚨 Problem Statement

The rapid growth of deepfake technology has led to manipulated media capable of spreading misinformation, enabling fraud, and eroding trust in digital content.

Our goal was to build a system that:

- Detects deepfake video frames reliably  
- Prioritizes minimizing False Negatives (missed fakes)  
- Maintains high user trust by controlling False Positives  
- Generalizes across multiple real-world datasets  

---

## 📊 Datasets Used

We combined multiple benchmark datasets for robustness:

- FaceForensics++ (c23 subset)
- Celeb-DF v2
- Deepfake Detection Challenge (DFDC)
- WildDeepfake

Approximately 50k samples per dataset were curated for training and evaluation.

---

## 🔎 Exploratory Data Analysis

We performed multi-level EDA:

### Video-Level
- Frame count
- Frame rate
- File size

### Image-Level
- Resolution
- Brightness
- Contrast
- Sharpness (Laplacian variance)

### Pixel-Level
- RGB channel mean and standard deviation

We observed measurable distribution differences between real and fake content in brightness and contrast statistics, though not strongly separable by simple thresholds.

---

# 🏗 Base Models

## 1️⃣ Vanilla CNN

Architecture:
- 2 Conv layers + ReLU
- MaxPooling
- Fully connected layers
- Output: Binary logits (Real vs Fake)

Performance:
- Test Accuracy: 97.42%
- False Positive Rate: 2.65%
- False Negative Rate: 2.51%

Strong baseline but still missing fakes at scale.

---

## 2️⃣ ResNet18 (No Pretrained Weights)

- ~11.2M parameters
- Modified final layer for binary classification
- Trained from scratch

Performance:
- ~90% train accuracy
- ~72% test accuracy
- False Negative Rate: ~42%

Issues:
- Poor generalization
- High computational cost
- Overfitting on limited training samples

---

# 🔍 What Went Wrong?

We analyzed failure modes:

### 1. Latent Space Clustering
Real and fake samples were not cleanly separable.

### 2. Quality Metrics
Misclassifications correlated with:
- Sharpness variations
- Brightness shifts
- Contrast differences

### 3. Model Attention
Grad-CAM revealed the model often focused on irrelevant regions instead of manipulation artifacts.

---

# 🚀 Final Model: CNN + VAE Hybrid

To address limitations, we introduced:

### 🔸 Variational Autoencoder (VAE)
- Conv Encoder + ConvTranspose Decoder
- Latent dimension: 128
- Trained only on REAL images
- Fake images produce higher reconstruction error

### 🔸 Quality-Aware Loss
We computed degradation metrics:
- Sharpness difference
- Brightness difference
- Contrast difference

### 🔸 Custom Weighted Loss Function

We modified the classification loss:

L_Total = Σ_i (L_VAE-Rec(X_i) + L_Quality(X_i)) · L_BCE(M(X_i), y_i)

Key Idea:
- Images harder to reconstruct or with degraded quality get higher weight
- Inspired by soft-margin SVM sample weighting

Model size: ~1–2M parameters (lightweight)

---

# 📈 Final Results

Test Accuracy: 98.93%

Per-Class Performance:
- Real: 99.05%
- Fake: 98.81%

Error Rates:
- False Positive Rate: 0.95%
- False Negative Rate: 1.19%

Major improvements:
- Lower missed fakes
- Better generalization
- Cleaner latent separation
- More meaningful attention maps

---

# 🧠 Why This Works

- CNN extracts spatial artifacts
- VAE models distribution of real images
- Quality metrics capture degradation patterns
- Weighted loss emphasizes suspicious samples

This creates a distribution-aware classifier rather than a naive pixel classifier.

---

# 🛠 Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib / Seaborn
- Scikit-learn


---

# 🎯 Key Takeaways

- Deepfake detection requires more than raw CNN classification.
- Modeling the real image distribution improves anomaly detection.
- Weighting samples based on reconstruction difficulty boosts robustness.
- Lightweight models can outperform heavy architectures when designed carefully.

---

# 🔮 Future Work

- Temporal modeling using 3D CNNs or Transformers
- Frequency domain features
- Pretrained backbone integration
- Explainability improvements
- Deployment optimization

---
