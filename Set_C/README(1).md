# 🚚 Predict Late Deliveries & Identify Operational Segments

> **Data Science & AI/ML Project** — Statistical Analysis • Data Preprocessing • Classification • Clustering • Artificial Neural Network

## 📌 Project Overview

This project analyzes delivery-operation data to **predict whether a delivery will be late** and to **identify operational segments** using unsupervised learning.

The workflow combines:

- 📊 Descriptive statistics and statistical inference
- 🧹 Data auditing, missing-value treatment, encoding, and scaling
- ⚙️ Feature engineering
- 🤖 Logistic Regression classification
- 🎯 Baseline comparison using a Dummy Classifier
- 🔎 K-Means clustering for operational segmentation
- 🧠 Artificial Neural Network (ANN) classification
- 📈 Model evaluation using Accuracy, Precision, Recall, and F1-score
- 🧩 Confusion-matrix analysis
- 📉 Training vs. validation loss analysis

---

## 🎯 Objectives

1. Understand the distribution of operational variables.
2. Test whether the mean delivery distance differs between operational groups.
3. Prepare the dataset without data leakage.
4. Build a baseline and a Logistic Regression classifier for late-delivery prediction.
5. Evaluate classification performance on an unseen test set.
6. Segment operations using K-Means clustering.
7. Build an ANN and compare it with Logistic Regression.
8. Save predictions, split information, and cluster profiles for further analysis.

---

## 📂 Dataset

The project uses a dataset containing **300 records** and the following variables:

| Feature | Description |
|---|---|
| `record_id` | Unique record identifier |
| `distance` | Delivery distance |
| `load` | Operational load |
| `traffic` | Traffic-related measure |
| `staff` | Staffing measure |
| `group` | Operational group (`G1` / `G2`) |
| `late` | Target variable: `0` = not late, `1` = late |

### 🔍 Dataset audit

- Records: **300**
- Features including target: **7**
- Missing values: **15 in `distance` and 15 in `load`**
- Target distribution:
  - `0` → **155**
  - `1` → **145**
- Exact duplicate rows detected after loading: **0**

---

## 🧪 Data Preprocessing

The preprocessing pipeline was fitted using the training/fit partition and then applied to validation and test data.

### Steps

1. 🔍 Audit missing values and target distribution.
2. 🧹 Remove exact duplicate records.
3. ✂️ Split data using stratification:
   - Fit: **192 records**
   - Validation: **48 records**
   - Test: **60 records**
4. 🩹 Impute numeric missing values using the **median** calculated from the fit data.
5. ⚙️ Create an engineered feature:

```text
engineered_feature = load / (staff + 1)
```

6. 📏 Standardize numeric features using `StandardScaler`.
7. 🔤 One-hot encode `group`.
8. ✅ Verify that the transformed feature matrices contain only finite values.

### Final model features

```text
distance
load
traffic
staff
engineered_feature
group_G1
group_G2
```

The final input contains **7 features**.

---

## 📊 Exploratory & Statistical Analysis

### 📈 Distribution of Distance

The observed distance values in the fit partition had:

- Mean: **49.74**
- Median: **50.20**
- Sample standard deviation: **10.84**
- Observed non-missing values: **184**

![Distribution of Distance](plots/histplot.png)

---

### 🧮 Statistical Inference

A Welch's independent two-sample t-test was used to compare the mean distance of `G1` and `G2`.

**Hypotheses**

- **H₀:** Mean distance of G1 = Mean distance of G2
- **H₁:** Mean distance of G1 ≠ Mean distance of G2

Result:

- G1 observations: **104**
- G2 observations: **80**
- t-statistic: **0.7062**
- p-value: **0.4811**
- 95% CI for the overall observed mean: **[48.17, 51.32]**
- Decision at α = 0.05: **Do not reject H₀**

---

## 🧮 Linear Algebra Analysis

For the complete `distance` and `traffic` observations in the fit partition, the covariance matrix was calculated manually.

### Covariance matrix

```text
[[117.4161,  -1.2365],
 [-1.2365, 107.4336]]
```

### PCA-related results

- Largest eigenvalue: **117.5670**
- Variance share: **52.29%**
- Principal direction:

```text
[-0.9926, 0.1211]
```

---

# 🤖 Supervised Learning

## 1. Baseline — Dummy Classifier

A `DummyClassifier` using the most-frequent strategy was used as the baseline.

## 2. Logistic Regression

A Logistic Regression model was trained with:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### 📊 Test-set results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Dummy | 0.5167 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | **0.8500** | **0.8846** | **0.7931** | **0.8364** |

The Logistic Regression model produced non-zero performance across all reported classification metrics and was evaluated against the same unseen test partition as the baseline.

---

## 🧩 Logistic Regression Confusion Matrix

The test-set confusion matrix is:

```text
                 Predicted
                 0      1
Actual 0        28      3
Actual 1         6     23
```

![Logistic Regression Confusion Matrix](plots/logistic_confusion_matrix.png)

### Interpretation

- **True Negatives (TN): 28**
- **False Positives (FP): 3**
- **False Negatives (FN): 6**
- **True Positives (TP): 23**

A **false positive** means a delivery is predicted as late when it is actually not late.

A **false negative** means a delivery is predicted as not late when it is actually late.

---

# 🔎 Unsupervised Learning — K-Means

K-Means was evaluated for:

```text
k = 2, 3, 4
```

| k | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 719.6321 | **0.2463** |
| 3 | 599.6594 | 0.2071 |
| 4 | 535.3835 | 0.1902 |

The notebook selected **k = 2** based on the highest silhouette score among the tested values.

### 🧩 Cluster Profiles

| Cluster | Distance | Load | Traffic | Staff |
|---:|---:|---:|---:|---:|
| 0 | 48.3920 | 46.9950 | 49.0663 | 54.0350 |
| 1 | 53.1775 | 57.7293 | 49.8562 | 41.4267 |

These profiles provide a compact view of the operational characteristics represented by each cluster.

---

# 🧠 Deep Learning — Artificial Neural Network

The ANN was implemented using **PyTorch**.

### Architecture

```text
Input: 7 features
       ↓
Linear(7 → 16)
       ↓
ReLU
       ↓
Linear(16 → 8)
       ↓
ReLU
       ↓
Linear(8 → 1)
```

### Training configuration

- Optimizer: **Adam**
- Learning rate: **0.001**
- Loss function: **BCEWithLogitsLoss**
- Batch size: **16**
- Maximum epochs: **50**
- Early-stopping patience: **5**
- Trainable parameters: **273**
- Actual epochs completed: **34**

---

## 📉 ANN Training vs Validation Loss

![ANN Training vs Validation Loss](plots/ann_train_vs_loss.png)

The training loss decreased throughout training, while the validation loss decreased initially and then leveled off around the later epochs. Early stopping was triggered after the validation loss stopped improving according to the configured patience.

---

## 📊 ANN vs Logistic Regression

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.8500** | **0.8846** | **0.7931** | **0.8364** |
| ANN | 0.8167 | 0.8750 | 0.7241 | 0.7925 |

Both models were evaluated on the same test partition.

---

# 📁 Project Structure

```text
Predict-Late-Deliveries/
│
├── data/
│   └── raw/
│       └── set_b.csv
│
├── outputs/
│   ├── splits.csv
│   ├── logistic_test_predictions.csv
│   ├── ann_test_predictions.csv
│   └── cluster_profiles.csv
│
├── plots/
│   ├── histplot.png
│   ├── logistic_confusion_matrix.png
│   └── ann_train_vs_loss.png
│
├── Predict late deliveries and identify operational segments.ipynb
└── README.md
```

---

# 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Programming | 🐍 Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Statistics | SciPy |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Data Storage | CSV |
| Development | Jupyter Notebook |

---

# 📦 Main Python Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
```

---

# 🔄 End-to-End Workflow

```text
📥 Load Data
     ↓
🔍 Audit Data
     ↓
🧹 Remove Exact Duplicates
     ↓
✂️ Stratified Fit / Validation / Test Split
     ↓
🩹 Median Imputation
     ↓
⚙️ Feature Engineering
     ↓
📏 Feature Scaling
     ↓
🔤 One-Hot Encoding
     ↓
┌───────────────────────────────┐
│                               │
🤖 Logistic Regression      🔎 K-Means
│                               │
└──────────────┬────────────────┘
               ↓
          🧠 ANN Model
               ↓
📊 Test Evaluation & Comparison
               ↓
📁 Save Predictions & Profiles
```

---

# 💾 Generated Outputs

The notebook creates the following outputs:

### `splits.csv`
Contains the partition assignment for each `record_id`:

```text
record_id | partition
```

### `logistic_test_predictions.csv`

Contains:

```text
record_id
true_label
predicted_label
class_1_probability
```

### `ann_test_predictions.csv`

Contains:

```text
record_id
true_label
predicted_label
class_1_probability
```

### `cluster_profiles.csv`

Contains the mean operational features for each K-Means cluster.

---

# 🏁 Key Findings

- 📊 The dataset contains **300 records** with a nearly balanced target distribution.
- 🧹 Missing values occurred in `distance` and `load` and were handled through fit-data median imputation.
- 📐 The statistical test returned **p = 0.4811** for the G1 vs G2 distance comparison.
- 🤖 Logistic Regression achieved **0.8500 accuracy** and **0.8364 F1** on the test set.
- 🔎 Among the tested K-Means values, **k = 2** had the highest silhouette score (**0.2463**).
- 🧠 The ANN used a **7 → 16 → 8 → 1** architecture with **273 trainable parameters**.
- 📈 The ANN training process used validation monitoring and early stopping.
- 📊 Logistic Regression and ANN were evaluated using the same held-out test partition.

---

# 👨‍💻 Project Purpose

This project demonstrates an end-to-end **Data Science + AI/ML workflow** starting from raw operational data and progressing through:

**Data Understanding → Statistical Analysis → Preprocessing → Feature Engineering → Machine Learning → Clustering → Deep Learning → Evaluation → Output Generation**

It is suitable as a portfolio project demonstrating practical skills in **Python, Data Analysis, Machine Learning, Unsupervised Learning, and Deep Learning**.

---

## ⭐ Skills Demonstrated

`Python` • `Pandas` • `NumPy` • `Matplotlib` • `SciPy` • `Statistics` • `Data Cleaning` • `Feature Engineering` • `Scikit-learn` • `Logistic Regression` • `K-Means` • `Clustering` • `PyTorch` • `ANN` • `Model Evaluation` • `Confusion Matrix` • `Jupyter Notebook`

---

### 📌 Note

All numerical results and interpretations in this README are based on the supplied project notebook and its generated outputs.
