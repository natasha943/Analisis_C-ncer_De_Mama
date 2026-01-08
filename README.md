# Predictive Analysis of the Wisconsin Breast Cancer Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A comprehensive machine learning approach combining **PCA**, **KNN**, and **Neural Networks** for breast cancer diagnosis support.

<p align="center">
  <img src="https://www.rededucom.org/img/proyectos/14/galeria/big/universidad-salesiana.jpg" width="200">
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [Authors](#authors)
- [References](#references)
- [License](#license)

---

## 🎯 Overview

This project implements a **robust diagnostic support system** for breast cancer classification using the Wisconsin Breast Cancer Diagnostic Dataset (WDBC). The analysis combines:

- **Exploratory Data Analysis (EDA)**
- **Dimensionality Reduction (PCA)**
- **Supervised Learning** (KNN, Neural Networks)
- **Unsupervised Learning** (K-Means Clustering)
- **Model Optimization** (GridSearchCV)

### Key Highlights

✅ **98.25% accuracy** achieved with KNN classifier  
✅ **95% variance** preserved with only 10 PCA components  
✅ **Perfect recall** for malignant class detection  
✅ **Unsupervised validation** confirms supervised learning patterns  

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

### Dataset Characteristics

| Property | Value |
|----------|-------|
| **Total Samples** | 569 patients |
| **Features** | 30 numerical measurements |
| **Target Variable** | Binary (Benign: 1, Malignant: 0) |
| **Missing Values** | None |
| **Class Distribution** | Imbalanced (357 benign, 212 malignant) |

### Feature Categories

1. **Mean values:** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
2. **Standard error values:** SE of above features
3. **Worst values:** largest/worst measurements of above features

---

## 📁 Project Structure

```
breast-cancer-analysis/
│
├── data/
│   ├── Examen_Listo_Fase1.csv          # Cleaned dataset
│   └── breast_cancer_original.csv       # Original dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb                     # Exploratory Data Analysis
│   ├── 02_Classification.ipynb          # Classification models
│   ├── 03_Regression.ipynb              # Regression analysis
│   └── 04_Clustering.ipynb              # Clustering implementation
│
├── src/
│   ├── preprocessing.py                 # Data preprocessing utilities
│   ├── models.py                        # Model definitions
│   ├── evaluation.py                    # Evaluation metrics
│   └── visualization.py                 # Plotting functions
│
├── results/
│   ├── figures/                         # Generated plots
│   └── metrics/                         # Performance metrics
│
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── LICENSE                              # MIT License
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/breast-cancer-analysis.git
cd breast-cancer-analysis
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
scipy>=1.7.0
```

---

## 🚀 Usage

### Quick Start

```python
# Load the cleaned dataset
import pandas as pd
df = pd.read_csv("data/Examen_Listo_Fase1.csv")

# Run complete analysis
python src/main.py --analysis all
```

### Running Individual Components

**1. Exploratory Data Analysis**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

**2. Classification Models**
```bash
python src/models.py --task classification
```

**3. Clustering Analysis**
```bash
python src/models.py --task clustering
```

### Making Predictions

```python
from src.models import load_model, predict

# Load trained model
model = load_model('results/best_knn_model.pkl')

# Make prediction
new_sample = [[17.99, 10.38, 122.8, ...]]  # 30 features
prediction = predict(model, new_sample)
print(f"Prediction: {'Malignant' if prediction == 0 else 'Benign'}")
```

---

## 🔬 Methodology

### Phase 1: Data Exploration & Preprocessing

#### 1.1 Statistical Analysis
- Descriptive statistics for all 30 features
- Distribution analysis by diagnosis class
- Correlation matrix visualization
- Outlier detection (IQR method)

#### 1.2 Feature Selection
- Top 10 features selected based on correlation with target
- Variables selected:
  - `worst concave points`
  - `worst perimeter`
  - `mean concave points`
  - `worst radius`
  - `mean perimeter`
  - `worst area`
  - `mean radius`
  - `mean area`
  - `mean concavity`
  - `worst concavity`

#### 1.3 Key Findings
- **Size variables** (radius, area, perimeter) show >200% difference between classes
- **High multicollinearity** detected (justifies PCA)
- **30.1% of rows** contain at least one outlier
- **Clear class separation** in morphological features

### Phase 2: Dimensionality Reduction

#### 2.1 PCA Implementation
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
```

**Results:**
- **10 components** explain 95% of variance
- Reduces from 30 to 10 dimensions
- Preserves essential discriminative information

### Phase 3: Supervised Learning

#### 3.1 Classification Models

##### **Model 1: K-Nearest Neighbors (KNN)**

**Hyperparameter Tuning (GridSearchCV):**
```python
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
```

**Performance:**
| Configuration | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| KNN (no PCA) | **98.25%** | 98.20% | **100%** | 99.09% |
| KNN (with PCA) | 95.61% | 95.45% | 97.67% | 96.55% |

##### **Model 2: Neural Network (Keras)**

**Architecture:**
```python
model = Sequential([
    Dense(64, activation='relu', input_dim=30),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

**Performance:**
| Configuration | Accuracy | Loss |
|--------------|----------|------|
| NN (no PCA) | 96.49% | 0.089 |
| NN (with PCA) | 96.49% | 0.092 |

#### 3.2 Regression Analysis

**Target Variable:** `mean area`

**Best Model:** KNN with PCA

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **KNN + PCA** | **56.45** | **80.31** | **0.9545** |
| KNN (no PCA) | 58.12 | 82.47 | 0.9512 |
| NN (no PCA) | 62.34 | 88.91 | 0.9432 |
| NN + PCA | 67.89 | 95.23 | 0.9301 |

### Phase 4: Unsupervised Learning (Clustering)

#### 4.1 Optimal K Selection (Elbow Method)

```python
from sklearn.cluster import KMeans

def descubrirK(X, max_k=10):
    distortions = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    return distortions
```

**Result:** K=2 identified as optimal (aligns with binary classification)

#### 4.2 K-Means Clustering

**Configuration:**
```python
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca[:, :10])
```

**Validation:**
- Clear spatial separation between clusters
- High alignment with true labels
- Independent confirmation of supervised learning patterns

---

## 📈 Results

### Classification Performance Summary

| Metric | KNN (Best) | Neural Network |
|--------|------------|----------------|
| **Accuracy** | 98.25% | 96.49% |
| **Precision** | 98.20% | 96.36% |
| **Recall** | 100.00% | 98.14% |
| **F1-Score** | 99.09% | 97.24% |
| **False Negatives** | **0** | 2 |

### Key Insights

1. **KNN without PCA** achieves the best overall performance
2. **Perfect recall** for malignant class (no false negatives)
3. **Neural networks** show stable performance with/without PCA
4. **PCA reduces dimensionality** by 67% with minimal accuracy loss
5. **Clustering validates** supervised learning patterns independently

### Confusion Matrix (KNN - Best Model)

```
                Predicted
              Benign  Malignant
Actual Benign    71       1
      Malignant   0      42
```

### Prediction Examples

#### Real Cases

**Benign Case:**
- Mean radius: 11.13, Mean area: 381.1
- **Prediction:** Benign (Prob: 1.0000) ✅

**Malignant Case:**
- Mean radius: 19.55, Worst area: 1926.0
- **Prediction:** Malignant (Prob: 1.0000) ✅

#### Synthetic Cases

Validated model consistency with artificially created edge cases confirming robustness.

---

## 🎨 Visualizations

### Correlation Heatmap
High correlation between size/shape features justifies PCA application.

### PCA Variance Explained
Exponential decay shows 10 components capture essential information.

### Elbow Method
Clear elbow at K=2 confirms binary nature of problem.

### Cluster Visualization
2D projection shows distinct spatial separation between diagnosed classes.

### Prediction Scatter Plots
Strong linear correlation between predicted and actual values in regression tasks.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include unit tests for new features

---

## 👥 Authors

**Erika Contreras** - *Data Analysis & Machine Learning*  
**Jorge Pizarro** - *Model Development & Evaluation*

**Universidad Politécnica Salesiana, Ecuador**

## 📚 References

1. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). *Nuclear feature extraction for breast tumor diagnosis.* University of Wisconsin.

2. Wolberg, W. H., & Mangasarian, O. L. (1990). *Multisurface method of pattern separation for medical diagnosis applied to breast cytology.* PNAS, 87(23), 9193–9196.

3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.

4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning.* Springer.

5. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.

[See full references in the paper]

<p align="center">
  Made with ❤️ for advancing medical AI diagnostics
</p>
