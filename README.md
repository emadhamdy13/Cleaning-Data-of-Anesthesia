# Medical Dataset Preprocessing & Pipeline Documentation

This repository documents the end-to-end data preparation, cleaning, feature engineering, and dimensionality reduction pipeline designed for building two Multi-Layer Perceptron (MLP) Neural Network models:
1. **Regression Model**: To predict intraoperative estimated blood loss (`intraop_ebl`).
2. **Classification Model**: To predict in-hospital mortality (`death_inhosp`).

---

## 📌 Dataset Overview
- **Initial Dataset Size**: 6,388 samples and 74 raw features.
- **Data Types**: Mixed clinical features including demographic data, laboratory results, surgical info, and patient vital signs (both numerical and categorical).
- **Target Variables**:
  - `intraop_ebl` (*Continuous / Regression Target*)
  - `death_inhosp` (*Binary Categorical / Classification Target*)

---

## 🛠️ Step-by-Step Preprocessing Pipeline

### 1. Data Cleaning & Missing Value Handling
- Identified missing values across all clinical variables.
- Applied rigorous medical-domain and algorithmic data imputation strategies to handle missing values.
- **Validation**: Generated missingness verification plots using `matplotlib` and `seaborn` to ensure 100% elimination of missing data (`total_missing = 0`).

### 2. Exploratory Data Analysis & Correlation Studies
- Standardized feature string types and handled whitespace/special string characters (`\n`, trailing spaces).
- Generated targeted Correlation Heatmaps using `seaborn` to examine feature inter-correlations as well as bivariate correlation against target variables (`intraop_ebl` and `death_inhosp`).

### 3. Target Separation (Preventing Data Leakage)
- Isolated feature matrix ($X$) from target variables ($y$) before performing any global transformations:
  - $X$: 72 predictive clinical features.
  - $y_{\text{reg}}$: `intraop_ebl`
  - $y_{\text{clf}}$: `death_inhosp`
- *Reasoning*: Separating targets strictly guarantees that target statistics do not leak into feature scaling or PCA decomposition steps.

### 4. Categorical Encoding (One-Hot Encoding)
- Identified 20 categorical/object features (e.g., surgical approach, patient status, anesthesia type).
- Converted non-numeric categorical columns into binary indicators using One-Hot Encoding (`pd.get_dummies(..., drop_first=True)`).
- Prevented multicollinearity (dummy variable trap) by dropping the first reference level.

### 5. Feature Scaling
- Standardized all continuous and encoded features using `StandardScaler` from `scikit-learn`:
  $$z = \frac{x - \mu}{\sigma}$$
- Scaling ensures equal feature variance weighting prior to principal components extraction and prevents gradient explosion in Multi-Layer Perceptrons (MLPs).

### 6. Dimensionality Reduction (PCA)
- Applied Principal Component Analysis (`PCA`) on scaled features ($X$):
  - Parameter: `n_components = 0.90` (Retaining 90% cumulative explained variance).
- Reduced high-dimensional encoded feature space down to **555 Principal Components**.
- **Benefits**: Eliminates noise, reduces computational latency, and prevents overfitting during neural network training.

---

## 📊 Final Dataset Dimensions
| Dataset Stage | Row Count ($N$) | Column Count ($D$) |
| :--- | :--- | :--- |
| **Raw Input Data** | 6,388 | 74 |
| **Cleaned & Encoded Data** | 3,233 (after filtering/cleaning) | High-dimensional (One-Hot) |
| **Final Preprocessed Features ($X_{\text{pca}}$)** | **3,233** | **555 Components** |

---

## 🚀 Next Steps
1. **Train-Test-Validation Split**: Perform stratified splitting for classification and random splitting for regression tasks.
2. **Architecture Design**: Construct MLP Regressor and MLP Classifier architectures using PyTorch or TensorFlow/Keras.
3. **Model Evaluation**: Track loss curves, RMSE/MAE for regression, and ROC-AUC/F1-Score for classification.
