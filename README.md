# 🏡 Multi-Task House Price & Category Prediction  

This repository implements a **multi-task learning model** using **PyTorch Lightning** to predict both **house prices (regression)** and **house categories (classification)** simultaneously. The project leverages the **Kaggle House Prices dataset** with extensive feature engineering, preprocessing, and model optimization using **Optuna**.  

---

## 📂 Project Structure  

- `Final_Assignment-Julien_Palummo1.py` → Full Python code (data preprocessing, feature engineering, model architecture, training, and evaluation).  
- `Final_Assignment-Julien_Palummo (3).pdf` → Detailed project report with methodology, results, and future improvements.  
- `train.csv`, `test.csv`, `sample_submission.csv` → Kaggle "House Prices - Advanced Regression Techniques" dataset.  

---

## 📊 Dataset  

The dataset comes from **Kaggle’s House Prices competition** and includes:  

- **House attributes** → size, rooms, year built, renovations.  
- **Lot and neighborhood information** → land contour, zoning, condition.  
- **Quality ratings** → overall material and finish quality.  
- **Target variables**:  
  - $SalePrice$ → continuous price value (regression).  
  - $Category$ → derived categorical class based on engineered house type features (classification).  

This rich mix enables **multi-task learning**, combining regression and classification in a shared model.  

---

## ⚙️ Workflow  

### 1. Data Preparation & Feature Engineering  
- Merged `train`, `test`, and `sample_submission` for consistent preprocessing.  
- Dropped irrelevant/missing-heavy features (`PoolQC`, `MiscFeature`, `Alley`, `FireplaceQu`).  
- Imputed missing values (mean for numeric, mode for categorical).  
- Engineered features:  
  - **Age Category** (`New` vs `Older`).  
  - **Type Category** (Residential, Multi-Unit, Special).  
  - **House Style Category** (Standard, Complex, Other).  
  - Combined categories into a **Composite House Category**.  
- Outlier removal via Z-score on $SalePrice$.  
- One-hot encoding for categorical variables.  
- MinMax normalization for numerical features.  
- Feature selection via **SelectKBest** (top 20 features kept).  

---

### 2. Model Architecture  

Implemented a **shared-bottom neural network** with:  

- **Shared layers** → 3 linear blocks (BatchNorm + ReLU/LeakyReLU + residuals).  
- **Regression head** → predicts house prices.  
- **Classification head** → predicts house category.  

**Loss functions:**  

- **Regression:** Heteroscedastic loss (variance-aware):  

$$
\mathcal{L}_{reg} = \frac{1}{N} \sum_{i=1}^N \frac{(y_i - \hat{y}_i)^2}{2\sigma_i^2} + \frac{1}{2}\log(\sigma_i^2)
$$

- **Classification:** Cross-Entropy loss:  

$$
\mathcal{L}_{cls} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \log(\hat{y}_{ic})
$$

- **Total Objective:**  

$$
\mathcal{L} = \alpha \mathcal{L}_{reg} + \beta \mathcal{L}_{cls}
$$  

where $\alpha, \beta$ are task weights.  

**Optimization:** Hyperparameter tuning with **Optuna** (learning rate, hidden size, activation, optimizer).  

---

### 3. Training & Evaluation  

- Train / validation / test split.  
- Early stopping + checkpointing.  
- Metrics:  
  - **Regression:** RMSE.  
  - **Classification:** Accuracy, Precision, Recall, F1-score.  

---

## 📈 Results  

- **Regression task:** RMSE ≈ **180,195**.  
- **Classification task:** Accuracy ≈ **42%**, F1 ≈ **40.5%**.  
- **Best hyperparameters (Optuna):**  
  - Learning rate: 0.0022  
  - Hidden size: 64  
  - Activation: Leaky ReLU  
  - Optimizer: Adam  

---

## 🔮 Future Work  

- Add external features (location, economic indicators).  
- Explore CNN/RNN architectures for feature extraction.  
- Improve regularization for generalization.  
- Multi-modal learning with structured + unstructured data.  
- Deploy as a web app for interactive predictions.  

---

## 🛠️ Tech Stack  

- **Language:** Python  
- **Frameworks:** PyTorch Lightning, Optuna, Scikit-learn  
- **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, TorchMetrics  

---

## 📘 References  

- Kaggle: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
