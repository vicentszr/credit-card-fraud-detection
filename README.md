# 💳 Credit Card Fraud Detection

Machine learning model to detect fraudulent credit card transactions using an highly imbalanced dataset.

---

## 📊 Overview

This project builds and compares two classification models to detect fraudulent transactions from 284,807 real credit card transactions, where only 0.17% are fraudulent.

**Key findings:**
- Fraudulent transactions tend to have lower amounts (max ~2,000€ vs ~25,000€ for legitimate)
- Random Forest outperforms Logistic Regression in all metrics
- Random Forest detects 77% of frauds with 97% precision
- Only 2 false alarms out of 56,962 test transactions

---

## 🛠️ Tech Stack

- **Python 3**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualizations
- **Scikit-learn** — machine learning models and evaluation

---

## 📁 Project Structure 

credit-card-fraud-detection/
├── fraud_detection.ipynb   # Main analysis notebook
├── outputs/                # Generated visualizations
│   ├── 01_class_distribution.png
│   ├── 02_amount_distribution.png
│   └── 03_confusion_matrix.png
└── README.md
---

## 📈 Visualizations

### Class Distribution
![Class Distribution](outputs/01_class_distribution.png)

### Amount Distribution — Fraud vs Legitimate
![Amount Distribution](outputs/02_amount_distribution.png)

### Confusion Matrix Comparison
![Confusion Matrix](outputs/03_confusion_matrix.png)

---

## 📉 Model Comparison

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Precision (Fraud) | 85% | **97%** |
| Recall (Fraud) | 56% | **77%** |
| F1-score (Fraud) | 0.67 | **0.86** |
| False Alarms | 10 | **2** |

---

## 🚀 How to Run

1. Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in the project folder
3. Install dependencies: pip install pandas numpy matplotlib seaborn scikit-learn
4. Open and run `fraud_detection.ipynb`

---

## 📌 Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group). The raw data file is not included in this repository.

---

## 👤 Author

**Vicente Sánchez Reza**
[github.com/vicentszr](https://github.com/vicentszr)
