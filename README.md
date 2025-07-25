# credit-card-fraud-detection
A data analytics project to detect fraudulent credit card transactions using Python and machine learning
# 💳 Credit Card Fraud Detection using Machine Learning

This project aims to detect fraudulent credit card transactions using machine learning techniques. Due to a significant class imbalance in real-world data, special attention is given to handling rare fraud cases.

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraud Cases**: 492 (~0.17%)
- **Attributes**: 30 features (including `Time`, `Amount`, and anonymized PCA-transformed columns `V1` to `V28`)

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib & Seaborn
- SMOTE (Synthetic Minority Oversampling Technique)
- VS Code
- Git & GitHub

---

## 🔍 Problem Statement

Fraudulent transactions represent less than 0.2% of the total. This project focuses on:

- Cleaning and preprocessing data
- Balancing the dataset using SMOTE
- Training machine learning models
- Evaluating model performance with metrics like precision, recall, and F1-score

---

## 🧪 Models Trained

- Logistic Regression
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)

---

## 📈 Performance Metrics

Due to imbalance, **accuracy** is not enough. Hence we focus on:

- Precision
- Recall
- F1-Score
- ROC AUC Curve

---

## 📁 Project Structure

