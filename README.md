# credit-card-fraud-detection
A data analytics project to detect fraudulent credit card transactions using Python and machine learning
# 💳 Credit Card Fraud Detection

This project is a beginner-friendly Data Analytics task that uses Machine Learning to detect fraudulent credit card transactions. It is based on an imbalanced dataset and includes data preprocessing, model building, and evaluation.

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions
- **Fraudulent**: 492 (0.17%)
- **Legitimate**: 284,315 (99.83%)

## 🛠 Tools & Technologies

- Python
- VS Code
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SMOTE (for handling class imbalance)

## 🧪 Steps Performed

1. **Data Cleaning & Exploration**
   - Null value check
   - Class imbalance analysis

2. **Data Preprocessing**
   - Feature scaling using StandardScaler
   - Train-Test split

3. **Handling Imbalanced Data**
   - Used **SMOTE** to oversample the minority class

4. **Model Building**
   - Logistic Regression
   - Random Forest (optional)
   - Evaluation using:
     - Confusion Matrix
     - Classification Report (Precision, Recall, F1-Score)

## 📊 Output Examples

| Class Distribution | Before SMOTE | After SMOTE |
|-------------------|--------------|-------------|
| Fraudulent (1)    | 492          | 284,315     |
| Legitimate (0)    | 284,315      | 284,315     |

### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

### Classification Report
Precision, Recall, F1-Score, and Accuracy.

## 📌 Conclusion

This project demonstrates how machine learning can be applied to detect fraud with high precision, even with imbalanced data. SMOTE significantly improved model performance by balancing the dataset.

## 🧑‍💻 Author

**Akshatha B**  
Computer Science Student | Aspiring Data Analyst  
🔗 [LinkedIn](www.linkedin.com/in/
akshatha-b-347a22328)  
