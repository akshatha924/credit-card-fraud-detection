print("Starting Credit Card Fraud Detection...")

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data/creditcard.csv')

# Preview dataset
print("Preview of dataset:")
print(data.head())

# Class distribution
print("\n--- Class Distribution ---")
print(data['Class'].value_counts())

# Percentage distribution
fraud_percent = (data['Class'].value_counts(normalize=True) * 100).round(2)
print("\n--- Class Distribution (%) ---")
print(fraud_percent)

# Plot class distribution
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data, hue='Class', palette='Set2', legend=False)
plt.title('Transaction Class Distribution')
plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'])
plt.ylabel('Count')
plt.xlabel('Transaction Type')
plt.tight_layout()
plt.show()

# Check for missing values
print("\n--- Missing Values ---")
print(data.isnull().sum())

# Describe numerical features
print("\n--- Data Description ---")
print(data.describe())

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

print("\n--- Features and Target Separated ---")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- Data Split Completed ---")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Apply SMOTE to balance data
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- After SMOTE ---")
print("Resampled class distribution:")
print(y_train_resampled.value_counts())

# Train model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)
# Train the model on resampled data
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# Predict on test data (same as before)
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))


# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n--- Model Evaluation ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
from sklearn.metrics import confusion_matrix, classification_report

# Predict again (if not already done)
y_pred = model.predict(X_test)

# Confusion matrix and classification report
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, digits=4))
from imblearn.over_sampling import SMOTE

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- After SMOTE Resampling ---")
print("Resampled X_train shape:", X_train_resampled.shape)
print("Resampled y_train distribution:")
print(y_train_resampled.value_counts())
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest on resampled data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predict on test data
y_pred = rf_model.predict(X_test)
# Save actual vs predicted results
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_csv('fraud_predictions.csv', index=False)
print("✅ Predictions saved in fraud_predictions.csv")

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")  # Saves the image
plt.show()
import joblib

# Save the trained Random Forest model
joblib.dump(rf_model, 'random_forest_fraud_model.pkl')
print("✅ Model saved successfully as random_forest_fraud_model.pkl")
import joblib

# Load the saved model
loaded_model = joblib.load('random_forest_fraud_model.pkl')

# Predict using loaded model
predictions = loaded_model.predict(X_test)
# Save the cleaned/preprocessed dataset
data.to_csv('cleaned_creditcard_data.csv', index=False)
print("✅ Cleaned dataset saved as cleaned_creditcard_data.csv")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png")
plt.show()

# 2. Feature Importance Plot (RandomForest)
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[-10:]  # Top 10 features

plt.figure(figsize=(8, 5))
plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
