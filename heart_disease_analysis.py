"""
Predicting Heart Disease Risk Using Clinical and Demographic Data
ANLY 500 - Principles of Analytics
Author: Aashna Mahajan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=columns, na_values='?')

# Binary target: 0 = no disease, 1 = disease (original values 1-4 → 1)
df['target'] = (df['target'] > 0).astype(int)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nClass distribution:\n{df['target'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDescriptive statistics:\n{df.describe().round(2)}")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
# Drop rows with missing values (small number)
df.dropna(inplace=True)
print(f"\nShape after dropping missing values: {df.shape}")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Stratified train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features for distance-based models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================
fig_dir = "/Users/maashna/Downloads/figures"
import os
os.makedirs(fig_dir, exist_ok=True)

# 3a. Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(f'{fig_dir}/correlation_matrix.png', dpi=150)
plt.close()
print("\nSaved: correlation_matrix.png")

# 3b. Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Heart Disease Distribution')
plt.xlabel('0 = No Disease, 1 = Disease')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{fig_dir}/target_distribution.png', dpi=150)
plt.close()
print("Saved: target_distribution.png")

# 3c. Boxplots for key continuous features
cont_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(1, len(cont_features), figsize=(18, 4))
for ax, feat in zip(axes, cont_features):
    sns.boxplot(x='target', y=feat, data=df, ax=ax, palette='Set2')
    ax.set_title(feat)
    ax.set_xlabel('Target')
plt.suptitle('Feature Distributions by Heart Disease Status', y=1.02)
plt.tight_layout()
plt.savefig(f'{fig_dir}/boxplots.png', dpi=150)
plt.close()
print("Saved: boxplots.png")

# 3d. Histograms
df[cont_features].hist(figsize=(14, 8), bins=20, edgecolor='black')
plt.suptitle('Feature Histograms')
plt.tight_layout()
plt.savefig(f'{fig_dir}/histograms.png', dpi=150)
plt.close()
print("Saved: histograms.png")

# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Use scaled data for LR and KNN, unscaled for tree-based
def get_data(name):
    if name in ['Logistic Regression', 'KNN']:
        return X_train_scaled, X_test_scaled
    return X_train, X_test

results = {}
print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

for name, model in models.items():
    Xtr, Xte = get_data(name)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='accuracy')

    results[name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': roc_auc, 'fpr': fpr, 'tpr': tpr,
        'y_pred': y_pred, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
    }

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {roc_auc:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

# ============================================================
# 5. COMPARISON TABLE
# ============================================================
print("=" * 60)
print("SUMMARY COMPARISON")
print("=" * 60)
summary = pd.DataFrame({
    name: {k: v for k, v in r.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'cv_mean']}
    for name, r in results.items()
}).T.round(4)
print(summary.to_string())

# ============================================================
# 6. CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_title(f'{name}\nAcc: {r["accuracy"]:.2f}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.suptitle('Confusion Matrices', y=1.02)
plt.tight_layout()
plt.savefig(f'{fig_dir}/confusion_matrices.png', dpi=150)
plt.close()
print("\nSaved: confusion_matrices.png")

# ============================================================
# 7. ROC CURVES
# ============================================================
plt.figure(figsize=(8, 6))
for name, r in results.items():
    plt.plot(r['fpr'], r['tpr'], label=f'{name} (AUC={r["auc"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig(f'{fig_dir}/roc_curves.png', dpi=150)
plt.close()
print("Saved: roc_curves.png")

# ============================================================
# 8. FEATURE IMPORTANCE (Random Forest)
# ============================================================
rf = models['Random Forest']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
importances.plot(kind='barh', color='steelblue')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{fig_dir}/feature_importance.png', dpi=150)
plt.close()
print("Saved: feature_importance.png")

print("\n" + "=" * 60)
print("ALL DONE! Figures saved to:", fig_dir)
print("=" * 60)