"""
Phishing Website Detection - Complete Analysis and Model Training
Author: Claude
Date: 2025-11-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("PHISHING WEBSITE DETECTION - ANALYSIS AND MODEL TRAINING")
print("="*80)

# ============================================================================
# 1. LOAD AND EXPLORE DATASET
# ============================================================================
print("\n" + "="*80)
print("1. LOADING AND EXPLORING DATASET")
print("="*80)

# Load dataset
df = pd.read_csv('Phishing_Legitimate_full.csv')

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   - Total samples: {df.shape[0]:,}")
print(f"   - Total features: {df.shape[1] - 1} (excluding target)")

print("\n📋 First 5 rows:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

print("\n🔍 Missing Values:")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("   ✅ No missing values found!")
else:
    print(missing_values[missing_values > 0])

print("\n🏷️ Feature Names:")
print(f"   Total features: {len(df.columns) - 1}")
print(f"   Features: {list(df.columns[:-1])[:10]}... (showing first 10)")

print("\n🎯 Target Variable (CLASS_LABEL):")
class_distribution = df['CLASS_LABEL'].value_counts()
print(class_distribution)
print(f"\n   Class Balance:")
print(f"   - Legitimate (0): {class_distribution[0]:,} ({class_distribution[0]/len(df)*100:.2f}%)")
print(f"   - Phishing (1): {class_distribution[1]:,} ({class_distribution[1]/len(df)*100:.2f}%)")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("2. EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# Create visualizations directory
import os
os.makedirs('visualizations', exist_ok=True)

# 2.1 Class Distribution
print("\n📊 Creating class distribution visualization...")
plt.figure(figsize=(8, 6))
class_counts = df['CLASS_LABEL'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.bar(['Legitimate (0)', 'Phishing (1)'], class_counts.values, color=colors, alpha=0.7, edgecolor='black')
plt.title('Class Distribution', fontsize=16, fontweight='bold')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Class', fontsize=12)
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: visualizations/01_class_distribution.png")

# 2.2 Feature Distributions (Top 10 features)
print("\n📊 Creating feature distribution visualizations...")
features_to_plot = df.columns[1:11]  # First 10 features
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for idx, feature in enumerate(features_to_plot):
    axes[idx].hist(df[df['CLASS_LABEL']==0][feature], bins=30, alpha=0.6, label='Legitimate', color='green')
    axes[idx].hist(df[df['CLASS_LABEL']==1][feature], bins=30, alpha=0.6, label='Phishing', color='red')
    axes[idx].set_title(feature, fontsize=10, fontweight='bold')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Feature Distributions (First 10 Features)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visualizations/02_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: visualizations/02_feature_distributions.png")

# 2.3 Correlation Matrix
print("\n📊 Creating correlation matrix...")
# Calculate correlation matrix (sample features due to large size)
sample_features = df.columns[1:21]  # First 20 features + target
corr_matrix = df[list(sample_features) + ['CLASS_LABEL']].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix (First 20 Features + Target)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/03_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: visualizations/03_correlation_matrix.png")

# 2.4 Features most correlated with target
print("\n📊 Top 10 features correlated with CLASS_LABEL:")
all_features = df.drop(['id', 'CLASS_LABEL'], axis=1, errors='ignore')
target_corr = df[all_features.columns.tolist() + ['CLASS_LABEL']].corr()['CLASS_LABEL'].abs().sort_values(ascending=False)
top_features = target_corr[1:11]  # Exclude self-correlation
print(top_features)

plt.figure(figsize=(12, 6))
top_features.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 10 Features Correlated with CLASS_LABEL', fontsize=16, fontweight='bold')
plt.xlabel('Absolute Correlation', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/04_top_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: visualizations/04_top_correlations.png")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("3. DATA PREPROCESSING")
print("="*80)

# Prepare features and target
X = df.drop(['CLASS_LABEL', 'id'], axis=1, errors='ignore')
y = df['CLASS_LABEL']

print(f"\n📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")

# Check for missing values
if X.isnull().sum().sum() > 0:
    print("\n⚠️ Handling missing values...")
    X = X.fillna(X.median())
    print("   ✅ Missing values filled with median")
else:
    print("\n✅ No missing values to handle")

# Feature Scaling
print("\n⚙️ Applying StandardScaler for feature scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✅ Features scaled successfully")

# Save feature names for later use
feature_names = X.columns.tolist()
print(f"\n📋 Total features for modeling: {len(feature_names)}")

# ============================================================================
# 4. TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("4. TRAIN/TEST SPLIT (80/20)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"📊 Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\n   Train class distribution:")
print(f"   - Legitimate (0): {sum(y_train == 0):,}")
print(f"   - Phishing (1): {sum(y_train == 1):,}")
print(f"\n   Test class distribution:")
print(f"   - Legitimate (0): {sum(y_test == 0):,}")
print(f"   - Phishing (1): {sum(y_test == 1):,}")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("5. TRAINING MULTIPLE MODELS")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1)
}

trained_models = {}
results = {}

for name, model in models.items():
    print(f"\n🤖 Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"   ✅ {name} trained successfully")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("6. MODEL EVALUATION")
print("="*80)

# Create results directory
os.makedirs('results', exist_ok=True)

evaluation_results = []

for name, model in trained_models.items():
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {name}")
    print(f"{'='*80}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    evaluation_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    })

    # Print metrics
    print(f"\n📈 Metrics:")
    print(f"   - Accuracy:  {accuracy:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall:    {recall:.4f}")
    print(f"   - F1-Score:  {f1:.4f}")
    print(f"   - AUC-ROC:   {auc_roc:.4f}")

    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'visualizations/05_confusion_matrix_{name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create comparison table
print("\n" + "="*80)
print("📊 MODEL COMPARISON")
print("="*80)
results_df = pd.DataFrame(evaluation_results)
results_df = results_df.sort_values('F1-Score', ascending=False)
print("\n" + results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('results/model_comparison.csv', index=False)
print("\n✅ Saved: results/model_comparison.csv")

# Plot model comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(metrics))
width = 0.25

for idx, (_, row) in enumerate(results_df.iterrows()):
    values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['AUC-ROC']]
    axes[0].bar(x + idx*width, values, width, label=row['Model'], alpha=0.8)

axes[0].set_xlabel('Metrics', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics, rotation=0)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0.5, 1.0])

# ROC Curves
for name, model in trained_models.items():
    y_pred_proba = results[name]['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = results[name]['auc_roc']
    axes[1].plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)

axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
axes[1].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/06_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: visualizations/06_model_comparison.png")

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("7. FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Random Forest Feature Importance
print("\n🌲 Random Forest Feature Importance:")
rf_model = trained_models['Random Forest']
rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(rf_importance.head(15))

# Plot Random Forest Feature Importance
plt.figure(figsize=(12, 8))
top_n = 20
rf_importance_top = rf_importance.head(top_n)
plt.barh(range(len(rf_importance_top)), rf_importance_top['Importance'], color='forestgreen', edgecolor='black')
plt.yticks(range(len(rf_importance_top)), rf_importance_top['Feature'])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title(f'Top {top_n} Feature Importance - Random Forest', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/07_feature_importance_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: visualizations/07_feature_importance_rf.png")

# XGBoost Feature Importance
print("\n🚀 XGBoost Feature Importance:")
xgb_model = trained_models['XGBoost']
xgb_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(xgb_importance.head(15))

# Plot XGBoost Feature Importance
plt.figure(figsize=(12, 8))
xgb_importance_top = xgb_importance.head(top_n)
plt.barh(range(len(xgb_importance_top)), xgb_importance_top['Importance'], color='steelblue', edgecolor='black')
plt.yticks(range(len(xgb_importance_top)), xgb_importance_top['Feature'])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title(f'Top {top_n} Feature Importance - XGBoost', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/07_feature_importance_xgb.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: visualizations/07_feature_importance_xgb.png")

# Save feature importance to CSV
rf_importance.to_csv('results/feature_importance_random_forest.csv', index=False)
xgb_importance.to_csv('results/feature_importance_xgboost.csv', index=False)
print("\n✅ Saved feature importance files to results/")

# ============================================================================
# 8. SAVE BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("8. SAVING BEST MODEL")
print("="*80)

# Find best model based on F1-Score
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_f1_score = results_df.iloc[0]['F1-Score']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score: {best_f1_score:.4f}")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save best model
model_filename = f'models/best_model_{best_model_name.lower().replace(" ", "_")}.pkl'
joblib.dump(best_model, model_filename)
print(f"\n✅ Best model saved: {model_filename}")

# Save scaler
scaler_filename = 'models/scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"✅ Scaler saved: {scaler_filename}")

# Save feature names
feature_filename = 'models/feature_names.pkl'
joblib.dump(feature_names, feature_filename)
print(f"✅ Feature names saved: {feature_filename}")

# Save all models
for name, model in trained_models.items():
    filename = f'models/model_{name.lower().replace(" ", "_")}.pkl'
    joblib.dump(model, filename)
print(f"\n✅ All models saved to models/ directory")

# Create model info file
model_info = {
    'best_model': best_model_name,
    'best_f1_score': float(best_f1_score),
    'all_models': results_df.to_dict('records'),
    'feature_count': len(feature_names),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

import json
with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print("✅ Model info saved: models/model_info.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)

print(f"""
📁 Output Structure:

   📂 visualizations/
      ├── 01_class_distribution.png
      ├── 02_feature_distributions.png
      ├── 03_correlation_matrix.png
      ├── 04_top_correlations.png
      ├── 05_confusion_matrix_*.png (3 files)
      ├── 06_model_comparison.png
      ├── 07_feature_importance_rf.png
      └── 07_feature_importance_xgb.png

   📂 results/
      ├── model_comparison.csv
      ├── feature_importance_random_forest.csv
      └── feature_importance_xgboost.csv

   📂 models/
      ├── best_model_{best_model_name.lower().replace(" ", "_")}.pkl
      ├── model_logistic_regression.pkl
      ├── model_random_forest.pkl
      ├── model_xgboost.pkl
      ├── scaler.pkl
      ├── feature_names.pkl
      └── model_info.json

🏆 Best Performing Model: {best_model_name}
   F1-Score: {best_f1_score:.4f}

📊 All Models Performance:
""")

print(results_df.to_string(index=False))

print("\n" + "="*80)
print("Thank you for using Phishing Detection Analysis!")
print("="*80)
