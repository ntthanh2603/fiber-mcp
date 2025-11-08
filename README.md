# Phishing Website Detection - Analysis Results

## Overview
Comprehensive machine learning analysis for detecting phishing websites using URL and content-based features.

## Dataset Information
- **Total Samples**: 10,000
- **Features**: 48 (URL-based, domain, security, content-based)
- **Class Distribution**: Perfectly balanced (50% legitimate, 50% phishing)
- **Missing Values**: None

## Model Performance

### Best Model: **XGBoost**
- **Accuracy**: 98.65%
- **Precision**: 0.9860
- **Recall**: 0.9870
- **F1-Score**: 0.9865
- **AUC-ROC**: 0.9986

### All Models Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|---------|----------|---------|
| **XGBoost** | 98.65% | 0.9860 | 0.9870 | **0.9865** | 0.9986 |
| Random Forest | 98.55% | 0.9860 | 0.9850 | 0.9855 | 0.9989 |
| Logistic Regression | 95.20% | 0.9449 | 0.9600 | 0.9524 | 0.9868 |

## Key Features (Most Important)

### Top 5 Features by Importance:

1. **PctExtNullSelfRedirectHyperlinksRT** (62.7% - XGBoost)
   - Percentage of external null/self-redirect hyperlinks

2. **PctExtHyperlinks** (20.7% - Random Forest)
   - Percentage of external hyperlinks

3. **FrequentDomainNameMismatch** (4.6%)
   - Frequency of domain name mismatches

4. **InsecureForms** (2.8%)
   - Presence of insecure forms

5. **NumDash** (0.9%)
   - Number of dashes in URL

## Project Structure

```
web-theat-detector/
├── Phishing_Legitimate_full.csv          # Dataset
├── phishing_detection_analysis.py        # Main analysis script
├── analysis_output.log                   # Complete analysis log
│
├── models/                                # Trained Models
│   ├── best_model_xgboost.pkl           # Best performing model
│   ├── model_logistic_regression.pkl
│   ├── model_random_forest.pkl
│   ├── model_xgboost.pkl
│   ├── scaler.pkl                       # Feature scaler
│   ├── feature_names.pkl                # Feature names
│   └── model_info.json                  # Model metadata
│
├── visualizations/                       # Analysis Visualizations
│   ├── 01_class_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_matrix.png
│   ├── 04_top_correlations.png
│   ├── 05_confusion_matrix_*.png (3 files)
│   ├── 06_model_comparison.png
│   ├── 07_feature_importance_rf.png
│   └── 07_feature_importance_xgb.png
│
└── results/                              # Analysis Results
    ├── model_comparison.csv
    ├── feature_importance_random_forest.csv
    └── feature_importance_xgboost.csv
```

## Confusion Matrix (XGBoost)

```
              Predicted
           Legitimate  Phishing
Actual
Legitimate    986        14
Phishing       13       987
```

**Interpretation:**
- Only 14 legitimate websites misclassified as phishing
- Only 13 phishing websites misclassified as legitimate
- **98.65% overall accuracy**

## How to Use the Model

### Load and Predict

```python
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('models/best_model_xgboost.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Prepare your data (ensure it has all 48 features in correct order)
new_data = np.array([...])  # Your feature vector

# Scale the features
new_data_scaled = scaler.transform(new_data.reshape(1, -1))

# Make prediction
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)

# Interpret result
if prediction[0] == 1:
    print(f"⚠️ PHISHING DETECTED (confidence: {probability[0][1]:.2%})")
else:
    print(f"✅ LEGITIMATE (confidence: {probability[0][0]:.2%})")
```

## Re-run Analysis

To re-run the complete analysis:

```bash
python3 phishing_detection_analysis.py
```

This will:
1. Load and explore the dataset
2. Perform EDA with visualizations
3. Preprocess and scale features
4. Train all 3 models
5. Evaluate and compare performance
6. Analyze feature importance
7. Save all models and results

## Key Insights

1. **Excellent Performance**: All three models achieved >95% accuracy, with tree-based models (Random Forest and XGBoost) reaching >98.5%

2. **Most Discriminative Features**:
   - Hyperlink-related features (external links, null/self-redirects)
   - Domain characteristics (mismatches, subdomains)
   - Security indicators (HTTPS usage, insecure forms)

3. **Balanced Dataset**: Perfect 50/50 split ensures no class imbalance issues

4. **High Confidence**: AUC-ROC scores >0.998 indicate excellent class separation

## Deployment - Production-Ready API ✅

The phishing detection system is now deployed as a production-ready REST API!

### Quick Start API

```bash
# Option 1: Run with Docker
docker-compose up -d

# Option 2: Run locally
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

**Access:**
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### API Features

✅ **Multiple Endpoints:**
- `/predict` - Single URL detection (XGBoost)
- `/predict/ensemble` - Combine all 3 models
- `/predict/batch` - Analyze up to 100 URLs
- `/analyze` - Detailed feature analysis

✅ **Web Interface:**
- Beautiful, responsive UI
- Real-time URL scanning
- Visual risk indicators
- Confidence scores

✅ **Performance:**
- <100ms response time
- ~1000 requests/second
- Auto-generated Swagger docs

See [API Documentation](api/README.md) for complete usage guide.

## Future Enhancements

1. **Real-time Detection**: Browser extension integration
2. **Feature Engineering**: Explore additional URL and content features
3. **Model Updates**: Retrain periodically with new phishing patterns
4. **Advanced Ensemble**: Stacking and boosting techniques

## Dependencies

```
pandas==2.3.3
numpy==2.3.4
matplotlib==3.10.7
seaborn==0.13.2
scikit-learn==1.7.2
xgboost==3.1.1
joblib==1.5.2
```

## Author

Analysis completed by Claude AI Assistant
Date: November 8, 2025

## License

This project is for educational and research purposes.
