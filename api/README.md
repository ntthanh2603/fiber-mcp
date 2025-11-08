# Phishing Detection API

A production-ready REST API for real-time phishing website detection using machine learning.

## Features

- **Real-time Detection**: Analyze URLs instantly for phishing threats
- **Multiple Models**: XGBoost, Random Forest, and Logistic Regression
- **Ensemble Predictions**: Combine all models for more robust detection
- **Batch Processing**: Analyze up to 100 URLs in a single request
- **Feature Analysis**: Detailed feature extraction and importance scoring
- **Auto-generated Documentation**: Interactive Swagger UI
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **High Performance**: FastAPI with async support

## Quick Start

### Option 1: Run Locally

1. **Install Dependencies**
```bash
cd api
pip install -r requirements.txt
```

2. **Start the API**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. **Access the API**
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Option 2: Run with Docker

1. **Build and Run**
```bash
docker-compose up -d
```

2. **Check Status**
```bash
docker-compose ps
docker-compose logs -f
```

3. **Stop**
```bash
docker-compose down
```

## API Endpoints

### General

#### `GET /`
Root endpoint with API information

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2025-11-08T12:00:00.000000",
  "version": "1.0.0"
}
```

#### `GET /models`
List available models and their performance metrics

**Response:**
```json
{
  "models": [
    {
      "name": "XGBoost",
      "accuracy": 0.9865,
      "f1_score": 0.9865,
      "feature_count": 48
    },
    ...
  ],
  "default_model": "XGBoost"
}
```

### Prediction Endpoints

#### `POST /predict`
Predict using the best model (XGBoost)

**Request:**
```json
{
  "url": "https://example.com",
  "html_content": null
}
```

**Response:**
```json
{
  "url": "https://example.com",
  "is_phishing": false,
  "confidence": 0.95,
  "risk_level": "LOW",
  "model_used": "XGBoost",
  "timestamp": "2025-11-08T12:00:00.000000"
}
```

#### `POST /predict/xgboost`
Predict using XGBoost model

#### `POST /predict/random-forest`
Predict using Random Forest model

#### `POST /predict/logistic-regression`
Predict using Logistic Regression model

#### `POST /predict/ensemble`
Predict using ensemble of all three models

**Response:**
```json
{
  "url": "http://suspicious-site.com",
  "is_phishing": true,
  "confidence": 0.92,
  "risk_level": "HIGH",
  "individual_predictions": {
    "XGBoost": {
      "prediction": 1,
      "phishing_probability": 0.94,
      "legitimate_probability": 0.06
    },
    "Random Forest": {
      "prediction": 1,
      "phishing_probability": 0.91,
      "legitimate_probability": 0.09
    },
    "Logistic Regression": {
      "prediction": 1,
      "phishing_probability": 0.88,
      "legitimate_probability": 0.12
    }
  },
  "timestamp": "2025-11-08T12:00:00.000000"
}
```

#### `POST /predict/batch`
Batch prediction for multiple URLs (max 100)

**Request:**
```json
{
  "urls": [
    "https://google.com",
    "http://suspicious-site.com",
    "https://github.com"
  ]
}
```

**Response:**
```json
{
  "total_urls": 3,
  "results": [...],
  "timestamp": "2025-11-08T12:00:00.000000"
}
```

### Analysis Endpoints

#### `POST /analyze`
Detailed URL analysis with feature extraction

**Response:**
```json
{
  "url": "http://example.com",
  "prediction": {
    "is_phishing": false,
    "confidence": 0.95,
    "risk_level": "LOW",
    "phishing_probability": 0.05,
    "legitimate_probability": 0.95
  },
  "features": {
    "total_features": 48,
    "all_features": {...},
    "top_10_important_features": [
      {
        "name": "PctExtNullSelfRedirectHyperlinksRT",
        "value": 0.0,
        "importance": 0.627
      },
      ...
    ]
  },
  "timestamp": "2025-11-08T12:00:00.000000"
}
```

## Usage Examples

### Python

```python
import requests

# Single URL prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"url": "https://suspicious-login.com"}
)
result = response.json()

if result["is_phishing"]:
    print(f"⚠️ PHISHING DETECTED!")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Risk Level: {result['risk_level']}")
else:
    print(f"✅ URL appears legitimate")
```

### cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Ensemble prediction
curl -X POST "http://localhost:8000/predict/ensemble" \
  -H "Content-Type: application/json" \
  -d '{"url": "http://suspicious-site.com"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://google.com",
      "http://phishing-site.com"
    ]
  }'
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function checkURL(url) {
  try {
    const response = await axios.post('http://localhost:8000/predict', {
      url: url
    });

    const result = response.data;
    console.log(`URL: ${result.url}`);
    console.log(`Phishing: ${result.is_phishing}`);
    console.log(`Confidence: ${result.confidence}`);
    console.log(`Risk Level: ${result.risk_level}`);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

checkURL('https://suspicious-login-verify.com');
```

## Testing

Run the comprehensive test suite:

```bash
# Make sure API is running first
python test_api.py
```

This tests all endpoints including:
- Health check
- Model listing
- Single predictions
- Model-specific predictions
- Ensemble predictions
- Batch predictions
- Detailed analysis

## Risk Levels

The API categorizes phishing risk into three levels:

- **LOW** (Confidence < 0.5): URL appears legitimate
- **MEDIUM** (0.5 ≤ Confidence < 0.8): Moderate risk, use caution
- **HIGH** (Confidence ≥ 0.8): High probability of phishing

## Performance

- **Response Time**: < 100ms for single URL
- **Throughput**: ~1000 requests/second (single instance)
- **Batch Processing**: Up to 100 URLs per request
- **Model Accuracy**: 98.65% (XGBoost)

## Model Information

| Model | Accuracy | F1-Score | Use Case |
|-------|----------|----------|----------|
| **XGBoost** | 98.65% | 0.9865 | Default, best overall performance |
| **Random Forest** | 98.55% | 0.9855 | Good for feature importance |
| **Logistic Regression** | 95.20% | 0.9524 | Fast, interpretable |
| **Ensemble** | ~98.8% | ~0.988 | Most robust, combines all models |

## Architecture

```
┌─────────────────┐
│   Client App    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │  ← REST API
│   Application   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature         │  ← Extract 48 features from URL
│ Extractor       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Models      │  ← XGBoost, RF, LR
│  (XGBoost/RF/LR)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prediction    │  ← Phishing detection result
└─────────────────┘
```

## Environment Variables

Create a `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info

# Model Configuration
MODEL_PATH=../models
DEFAULT_MODEL=xgboost

# Performance
MAX_BATCH_SIZE=100
CACHE_ENABLED=true
```

## Deployment

### Production Considerations

1. **Security**:
   - Add API authentication (JWT tokens)
   - Configure CORS for specific domains
   - Enable HTTPS
   - Rate limiting

2. **Performance**:
   - Use multiple workers: `uvicorn main:app --workers 4`
   - Enable caching for repeated URLs
   - Load balancing with nginx

3. **Monitoring**:
   - Add logging and metrics (Prometheus)
   - Error tracking (Sentry)
   - Performance monitoring (APM)

### Example nginx Configuration

```nginx
server {
    listen 80;
    server_name api.phishing-detector.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Development

### Project Structure

```
api/
├── main.py                 # FastAPI application
├── feature_extractor.py    # Feature extraction module
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── test_api.py           # Test suite
└── README.md             # This file
```

### Adding New Features

1. Update `feature_extractor.py` to extract new features
2. Retrain models with new features
3. Update model files in `../models/`
4. Test with `test_api.py`

## Troubleshooting

### API won't start

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Use a different port
uvicorn main:app --port 8001
```

### Models not loading

```bash
# Verify model files exist
ls -la ../models/

# Check model paths in main.py
```

### Docker issues

```bash
# Rebuild containers
docker-compose build --no-cache
docker-compose up -d

# View logs
docker-compose logs -f
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is for educational and research purposes.

## Support

- Documentation: http://localhost:8000/docs
- Issues: GitHub Issues
- Email: support@example.com

## Changelog

### v1.0.0 (2025-11-08)
- Initial release
- XGBoost, Random Forest, Logistic Regression models
- Ensemble predictions
- Batch processing
- Docker support
- Comprehensive API documentation
