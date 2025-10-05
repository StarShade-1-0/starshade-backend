# 🌟 StarShade Backend

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11.3-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-009688.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

**Advanced Machine Learning API for Exoplanet Detection and Classification**

🔗 **Live API**: [https://api-server-772504622905.asia-southeast1.run.app](https://api-server-772504622905.asia-southeast1.run.app)

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [API Documentation](#api-documentation) • [Deployment](#deployment)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
  - [Local Development](#local-development)
  - [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
  - [Health Check](#health-check)
  - [Dataset Endpoints](#dataset-endpoints)
  - [Prediction Endpoints](#prediction-endpoints)
- [Model Information](#model-information)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🌌 Overview

**StarShade Backend** is a high-performance RESTful API service designed for exoplanet detection and classification using state-of-the-art machine learning models. The system processes astronomical data from multiple space telescope missions (Kepler, K2, TESS) and provides real-time predictions using ensemble learning techniques.

### Key Capabilities

- 🔭 **Multi-Mission Support**: Processes data from Kepler, K2, TESS, and merged datasets
- 🤖 **Advanced ML Models**: Employs ensemble methods including Stacking, Voting, and Random Forest classifiers
- ⚡ **High Performance**: Built with FastAPI for low-latency, high-throughput predictions
- 📊 **Batch Processing**: Supports both single and batch prediction modes with CSV upload
- 🐳 **Container-Ready**: Fully Dockerized for seamless deployment
- 🔒 **Production-Ready**: CORS support, health checks, and environment-based configuration

---

## ✨ Features

### Machine Learning Models

- **K2 Mission**: Stacking Random Forest Classifier
- **Kepler Mission**: Soft Voting Classifier
- **TESS Mission**: Hard Voting Classifier
- **Merged Dataset**: Stacking Logistic Regression

### API Capabilities

- ✅ Single prediction with confidence scores
- ✅ Batch predictions via CSV upload
- ✅ Dataset download endpoints
- ✅ Probability distributions across classes
- ✅ Comprehensive error handling and validation
- ✅ CORS support for frontend integration
- ✅ Health monitoring endpoints

---

## 🏗️ Architecture

```
┌─────────────────┐
│   API Gateway   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ FastAPI │
    └────┬────┘
         │
    ┌────▼──────────────────────────┐
    │      Router Layer             │
    ├───────────┬───────────────────┤
    │           │                   │
┌───▼───┐  ┌───▼────┐  ┌──▼────┐  ┌▼────┐
│  K2   │  │ Kepler │  │ TESS  │  │Merge│
└───┬───┘  └───┬────┘  └───┬───┘  └─┬───┘
    │          │            │         │
┌───▼──────────▼────────────▼─────────▼───┐
│        ML Model Layer                   │
│  (Stacking, Voting, Ensemble Methods)   │
└─────────────────────────────────────────┘
```

### Request Flow

1. **Client Request** → API endpoint with astronomical features
2. **Validation** → Pydantic schema validation
3. **Preprocessing** → Imputation → Scaling
4. **Prediction** → Ensemble model inference
5. **Response** → JSON with prediction, confidence, and probabilities

---

## 🛠️ Technology Stack

### Core Framework
- **FastAPI** (v0.118.0) - Modern, fast web framework for building APIs
- **Uvicorn** - ASGI server for production deployment
- **Pydantic** (v2.11.10) - Data validation using Python type annotations

### Machine Learning
- **scikit-learn** (v1.7.2) - Core ML algorithms and preprocessing
- **XGBoost** - Gradient boosting framework
- **LightGBM** (v4.6.0) - High-performance gradient boosting
- **imbalanced-learn** (v0.14.0) - Handling imbalanced datasets
- **TensorFlow/Keras** (v3.11.3) - Deep learning capabilities

### Data Processing
- **NumPy** (v2.3.3) - Numerical computing
- **Pandas** (v2.3.3) - Data manipulation and analysis
- **SciPy** (v1.16.2) - Scientific computing

### Visualization & Monitoring
- **Matplotlib** (v3.10.6) - Plotting and visualization
- **Seaborn** - Statistical data visualization
- **Rich** (v14.1.0) - Terminal formatting

### DevOps
- **Docker** - Containerization
- **Python 3.11.3** - Runtime environment

---

## 🚀 Installation

### Prerequisites

- Python 3.11.3+
- pip (Python package manager)
- Docker (optional, for containerized deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/StarShade-1-0/starshade-backend.git
   cd starshade-backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the development server**
   ```bash
   ./run.sh
   # Or manually:
   python -m uvicorn app.main:app --reload
   ```

5. **Access the API**
   - Local API: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc
   - **Production API**: https://api-server-772504622905.asia-southeast1.run.app
   - **Production Docs**: https://api-server-772504622905.asia-southeast1.run.app/docs

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t starshade-backend .
   ```

2. **Run the container**
   ```bash
   docker run -d -p 8080:8080 \
     -e ENV=production \
     -e ALLOW_ORIGINS="https://yourdomain.com" \
     --name starshade-api \
     starshade-backend
   ```

3. **Verify deployment**
   ```bash
   curl http://localhost:8080/health
   ```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENV` | Environment mode (`dev`, `staging`, `production`) | `dev` | No |
| `ALLOW_ORIGINS` | Comma-separated list of allowed CORS origins | `*` | No |

### Example Configuration

**Development:**
```bash
export ENV=dev
export ALLOW_ORIGINS="http://localhost:3000,http://localhost:5173"
```

**Production:**
```bash
export ENV=production
export ALLOW_ORIGINS="https://starshade.app,https://www.starshade.app"
```

---

## 📚 API Documentation

### Health Check

#### `GET /health`

Check API status and environment configuration.

**Response:**
```json
{
  "status": "ok",
  "env": "dev"
}
```

---

### Dataset Endpoints

Download raw datasets used for training the models.

#### `GET /k2/dataset`
Download K2 mission dataset (CSV)

#### `GET /kepler/dataset`
Download Kepler mission dataset (CSV)

#### `GET /tess/dataset`
Download TESS mission dataset (CSV)

#### `GET /merged/dataset`
Download merged multi-mission dataset (CSV)

---

### Prediction Endpoints

#### K2 Mission - Stacking Random Forest

##### `POST /k2/stacking_rf/predict`

Single prediction using K2 mission features.

**Request Body:**
```json
{
  "pl_orbper": 3.5234,
  "pl_tranmid": 2455123.456,
  "pl_trandur": 2.345,
  "pl_rade": 1.23,
  "pl_radj": 0.109,
  "pl_radjerr1": 0.005,
  "pl_radjerr2": -0.005,
  "pl_ratror": 0.0123,
  "st_rad": 1.02,
  "st_raderr1": 0.03,
  "st_raderr2": -0.03,
  "sy_dist": 123.45,
  "sy_disterr1": 5.0,
  "sy_disterr2": -5.0,
  "sy_plx": 8.1,
  "sy_plxerr1": 0.2,
  "sy_plxerr2": -0.2
}
```

**Response:**
```json
{
  "prediction": "CONFIRMED",
  "confidence": 0.95,
  "class_probabilities": {
    "CONFIRMED": 0.95,
    "FALSE POSITIVE": 0.03,
    "CANDIDATE": 0.02
  }
}
```

##### `POST /k2/stacking_rf/predict_batch`

Batch prediction via CSV file upload.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: CSV file with feature columns

**Response:**
```json
{
  "total_predictions": 100,
  "successful": 98,
  "failed": 2,
  "warnings": ["Row 5: More than 80% of values are missing"],
  "errors": [],
  "results": [
    {
      "row_number": 1,
      "prediction": "CONFIRMED",
      "confidence": 0.92,
      "class_probabilities": {
        "CONFIRMED": 0.92,
        "FALSE POSITIVE": 0.05,
        "CANDIDATE": 0.03
      },
      "error": null
    }
  ]
}
```

#### Kepler Mission - Soft Voting

##### `POST /kepler/voting_soft/predict`
Single prediction for Kepler mission data.

##### `POST /kepler/voting_soft/predict_batch`
Batch prediction for Kepler mission data.

#### TESS Mission - Hard Voting

##### `POST /tess/voting_hard/predict`
Single prediction for TESS mission data.

##### `POST /tess/voting_hard/predict_batch`
Batch prediction for TESS mission data.

#### Merged Dataset - Stacking Logistic Regression

##### `POST /merged/stacking_logreg/predict`
Single prediction using merged multi-mission features.

##### `POST /merged/stacking_logreg/predict_batch`
Batch prediction using merged multi-mission features.

---

## 🧠 Model Information

### Feature Engineering

All models use carefully selected astronomical features:

- **Orbital Parameters**: `pl_orbper` (orbital period)
- **Transit Measurements**: `pl_tranmid` (mid-transit time), `pl_trandur` (transit duration)
- **Planetary Characteristics**: `pl_rade` (Earth radii), `pl_radj` (Jupiter radii), `pl_ratror` (planet-star radius ratio)
- **Stellar Properties**: `st_rad` (stellar radius)
- **System Metrics**: `sy_dist` (distance), `sy_plx` (parallax)
- **Error Margins**: Various `err1` and `err2` parameters for uncertainty quantification

### Preprocessing Pipeline

1. **Imputation**: Missing values handled using `SimpleImputer`
2. **Scaling**: Feature standardization with `StandardScaler`
3. **Encoding**: Label encoding for categorical outputs

### Model Types

| Mission | Model Type | Algorithm | Use Case |
|---------|------------|-----------|----------|
| K2 | Stacking | Random Forest meta-learner | High accuracy, interpretable |
| Kepler | Voting | Soft voting ensemble | Balanced predictions |
| TESS | Voting | Hard voting ensemble | Robust classification |
| Merged | Stacking | Logistic Regression meta-learner | Cross-mission generalization |

---

## 📁 Project Structure

```
starshade-backend/
├── app/
│   ├── __init__.py
│   ├── config.py              # Environment configuration
│   ├── main.py                # FastAPI application entry point
│   └── routers/
│       ├── __init__.py
│       ├── k2/                # K2 mission endpoints
│       │   ├── __init__.py
│       │   ├── dataset.csv
│       │   ├── schemas.py
│       │   └── stacking_rf/
│       │       ├── __init__.py
│       │       ├── model_impl.py
│       │       ├── schemas.py
│       │       └── stacking_rf.pkl
│       ├── kepler/            # Kepler mission endpoints
│       │   ├── __init__.py
│       │   ├── dataset.csv
│       │   ├── schemas.py
│       │   └── voting_soft/
│       │       └── ...
│       ├── tess/              # TESS mission endpoints
│       │   ├── __init__.py
│       │   ├── dataset.csv
│       │   ├── schemas.py
│       │   └── voting_hard/
│       │       └── ...
│       └── merged/            # Merged dataset endpoints
│           ├── __init__.py
│           ├── dataset.csv
│           ├── schemas.py
│           └── stacking_logreg/
│               └── ...
├── Dockerfile                 # Container configuration
├── requirements.txt           # Python dependencies
├── run.sh                     # Development server script
└── README.md                  # Project documentation
```

---

## 💻 Development

### Code Style

This project follows PEP 8 guidelines. Key conventions:

- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints where applicable
- Document functions with docstrings

### Adding a New Model

1. Create a new directory under `app/routers/{mission}/{model_name}/`
2. Implement `model_impl.py` with `predict()` and `batch_predict()` functions
3. Define response schemas in `schemas.py`
4. Register router in `__init__.py`
5. Update main router in `app/routers/{mission}/__init__.py`

### Running Tests

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Coverage report
pytest --cov=app tests/
```

---

## 🚢 Deployment

### Google Cloud Platform (GCP)

The project includes configuration for GCP deployment on the `dev/setup-gcp-deployment` branch.

**🌐 Production Deployment**: The API is currently deployed on GCP Cloud Run at:
- **Base URL**: https://api-server-772504622905.asia-southeast1.run.app
- **Interactive Docs**: https://api-server-772504622905.asia-southeast1.run.app/docs
- **Health Check**: https://api-server-772504622905.asia-southeast1.run.app/health

**Cloud Run Deployment:**

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/starshade-backend

# Deploy to Cloud Run
gcloud run deploy starshade-backend \
  --image gcr.io/PROJECT_ID/starshade-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ENV=production,ALLOW_ORIGINS="https://yourdomain.com"
```

### AWS Elastic Container Service (ECS)

```bash
# Build and tag
docker build -t starshade-backend .
docker tag starshade-backend:latest AWS_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/starshade-backend:latest

# Push to ECR
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin AWS_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
docker push AWS_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/starshade-backend:latest

# Deploy via ECS (configure task definition and service)
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: starshade-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: starshade-backend
  template:
    metadata:
      labels:
        app: starshade-backend
    spec:
      containers:
      - name: starshade-backend
        image: starshade-backend:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENV
          value: "production"
        - name: ALLOW_ORIGINS
          value: "https://yourdomain.com"
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Write clean, documented code
- Add tests for new features
- Update documentation as needed
- Follow existing code style
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**StarShade Team** - [StarShade-1-0](https://github.com/StarShade-1-0)

---

## 🙏 Acknowledgments

- NASA Exoplanet Archive for providing the astronomical datasets
- Kepler, K2, and TESS missions for groundbreaking exoplanet discoveries
- The open-source community for excellent ML libraries and tools

---

## 📞 Support

For issues, questions, or suggestions:

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/StarShade-1-0/starshade-backend/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/StarShade-1-0/starshade-backend/discussions)
- 📧 **Email**: support@starshade.app

---

<div align="center">

**Made with ❤️ by the StarShade Team**

⭐ Star us on GitHub — it helps!

[Production API](https://api-server-772504622905.asia-southeast1.run.app) • [API Documentation](https://api-server-772504622905.asia-southeast1.run.app/docs) • [GitHub](https://github.com/StarShade-1-0/starshade-backend)

</div>
