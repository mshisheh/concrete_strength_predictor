# 🏗️ Concrete Strength Predictor

A professional machine learning project for predicting concrete compressive strength using the UCI Concrete dataset. This project demonstrates end-to-end MLOps practices including data preprocessing, model training, evaluation, deployment, and monitoring.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![MLflow](https://img.shields.io/badge/mlflow-v2.8+-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange.svg)

## 🎯 Overview

This project predicts concrete compressive strength based on its constituent materials and age using various machine learning algorithms. It includes a complete MLOps pipeline with experiment tracking, model versioning, and deployment capabilities.

## ✨ Features

- **Data Processing**: Automated data loading, cleaning, and preprocessing
- **ML Models**: Multiple algorithms (Linear, Tree-based, Neural Networks)
- **Experiment Tracking**: MLflow integration for experiment management
- **Model Interpretation**: SHAP explanations for model transparency
- **Web Interface**: Interactive Streamlit dashboard
- **Containerization**: Docker support for easy deployment
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Comprehensive Testing**: Unit tests with pytest and coverage reporting

## 📊 Dataset

The project uses the [UCI Concrete Compressive Strength Dataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength):

- **Features**: 8 input variables (cement, water, aggregates, etc.)
- **Target**: Compressive strength (MPa)
- **Samples**: ~1000 concrete test records
- **Type**: Regression problem

### Features Description

| Feature | Description | Unit |
|---------|-------------|------|
| Cement | Cement content | kg/m³ |
| Blast Furnace Slag | Blast furnace slag content | kg/m³ |
| Fly Ash | Fly ash content | kg/m³ |
| Water | Water content | kg/m³ |
| Superplasticizer | Superplasticizer content | kg/m³ |
| Coarse Aggregate | Coarse aggregate content | kg/m³ |
| Fine Aggregate | Fine aggregate content | kg/m³ |
| Age | Age of concrete | days |

## 🏗️ Project Structure

```
concrete_strength_predictor/
├── concrete/                 # Source code package
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── trainer.py           # Model training and hyperparameter tuning
│   ├── evaluator.py         # Model evaluation and metrics
│   ├── visualizer.py        # Data and model visualization
│   └── app.py              # Streamlit web application
├── tests/                   # Unit tests
│   └── test_data_loader.py
├── data/                    # Raw and processed data
├── models/                  # Trained model artifacts
├── mlruns/                  # MLflow experiment tracking
├── .github/workflows/       # CI/CD pipeline
│   └── ci.yml
├── pyproject.toml          # Poetry dependencies and configuration
├── Dockerfile              # Container configuration
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- Docker (optional, for containerization)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/concrete-strength-predictor.git
   cd concrete-strength-predictor
   ```

2. **Install dependencies with Poetry**
   ```bash
   pip install poetry
   poetry install
   ```

3. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

### Usage

#### 1. Data Preparation
```bash
python -m concrete.data_loader
```

#### 2. Train Models
```bash
python -m concrete.trainer
```

#### 3. Evaluate Models
```bash
python -m concrete.evaluator
```

#### 4. Launch Web Application
```bash
streamlit run concrete/app.py
```

Visit `http://localhost:8501` to access the interactive dashboard.

## 🐳 Docker Deployment

### Build the image
```bash
docker build -t concrete-strength-predictor .
```

### Run the container
```bash
docker run -p 8501:8501 concrete-strength-predictor
```

## 🚀 Production Deployment

### Quick Start (Docker Hub)
```bash
# Pull and run the production-ready image
docker pull mshisheh/concrete-strength-predictor:latest
docker run -d -p 8501:8501 --name concrete-predictor mshisheh/concrete-strength-predictor:latest

# Access the application at http://localhost:8501
```

### Production Platforms
- **AWS**: ECS, Fargate, App Runner
- **Google Cloud**: Cloud Run, GKE
- **Azure**: Container Instances, App Service
- **Kubernetes**: Any cluster

📖 **Complete deployment guide**: [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md)

---

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=concrete --cov-report=html

# Run specific test file
poetry run pytest tests/test_data_loader.py -v
```

## 📈 Model Performance

The project includes multiple ML algorithms with automated hyperparameter tuning:

| Model | Validation RMSE | R² Score | Training Time |
|-------|----------------|----------|---------------|
| Random Forest | ~6.5 MPa | ~0.85 | ~30s |
| Gradient Boosting | ~6.8 MPa | ~0.84 | ~45s |
| Ridge Regression | ~7.2 MPa | ~0.81 | ~5s |
| Neural Network | ~7.0 MPa | ~0.82 | ~60s |

*Results may vary based on data splits and hyperparameter optimization.*

## 🔬 MLflow Experiment Tracking

The project uses MLflow for experiment management:

```bash
# Start MLflow UI
mlflow ui

# View experiments at http://localhost:5000
```

Features tracked:
- Model parameters and hyperparameters
- Performance metrics (RMSE, R², MAE)
- Model artifacts and serialized models
- Data versioning and lineage

## 📊 Web Application Features

The Streamlit dashboard provides:

- **Data Exploration**: Interactive visualizations and statistics
- **Model Training**: Train and compare multiple algorithms
- **Model Evaluation**: Comprehensive performance analysis
- **Predictions**: Single and batch prediction capabilities
- **Model Interpretation**: SHAP explanations

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline includes:

- **Testing**: Automated testing across Python versions
- **Code Quality**: Linting, formatting, and type checking
- **Security**: Dependency vulnerability scanning
- **Docker**: Container building and testing
- **Deployment**: Automated deployment to staging/production
- **Monitoring**: Model performance monitoring

## 🛠️ Development

### Code Quality Tools

The project uses several tools to maintain code quality:

```bash
# Code formatting
poetry run black concrete/ tests/

# Import sorting
poetry run isort concrete/ tests/

# Linting
poetry run flake8 concrete/

# Type checking
poetry run mypy concrete/
```

### Pre-commit Hooks

Install pre-commit hooks for automated code quality checks:

```bash
poetry run pre-commit install
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write comprehensive tests for new features
- Follow PEP 8 style guidelines
- Update documentation for API changes
- Ensure all CI/CD checks pass

## 📝 API Documentation

### Core Classes

#### `ConcreteDataLoader`
Handles data loading, cleaning, and preprocessing.

```python
from concrete.data_loader import ConcreteDataLoader

loader = ConcreteDataLoader()
X_train, X_val, X_test, y_train, y_val, y_test = loader.load_and_prepare_data()
```

#### `ConcreteTrainer`
Manages model training and hyperparameter optimization.

```python
from concrete.trainer import ConcreteTrainer

trainer = ConcreteTrainer()
results = trainer.train_all_models(X_train, y_train, X_val, y_val)
```

#### `ConcreteEvaluator`
Provides comprehensive model evaluation capabilities.

```python
from concrete.evaluator import ConcreteEvaluator

evaluator = ConcreteEvaluator()
metrics = evaluator.evaluate_model(model, X_test, y_test)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `./mlruns` |
| `DATA_DIR` | Data directory path | `./data` |
| `MODELS_DIR` | Models directory path | `./models` |

### Model Configuration

Models can be configured in `concrete/trainer.py`:

```python
model_configs = {
    'random_forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None]
        }
    }
}
```

## 📚 Resources

- [UCI Concrete Dataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## 🏆 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Open source community for the amazing ML tools
- Contributors and maintainers of the project dependencies

## ⚠️ Disclaimer

This tool is for educational and research purposes. Always validate predictions with physical testing for critical applications.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎯 New Agenda

The following features are planned for future development to enhance the production capabilities and MLOps maturity of the system:

### 1. REST API with FastAPI

**Priority**: Highest  
**Story Points**: 21

Build a production-ready REST API to enable programmatic access and system integrations.

**Planned Features**:
- **POST** `/api/v1/predict` - Single prediction endpoint
- **POST** `/api/v1/predict/batch` - Batch predictions with async processing
- **GET** `/api/v1/models` - List all registered models from MLflow Registry
- **GET** `/api/v1/models/{model_name}/versions` - Get model version details
- **GET** `/health` - Health check endpoint for orchestration
- OpenAPI/Swagger documentation at `/docs`
- Pydantic validation for all inputs
- Error handling with proper HTTP status codes
- Integration with MLflow Model Registry for model loading

**Expected Outcomes**:
- Enable integration with external systems and microservices
- Support programmatic predictions from other applications
- Facilitate CI/CD integration testing
- Docker deployment with both API and Streamlit services

---

### 2. Production Monitoring Dashboard

**Priority**: Highest  
**Story Points**: 18

Implement comprehensive monitoring and observability for production model performance.

**Planned Components**:
- **Prometheus** metrics collection:
  - Request rate, latency, error rate (RED metrics)
  - Prediction distribution tracking
  - Model RMSE over time
  - Feature distribution statistics
- **Grafana** dashboards:
  - API Performance (RPS, latency percentiles, error rates)
  - Model Performance (RMSE drift, prediction accuracy)
  - System Health (CPU, memory, disk usage)
- **Alerting System** (Alertmanager):
  - High error rate alerts (>5%)
  - High latency alerts (>2s p95)
  - Model RMSE degradation alerts (>10% increase)
  - Email/Slack notifications
- **Structured Logging**:
  - JSON format logs with trace IDs
  - Centralized log aggregation
  - Request/response logging with sampling

**Expected Outcomes**:
- Real-time visibility into model performance
- Proactive detection of issues before users report them
- Performance degradation alerts
- Improved debugging capabilities

---

### 3. Data Drift Detection

**Priority**: High  
**Story Points**: 15

Automated detection of data drift to maintain model accuracy over time.

**Planned Features**:
- **Statistical Drift Tests**:
  - Kolmogorov-Smirnov (KS) test for feature distribution shifts
  - Population Stability Index (PSI) for stability monitoring
  - Prediction drift detection
  - Concept drift identification
- **Automated Monitoring**:
  - Scheduled daily drift checks via cron/Airflow
  - Compare production data vs. training reference data
  - Store drift metrics in database for trend analysis
- **Drift Visualization Dashboard**:
  - "Data Health" page in Streamlit app
  - Feature distribution comparisons
  - Drift score trends over time
  - Recommendations for retraining triggers
- **Integration**:
  - Alert system integration for significant drift
  - Automated retraining pipeline triggers
  - DVC activation for data versioning

**Expected Outcomes**:
- Prevent model degradation through early drift detection
- Automated retraining triggers when drift exceeds thresholds
- Maintain prediction quality over time
- Data quality monitoring

---

### 4. Automated Model Validation Pipeline

**Priority**: High  
**Story Points**: 13

Implement quality gates to prevent poor models from reaching production.

**Planned Features**:
- **Model Validator Module** (`concrete/model_validator.py`):
  - Performance thresholds (RMSE < 7.0 MPa, R² > 0.80)
  - Prediction bias checks
  - Inference time validation (< 100ms)
  - Overfitting detection (train vs. validation comparison)
  - Feature importance stability checks
- **MLflow Registry Integration**:
  - Auto-validate before model registration
  - Block registration if validation fails
  - Log validation results and reports to MLflow
  - Generate validation report PDFs
- **CI/CD Integration**:
  - Add model validation to GitHub Actions pipeline
  - Run validation on model commits
  - Block PRs if validation criteria not met
  - Validation badges in README

**Expected Outcomes**:
- Prevent deployment of under-performing models
- Enforce quality standards automatically
- Reduce production incidents
- Maintain consistent model quality

---

### 5. Batch Prediction Service

**Priority**: Medium  
**Story Points**: 8

Enable large-scale batch predictions with async processing.

**Planned Features**:
- **Async Job Queue**:
  - Celery + Redis for task management
  - Background job processing
  - Job status tracking and progress updates
- **Storage Integration**:
  - S3/Cloud Storage support for large input files
  - Efficient CSV/Parquet file processing
  - Result file storage and download links
- **Notification System**:
  - Email notifications on job completion
  - Webhook support for job status updates
  - Failed job retry logic
- **API Endpoints**:
  - **POST** `/api/v1/batch/submit` - Submit batch job
  - **GET** `/api/v1/batch/{job_id}/status` - Check job status
  - **GET** `/api/v1/batch/{job_id}/results` - Download results

**Expected Outcomes**:
- Process thousands of predictions efficiently
- Non-blocking async processing
- Support for enterprise-scale workloads
- Improved user experience for large datasets

---

### 📋 Implementation Roadmap

**Phase 1 (Q2 2026)**: REST API + Monitoring Dashboard  
**Phase 2 (Q3 2026)**: Drift Detection + Model Validation  
**Phase 3 (Q4 2026)**: Batch Prediction Service  

**Note**: DVC setup is already prepared. Run `python setup_dvc.py` to activate data versioning

