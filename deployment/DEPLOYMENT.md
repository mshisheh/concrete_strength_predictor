# 🚀 Deployment Guide for Concrete Strength Predictor

This guide covers multiple deployment options for the concrete strength prediction application.

## 📋 Prerequisites

- Docker installed and running
- Git repository set up
- (Optional) Cloud platform account (AWS, GCP, Azure)

## 🐳 Local Docker Deployment

### Option 1: Using Docker Compose (Recommended)
```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Option 2: Using Deployment Scripts
```bash
# Linux/Mac
chmod +x deployment/deploy-local.sh
./deployment/deploy-local.sh

# Windows
deployment\deploy-local.bat
```

### Option 3: Manual Docker Commands
```bash
# Build the image
docker build -t concrete-predictor .

# Run the container
docker run -d -p 8501:8501 --name concrete-app concrete-predictor

# Access the app
open http://localhost:8501
```

## ☁️ Cloud Deployment Options

### 1. Google Cloud Run (Recommended for beginners)

#### Setup:
```bash
# Install Google Cloud SDK
# Set up authentication: gcloud auth login

# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
```

#### Deploy:
```bash
# Build and deploy in one command
gcloud run deploy concrete-predictor \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1

# Or use the configuration file
gcloud run services replace deployment/cloud-run.yml
```

### 2. AWS ECS/Fargate

#### Setup:
```bash
# Install AWS CLI and configure credentials
aws configure

# Create ECS cluster
aws ecs create-cluster --cluster-name concrete-predictor-cluster
```

#### Deploy:
```bash
# Build and push to ECR
aws ecr create-repository --repository-name concrete-predictor
docker build -t concrete-predictor .
docker tag concrete-predictor:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/concrete-predictor:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/concrete-predictor:latest

# Create task definition and service using AWS Console or CLI
```

### 3. Azure Container Instances

#### Setup:
```bash
# Install Azure CLI
az login

# Create resource group
az group create --name concrete-predictor-rg --location eastus
```

#### Deploy:
```bash
# Create container instance
az container create \
  --resource-group concrete-predictor-rg \
  --name concrete-predictor \
  --image your-registry/concrete-predictor:latest \
  --dns-name-label concrete-predictor-app \
  --ports 8501 \
  --memory 2 \
  --cpu 1
```

### 4. Kubernetes (Advanced)

#### Create deployment files:
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: concrete-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: concrete-predictor
  template:
    metadata:
      labels:
        app: concrete-predictor
    spec:
      containers:
      - name: concrete-predictor
        image: your-registry/concrete-predictor:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: concrete-predictor-service
spec:
  selector:
    app: concrete-predictor
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

#### Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
kubectl get services
```

## 🔧 GitHub Actions CI/CD Setup

### 1. Repository Setup

1. **Create a new GitHub repository** (use your other GitHub account)
2. **Push your code:**
```bash
git init
git add .
git commit -m "Initial commit: Concrete Strength Predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/concrete-strength-predictor.git
git push -u origin main
```

### 2. Configure Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:
- `DOCKER_USERNAME`: Your Docker Hub username (optional)
- `DOCKER_PASSWORD`: Your Docker Hub password (optional)
- `GCP_PROJECT_ID`: Your Google Cloud project ID (if using GCP)
- `GCP_SA_KEY`: Google Cloud service account JSON key (if using GCP)

### 3. Workflow Triggers

The CI/CD pipeline will automatically run on:
- **Push to main**: Runs tests, security checks, builds Docker image, and deploys
- **Push to develop**: Runs tests and builds staging image
- **Pull Requests**: Runs tests and security checks
- **Releases**: Builds and tags production images

### 4. Monitor Deployments

- Check the **Actions** tab in your GitHub repository
- View build logs and deployment status
- Monitor deployed applications through your cloud provider's console

## 🔍 Monitoring and Troubleshooting

### Health Checks
All deployments include health checks at: `http://your-app-url/_stcore/health`

### View Logs
```bash
# Docker
docker logs concrete-predictor

# Docker Compose
docker-compose logs -f

# Kubernetes
kubectl logs deployment/concrete-predictor

# Google Cloud Run
gcloud run logs tail --service=concrete-predictor
```

### Common Issues

1. **Port binding errors**: Ensure port 8501 is available
2. **Memory issues**: Increase container memory limits
3. **Import errors**: Verify all dependencies are in requirements.txt
4. **Data loading failures**: Check internet connectivity for UCI dataset download

## 🎯 Best Practices

1. **Environment Variables**: Use environment-specific configurations
2. **Secrets Management**: Never commit secrets to the repository
3. **Resource Limits**: Set appropriate CPU and memory limits
4. **Monitoring**: Set up application monitoring and alerts
5. **Backup**: Regular backup of models and data
6. **Security**: Keep dependencies updated and scan for vulnerabilities

## 🆘 Support

If you encounter issues:
1. Check the application logs
2. Verify all prerequisites are met
3. Review the troubleshooting section
4. Check GitHub Actions workflow status

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
