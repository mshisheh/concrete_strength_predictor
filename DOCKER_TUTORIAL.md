# 🚀 Docker and CI/CD Tutorial

## 📚 What You've Just Learned

Congratulations! You now have a production-ready Docker and CI/CD setup. Let me explain what each component does:

### 🐳 Docker Components

1. **`Dockerfile`** - Recipe to build your app into a container
   - Uses Python 3.12 slim image for smaller size
   - Installs dependencies from requirements.txt
   - Creates non-root user for security
   - Includes health checks

2. **`docker-compose.yml`** - Orchestrates multiple services
   - Main app service on port 8501
   - Optional MLflow server on port 5000
   - Volume mounts for data persistence
   - Networks for service communication

3. **`.dockerignore`** - Excludes files from Docker build
   - Reduces image size
   - Improves security

### 🔄 CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

Your pipeline includes:
1. **Testing** - Runs on Python 3.11 & 3.12
2. **Security** - Safety and Bandit checks
3. **Docker** - Builds and pushes to GitHub Container Registry
4. **Deployment** - Ready for cloud deployment

## 🎯 Step-by-Step Commands You Can Use

### Local Development

```powershell
# 1. Start with Docker Compose (Recommended)
docker-compose up -d

# 2. View logs
docker-compose logs -f

# 3. Stop services
docker-compose down

# 4. Start with MLflow UI
docker-compose --profile mlflow up -d
```

### Manual Docker Commands

```powershell
# Build image
docker build -t concrete-predictor .

# Run container
docker run -d -p 8501:8501 --name concrete-app concrete-predictor

# View logs
docker logs concrete-app

# Stop container
docker stop concrete-app
docker rm concrete-app
```

### Using the Deployment Scripts

```powershell
# Windows
.\deployment\deploy-local.bat

# Check status
docker ps
```

## 🌐 Setting Up GitHub CI/CD

### For Your New GitHub Repository:

1. **Create new repository** on GitHub
2. **Push your code**:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit with Docker and CI/CD"
   git branch -M main
   git remote add origin https://github.com/yourusername/concrete-predictor.git
   git push -u origin main
   ```

3. **Enable GitHub Actions**:
   - Go to repository → Actions tab
   - GitHub will automatically detect your workflow
   - First run will create container registry

4. **Set up secrets** (if needed):
   - Go to Settings → Secrets and variables → Actions
   - Add any deployment secrets

### What Happens When You Push Code:

1. **On Push to `main`**:
   - ✅ Runs tests on Python 3.11 & 3.12
   - 🔍 Security scanning
   - 🐳 Builds Docker image
   - 📦 Pushes to GitHub Container Registry
   - 🚀 Ready for deployment

2. **On Pull Request**:
   - ✅ Runs tests only
   - 🔍 Security scanning

## 🔧 Troubleshooting

### Common Issues:

1. **Port already in use**:
   ```powershell
   docker stop $(docker ps -q)  # Stop all containers
   ```

2. **Permission denied**:
   - Make sure Docker Desktop is running as administrator

3. **Build fails**:
   ```powershell
   docker system prune -f  # Clean up
   docker build --no-cache -t concrete-predictor .
   ```

4. **Container won't start**:
   ```powershell
   docker logs concrete-predictor-app  # Check logs
   ```

## 🌟 Next Steps

### For Production Deployment:

1. **Cloud Platforms**:
   - **Google Cloud Run** (easiest)
   - **AWS ECS/Fargate**
   - **Azure Container Instances**
   - **DigitalOcean App Platform**

2. **Domain Setup**:
   - Purchase domain
   - Set up SSL certificate
   - Configure DNS

3. **Monitoring**:
   - Set up logging
   - Add monitoring dashboards
   - Configure alerts

### Advanced Features to Add:

1. **Database Integration**:
   - PostgreSQL for storing predictions
   - Redis for caching

2. **API Endpoints**:
   - REST API with FastAPI
   - Authentication

3. **Scaling**:
   - Kubernetes deployment
   - Load balancing

## 🏆 What You've Achieved

✅ Production-ready Docker setup
✅ Automated CI/CD pipeline
✅ Security scanning
✅ Multi-platform testing
✅ Container registry integration
✅ Local development environment
✅ Deployment scripts
✅ Health checks and monitoring

You're now ready to deploy a professional ML application! 🎉

## 📞 Quick Reference

- **App URL**: http://localhost:8501
- **MLflow URL**: http://localhost:5000 (with --profile mlflow)
- **Health Check**: http://localhost:8501/_stcore/health
- **Logs**: `docker-compose logs -f`
- **Stop**: `docker-compose down`
