@echo off
REM Windows batch script for local Docker deployment

echo 🐳 Building and running Concrete Strength Predictor locally with Docker

REM Build the Docker image
echo Building Docker image...
docker build -t concrete-predictor:local .

REM Stop any existing container
echo Stopping existing containers...
docker stop concrete-predictor-local 2>nul
docker rm concrete-predictor-local 2>nul

REM Run the container
echo Starting new container...
docker run -d ^
  --name concrete-predictor-local ^
  -p 8501:8501 ^
  -v "%cd%/data:/app/data" ^
  -v "%cd%/models:/app/models" ^
  -v "%cd%/mlruns:/app/mlruns" ^
  -e ENVIRONMENT=development ^
  concrete-predictor:local

REM Wait for the app to start
echo Waiting for app to start...
timeout /t 10 /nobreak > nul

REM Check if the app is running
curl -f http://localhost:8501/_stcore/health > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ App is running successfully!
    echo 🌐 Access the app at: http://localhost:8501
    echo 📊 View container logs: docker logs concrete-predictor-local
    echo 🛑 Stop the app: docker stop concrete-predictor-local
) else (
    echo ❌ App failed to start. Check logs:
    docker logs concrete-predictor-local
    exit /b 1
)
