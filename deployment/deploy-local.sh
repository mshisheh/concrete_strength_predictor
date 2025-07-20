#!/bin/bash
# Local Docker deployment script

set -e

echo "🐳 Building and running Concrete Strength Predictor locally with Docker"

# Build the Docker image
echo "Building Docker image..."
docker build -t concrete-predictor:local .

# Stop any existing container
echo "Stopping existing containers..."
docker stop concrete-predictor-local 2>/dev/null || true
docker rm concrete-predictor-local 2>/dev/null || true

# Run the container
echo "Starting new container..."
docker run -d \
  --name concrete-predictor-local \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  -e ENVIRONMENT=development \
  concrete-predictor:local

# Wait for the app to start
echo "Waiting for app to start..."
sleep 10

# Check if the app is running
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ App is running successfully!"
    echo "🌐 Access the app at: http://localhost:8501"
    echo "📊 View container logs: docker logs concrete-predictor-local"
    echo "🛑 Stop the app: docker stop concrete-predictor-local"
else
    echo "❌ App failed to start. Check logs:"
    docker logs concrete-predictor-local
    exit 1
fi
