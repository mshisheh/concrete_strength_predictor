# PowerShell script for common Docker operations

param(
    [Parameter(Mandatory=$false)]
    [string]$Command = "help"
)

$ProjectName = "concrete-predictor"
$ImageName = "${ProjectName}:latest"
$ContainerName = "${ProjectName}-app"

function Show-Help {
    Write-Host "Concrete Predictor Docker Manager" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\docker-manager.ps1 <command>" -ForegroundColor Green
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  build   - Build Docker image"
    Write-Host "  run     - Run container (build if needed)"
    Write-Host "  stop    - Stop and remove container"
    Write-Host "  logs    - Show container logs"
    Write-Host "  clean   - Clean up containers and images"
    Write-Host "  deploy  - Deploy with docker-compose"
    Write-Host "  help    - Show this help"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Magenta
    Write-Host "  .\docker-manager.ps1 build"
    Write-Host "  .\docker-manager.ps1 run"
    Write-Host "  .\docker-manager.ps1 logs"
}

function Build-Image {
    Write-Host "Building Docker image..." -ForegroundColor Blue
    docker build -t $ImageName .
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
}

function Run-Container {
    Write-Host "Starting container..." -ForegroundColor Blue
    
    # Stop existing container if running
    docker stop $ContainerName 2>$null | Out-Null
    docker rm $ContainerName 2>$null | Out-Null
    
    # Check if image exists, build if not
    $imageExists = docker images $ImageName -q
    if (-not $imageExists) {
        Write-Host "Image not found, building..." -ForegroundColor Yellow
        Build-Image
    }
    
    # Run container
    docker run -d --name $ContainerName -p 8501:8501 -e ENVIRONMENT=development $ImageName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Container started successfully!" -ForegroundColor Green
        Write-Host "Access the app at: http://localhost:8501" -ForegroundColor Cyan
    } else {
        Write-Host "Failed to start container!" -ForegroundColor Red
        exit 1
    }
}

function Stop-Container {
    Write-Host "Stopping container..." -ForegroundColor Blue
    docker stop $ContainerName 2>$null | Out-Null
    docker rm $ContainerName 2>$null | Out-Null
    Write-Host "Container stopped and removed!" -ForegroundColor Green
}

function Show-Logs {
    Write-Host "Showing container logs..." -ForegroundColor Blue
    docker logs -f $ContainerName
}

function Clean-Up {
    Write-Host "Cleaning up Docker resources..." -ForegroundColor Blue
    
    # Stop and remove containers
    docker stop $ContainerName 2>$null | Out-Null
    docker rm $ContainerName 2>$null | Out-Null
    
    # Remove images
    docker rmi $ImageName 2>$null | Out-Null
    
    # Clean up dangling images and volumes
    docker system prune -f
    
    Write-Host "Cleanup completed!" -ForegroundColor Green
}

function Deploy-App {
    Write-Host "Deploying with Docker Compose..." -ForegroundColor Blue
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Deployment successful!" -ForegroundColor Green
        Write-Host "App: http://localhost:8501" -ForegroundColor Cyan
        Write-Host "Logs: docker-compose logs -f" -ForegroundColor Cyan
        Write-Host "Stop: docker-compose down" -ForegroundColor Cyan
    } else {
        Write-Host "Deployment failed!" -ForegroundColor Red
        exit 1
    }
}

# Main script logic
switch ($Command.ToLower()) {
    "build" { Build-Image }
    "run" { Run-Container }
    "stop" { Stop-Container }
    "logs" { Show-Logs }
    "clean" { Clean-Up }
    "deploy" { Deploy-App }
    "help" { Show-Help }
    default { Show-Help }
}
