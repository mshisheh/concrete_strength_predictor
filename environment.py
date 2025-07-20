"""
Environment configuration and utilities.

This module handles environment variables, secrets management,
and environment-specific configurations.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Environment:
    """Environment configuration manager."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize environment configuration.
        
        Args:
            env_file: Path to .env file (optional)
        """
        self.env_file = env_file or ".env"
        self.load_env_file()
    
    def load_env_file(self) -> None:
        """Load environment variables from .env file if it exists."""
        env_path = Path(self.env_file)
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ.setdefault(key.strip(), value.strip())
                logger.info(f"Loaded environment variables from {env_path}")
            except Exception as e:
                logger.warning(f"Failed to load .env file: {e}")
    
    @staticmethod
    def get_mlflow_config() -> Dict[str, str]:
        """Get MLflow configuration from environment variables.
        
        Returns:
            Dictionary with MLflow configuration
        """
        return {
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'),
            'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', 'concrete_strength'),
            'artifact_location': os.getenv('MLFLOW_ARTIFACT_LOCATION', ''),
            'registry_uri': os.getenv('MLFLOW_REGISTRY_URI', ''),
        }
    
    @staticmethod
    def get_database_config() -> Dict[str, str]:
        """Get database configuration from environment variables.
        
        Returns:
            Dictionary with database configuration
        """
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'name': os.getenv('DB_NAME', 'concrete_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'url': os.getenv('DATABASE_URL', ''),
        }
    
    @staticmethod
    def get_kaggle_config() -> Dict[str, str]:
        """Get Kaggle API configuration from environment variables.
        
        Returns:
            Dictionary with Kaggle configuration
        """
        return {
            'username': os.getenv('KAGGLE_USERNAME', ''),
            'key': os.getenv('KAGGLE_KEY', ''),
        }
    
    @staticmethod
    def get_deployment_config() -> Dict[str, Any]:
        """Get deployment configuration from environment variables.
        
        Returns:
            Dictionary with deployment configuration
        """
        return {
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'debug': os.getenv('DEBUG', 'False').lower() == 'true',
            'port': int(os.getenv('PORT', '8501')),
            'host': os.getenv('HOST', '0.0.0.0'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        }
    
    @staticmethod
    def is_production() -> bool:
        """Check if running in production environment.
        
        Returns:
            True if production, False otherwise
        """
        return os.getenv('ENVIRONMENT', '').lower() == 'production'
    
    @staticmethod
    def is_development() -> bool:
        """Check if running in development environment.
        
        Returns:
            True if development, False otherwise
        """
        env = os.getenv('ENVIRONMENT', 'development').lower()
        return env in ['development', 'dev', '']
    
    @staticmethod
    def setup_logging() -> None:
        """Set up logging configuration based on environment."""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_format = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('app.log') if Environment.is_production() else logging.NullHandler()
            ]
        )


def create_env_template() -> None:
    """Create a template .env file with all possible configurations."""
    template_content = """# MLflow Configuration
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=concrete_strength
# MLFLOW_ARTIFACT_LOCATION=s3://my-bucket/artifacts
# MLFLOW_REGISTRY_URI=sqlite:///mlflow_registry.db

# Database Configuration (if using database storage)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=concrete_db
# DB_USER=postgres
# DB_PASSWORD=your_password
# DATABASE_URL=postgresql://user:password@localhost:5432/concrete_db

# Kaggle API Configuration
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_api_key

# Deployment Configuration
ENVIRONMENT=development
DEBUG=True
PORT=8501
HOST=0.0.0.0
LOG_LEVEL=INFO

# Model Configuration
RANDOM_STATE=42
TEST_SIZE=0.2
VAL_SIZE=0.2
CV_FOLDS=5

# Security (for production)
# SECRET_KEY=your_secret_key
# ALLOWED_HOSTS=yourdomain.com,localhost
"""
    
    env_path = Path('.env.template')
    with open(env_path, 'w') as f:
        f.write(template_content)
    
    print(f"Created .env template at {env_path}")
    print("Copy this to .env and fill in your values.")


if __name__ == "__main__":
    # Create environment template
    create_env_template()
    
    # Test environment loading
    env = Environment()
    
    print("MLflow config:", env.get_mlflow_config())
    print("Deployment config:", env.get_deployment_config())
    print("Is production:", env.is_production())
    print("Is development:", env.is_development())
