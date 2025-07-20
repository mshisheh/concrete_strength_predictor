"""
Configuration module for the concrete strength prediction project.

This module centralizes all configuration settings including
data sources, model parameters, and application settings.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for the concrete strength prediction project."""
    
    # Data configuration
    DATA_SOURCES = {
        'uci': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
            'filename': 'Concrete_Data.xls'
        },
        'kaggle': {
            'dataset': 'sinamhd9/concrete-comprehensive-strength',
            'filename': 'concrete_data.csv'
        }
    }
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    PLOTS_DIR = PROJECT_ROOT / "plots"
    MLRUNS_DIR = PROJECT_ROOT / "mlruns"
    
    # Model configuration
    MODEL_CONFIGS = {
        'random_state': 42,
        'test_size': 0.2,
        'val_size': 0.2,
        'cv_folds': 5,
        'n_jobs': -1
    }
    
    # MLflow configuration
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
    EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'concrete_strength')
    
    # Streamlit configuration
    STREAMLIT_CONFIG = {
        'page_title': "Concrete Strength Predictor",
        'page_icon': "🏗️",
        'layout': "wide",
        'initial_sidebar_state': "expanded"
    }
    
    # Feature information
    FEATURE_INFO = {
        'cement': {
            'description': 'Cement content',
            'unit': 'kg/m³',
            'min_value': 0,
            'max_value': 600,
            'typical_range': (100, 500)
        },
        'blast_furnace_slag': {
            'description': 'Blast furnace slag content',
            'unit': 'kg/m³',
            'min_value': 0,
            'max_value': 400,
            'typical_range': (0, 200)
        },
        'fly_ash': {
            'description': 'Fly ash content',
            'unit': 'kg/m³',
            'min_value': 0,
            'max_value': 300,
            'typical_range': (0, 150)
        },
        'water': {
            'description': 'Water content',
            'unit': 'kg/m³',
            'min_value': 100,
            'max_value': 300,
            'typical_range': (150, 250)
        },
        'superplasticizer': {
            'description': 'Superplasticizer content',
            'unit': 'kg/m³',
            'min_value': 0,
            'max_value': 50,
            'typical_range': (0, 20)
        },
        'coarse_aggregate': {
            'description': 'Coarse aggregate content',
            'unit': 'kg/m³',
            'min_value': 500,
            'max_value': 1500,
            'typical_range': (800, 1200)
        },
        'fine_aggregate': {
            'description': 'Fine aggregate content',
            'unit': 'kg/m³',
            'min_value': 400,
            'max_value': 1000,
            'typical_range': (600, 900)
        },
        'age': {
            'description': 'Age of concrete',
            'unit': 'days',
            'min_value': 1,
            'max_value': 365,
            'typical_range': (7, 90)
        }
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'handlers': ['console', 'file']
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.PLOTS_DIR, cls.MLRUNS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_data_source_url(cls, source: str = 'uci') -> str:
        """Get the URL for a specific data source.
        
        Args:
            source: Data source ('uci' or 'kaggle')
            
        Returns:
            URL string
        """
        return cls.DATA_SOURCES.get(source, {}).get('url', '')
    
    @classmethod
    def get_feature_ranges(cls) -> Dict[str, tuple]:
        """Get typical ranges for all features.
        
        Returns:
            Dictionary of feature name to (min, max) tuples
        """
        return {name: info['typical_range'] for name, info in cls.FEATURE_INFO.items()}
