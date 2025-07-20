"""
Additional test cases for extended functionality.

This module contains tests for the new data sources and configuration modules.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from concrete.data_sources import DataSourceManager
from config import Config
from environment import Environment


class TestDataSourceManager(unittest.TestCase):
    """Test cases for DataSourceManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = DataSourceManager(data_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of DataSourceManager."""
        self.assertEqual(self.manager.data_dir, Path(self.test_dir))
        self.assertTrue(self.manager.data_dir.exists())
    
    @patch('builtins.__import__')
    def test_download_from_kaggle_success(self, mock_import):
        """Test successful Kaggle download."""
        # Mock the kaggle module import
        mock_kaggle = MagicMock()
        mock_kaggle.api.dataset_download_files.return_value = None
        
        def side_effect(name, *args, **kwargs):
            if name == 'kaggle':
                return mock_kaggle
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = side_effect
        
        # Create a mock zip file
        zip_path = Path(self.test_dir) / "kaggle_concrete_data.zip"
        zip_path.touch()
        
        with patch.object(self.manager, '_extract_kaggle_data') as mock_extract:
            mock_extract.return_value = Path(self.test_dir) / "data.csv"
            result = self.manager.download_from_kaggle("test/dataset")
            self.assertIsNotNone(result)
    
    def test_validate_dataset_valid(self):
        """Test dataset validation with valid data."""
        # Create valid test data
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 150),
            'feature_2': np.random.normal(0, 1, 150),
            'feature_3': np.random.normal(0, 1, 150),
            'feature_4': np.random.normal(0, 1, 150),
            'feature_5': np.random.normal(0, 1, 150),
            'feature_6': np.random.normal(0, 1, 150),
            'feature_7': np.random.normal(0, 1, 150),
            'feature_8': np.random.normal(0, 1, 150),
            'target': np.random.normal(0, 1, 150)
        })
        
        test_file = Path(self.test_dir) / "valid_data.csv"
        data.to_csv(test_file, index=False)
        
        result = self.manager.validate_dataset(test_file)
        self.assertTrue(result)
    
    def test_validate_dataset_invalid(self):
        """Test dataset validation with invalid data."""
        # Create invalid test data (too few rows)
        data = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'target': [10, 20, 30]
        })
        
        test_file = Path(self.test_dir) / "invalid_data.csv"
        data.to_csv(test_file, index=False)
        
        result = self.manager.validate_dataset(test_file)
        self.assertFalse(result)
    
    def test_get_available_datasets(self):
        """Test getting available datasets."""
        # Create test CSV file
        data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        csv_file = Path(self.test_dir) / "test.csv"
        data.to_csv(csv_file, index=False)
        
        datasets = self.manager.get_available_datasets()
        self.assertIn("test", datasets)
        self.assertEqual(datasets["test"]["format"], "csv")
        self.assertEqual(datasets["test"]["shape"], (3, 2))


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def test_data_sources(self):
        """Test data source configuration."""
        self.assertIn('uci', Config.DATA_SOURCES)
        self.assertIn('kaggle', Config.DATA_SOURCES)
        self.assertIn('url', Config.DATA_SOURCES['uci'])
    
    def test_feature_info(self):
        """Test feature information configuration."""
        self.assertIn('cement', Config.FEATURE_INFO)
        self.assertIn('description', Config.FEATURE_INFO['cement'])
        self.assertIn('unit', Config.FEATURE_INFO['cement'])
        self.assertIn('typical_range', Config.FEATURE_INFO['cement'])
    
    def test_get_data_source_url(self):
        """Test getting data source URL."""
        uci_url = Config.get_data_source_url('uci')
        self.assertTrue(uci_url.startswith('http'))
        
        # Test invalid source
        invalid_url = Config.get_data_source_url('invalid')
        self.assertEqual(invalid_url, '')
    
    def test_get_feature_ranges(self):
        """Test getting feature ranges."""
        ranges = Config.get_feature_ranges()
        self.assertIn('cement', ranges)
        self.assertIsInstance(ranges['cement'], tuple)
        self.assertEqual(len(ranges['cement']), 2)
    
    def test_create_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the class attributes
            with patch.object(Config, 'DATA_DIR', Path(temp_dir) / 'data'):
                with patch.object(Config, 'MODELS_DIR', Path(temp_dir) / 'models'):
                    with patch.object(Config, 'PLOTS_DIR', Path(temp_dir) / 'plots'):
                        with patch.object(Config, 'MLRUNS_DIR', Path(temp_dir) / 'mlruns'):
                            Config.create_directories()
                            self.assertTrue((Path(temp_dir) / 'data').exists())
                            self.assertTrue((Path(temp_dir) / 'models').exists())
                            self.assertTrue((Path(temp_dir) / 'plots').exists())
                            self.assertTrue((Path(temp_dir) / 'mlruns').exists())


class TestEnvironment(unittest.TestCase):
    """Test cases for Environment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.env_file = Path(self.test_dir) / ".env"
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_load_env_file_exists(self):
        """Test loading environment file when it exists."""
        # Create test .env file
        env_content = """
TEST_VAR=test_value
ANOTHER_VAR=another_value
# This is a comment
EMPTY_VAR=
        """
        with open(self.env_file, 'w') as f:
            f.write(env_content)
        
        # Load environment
        env = Environment(env_file=str(self.env_file))
        
        # Check that variables were loaded
        import os
        self.assertEqual(os.getenv('TEST_VAR'), 'test_value')
        self.assertEqual(os.getenv('ANOTHER_VAR'), 'another_value')
    
    def test_load_env_file_not_exists(self):
        """Test loading environment file when it doesn't exist."""
        non_existent_file = self.test_dir + "/nonexistent.env"
        env = Environment(env_file=non_existent_file)
        # Should not raise an error
        self.assertIsNotNone(env)
    
    def test_get_mlflow_config(self):
        """Test getting MLflow configuration."""
        config = Environment.get_mlflow_config()
        self.assertIn('tracking_uri', config)
        self.assertIn('experiment_name', config)
        self.assertIn('artifact_location', config)
        self.assertIn('registry_uri', config)
    
    def test_get_database_config(self):
        """Test getting database configuration."""
        config = Environment.get_database_config()
        self.assertIn('host', config)
        self.assertIn('port', config)
        self.assertIn('name', config)
        self.assertIn('user', config)
        self.assertIn('password', config)
        self.assertIn('url', config)
    
    def test_get_kaggle_config(self):
        """Test getting Kaggle configuration."""
        config = Environment.get_kaggle_config()
        self.assertIn('username', config)
        self.assertIn('key', config)
    
    def test_get_deployment_config(self):
        """Test getting deployment configuration."""
        config = Environment.get_deployment_config()
        self.assertIn('environment', config)
        self.assertIn('debug', config)
        self.assertIn('port', config)
        self.assertIn('host', config)
        self.assertIn('log_level', config)
        
        # Test type conversions
        self.assertIsInstance(config['debug'], bool)
        self.assertIsInstance(config['port'], int)
    
    @patch.dict('os.environ', {'ENVIRONMENT': 'production'})
    def test_is_production(self):
        """Test production environment detection."""
        self.assertTrue(Environment.is_production())
        self.assertFalse(Environment.is_development())
    
    @patch.dict('os.environ', {'ENVIRONMENT': 'development'})
    def test_is_development(self):
        """Test development environment detection."""
        self.assertTrue(Environment.is_development())
        self.assertFalse(Environment.is_production())
    
    @patch('logging.basicConfig')
    def test_setup_logging(self, mock_logging):
        """Test logging setup."""
        Environment.setup_logging()
        mock_logging.assert_called_once()


if __name__ == '__main__':
    unittest.main()
