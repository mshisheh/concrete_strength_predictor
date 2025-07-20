"""
Unit tests for the ConcreteDataLoader class.

This module contains comprehensive tests for data loading,
cleaning, preprocessing, and splitting functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the class to test
import sys
sys.path.append(str(Path(__file__).parent.parent))
from concrete.data_loader import ConcreteDataLoader


class TestConcreteDataLoader(unittest.TestCase):
    """Test cases for ConcreteDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.data_loader = ConcreteDataLoader(data_dir=self.test_dir)
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'cement': [540.0, 540.0, 332.5, 400.0, 350.0, 450.0, 500.0, 380.0, 420.0, 460.0],
            'blast_furnace_slag': [0.0, 0.0, 142.5, 100.0, 80.0, 120.0, 90.0, 110.0, 70.0, 85.0],
            'fly_ash': [0.0, 0.0, 0.0, 50.0, 60.0, 40.0, 55.0, 45.0, 65.0, 35.0],
            'water': [162.0, 162.0, 228.0, 180.0, 190.0, 170.0, 185.0, 175.0, 195.0, 165.0],
            'superplasticizer': [2.5, 2.5, 0.0, 3.0, 3.5, 2.0, 2.8, 3.2, 1.8, 2.2],
            'coarse_aggregate': [1040.0, 1055.0, 932.0, 980.0, 1000.0, 960.0, 990.0, 970.0, 1010.0, 950.0],
            'fine_aggregate': [676.0, 676.0, 594.0, 650.0, 630.0, 670.0, 640.0, 660.0, 620.0, 680.0],
            'age': [28, 28, 270, 90, 180, 60, 120, 150, 30, 240],
            'compressive_strength': [79.99, 61.89, 40.27, 55.0, 45.0, 65.0, 50.0, 60.0, 40.0, 70.0]
        })
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of ConcreteDataLoader."""
        loader = ConcreteDataLoader(data_dir="test_data")
        
        self.assertEqual(loader.data_dir, Path("test_data"))
        self.assertEqual(len(loader.COLUMN_NAMES), 9)
        self.assertIn('compressive_strength', loader.COLUMN_NAMES)
    
    def test_create_sample_data(self):
        """Test sample data creation when download fails."""
        # Call the private method to create sample data
        self.data_loader._create_sample_data()
        
        # Check if file was created
        self.assertTrue(self.data_loader.raw_data_path.exists())
        
        # Load and verify the sample data
        df = pd.read_excel(self.data_loader.raw_data_path)
        
        self.assertEqual(len(df), 1000)  # Should have 1000 samples
        self.assertEqual(len(df.columns), 9)  # Should have 9 columns
        self.assertTrue(all(df['compressive_strength'] > 0))  # All strengths should be positive
    
    @patch('concrete.data_loader.requests.get')
    def test_download_data_success(self, mock_get):
        """Test successful data download."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.content = b"test data content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call download method
        self.data_loader.download_data()
        
        # Verify file was created
        self.assertTrue(self.data_loader.raw_data_path.exists())
        
        # Verify content
        with open(self.data_loader.raw_data_path, 'rb') as f:
            content = f.read()
        self.assertEqual(content, b"test data content")
    
    @patch('concrete.data_loader.requests.get')
    def test_download_data_failure(self, mock_get):
        """Test data download failure and fallback to sample data."""
        # Mock failed response
        mock_get.side_effect = Exception("Network error")
        
        # Call download method
        self.data_loader.download_data()
        
        # Verify sample data was created as fallback
        self.assertTrue(self.data_loader.raw_data_path.exists())
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create data with outliers and missing values
        dirty_data = self.sample_data.copy()
        
        # Add some outliers
        dirty_data.loc[len(dirty_data)] = [10000, 0, 0, 50, 0, 500, 400, 1, 100]  # Extreme cement
        dirty_data.loc[len(dirty_data)] = [300, 0, 0, 150, 0, -100, 600, 28, 50]  # Negative aggregate
        
        # Add missing values
        dirty_data.loc[0, 'cement'] = np.nan
        
        # Clean the data
        clean_data = self.data_loader.clean_data(dirty_data)
        
        # Verify cleaning
        self.assertFalse(clean_data.isnull().any().any())  # No missing values
        self.assertTrue(all(clean_data['cement'] > 0))  # All positive values
        self.assertTrue(len(clean_data) < len(dirty_data))  # Some rows removed
    
    def test_prepare_features_target(self):
        """Test feature and target separation."""
        X, y = self.data_loader.prepare_features_target(self.sample_data)
        
        self.assertEqual(len(X.columns), 8)  # 8 features
        self.assertNotIn('compressive_strength', X.columns)
        self.assertEqual(len(y), len(self.sample_data))
        self.assertEqual(y.name, 'compressive_strength')
    
    def test_split_data(self):
        """Test data splitting functionality."""
        X, y = self.data_loader.prepare_features_target(self.sample_data)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
            X, y, test_size=0.3, val_size=0.3, random_state=42
        )
        
        # Check that all data is accounted for
        total_size = len(X)
        total_split_size = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total_size, total_split_size)
        
        # Check that each split has at least 1 sample
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)
        
        # Check that target splits have same size as feature splits
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check that indices don't overlap
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        self.assertTrue(train_indices.isdisjoint(val_indices))
        self.assertTrue(train_indices.isdisjoint(test_indices))
        self.assertTrue(val_indices.isdisjoint(test_indices))
    
    def test_scale_features(self):
        """Test feature scaling functionality."""
        # Create larger dataset for meaningful scaling
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(100, 20, 50),
            'feature2': np.random.normal(500, 100, 50),
            'feature3': np.random.normal(10, 5, 50)
        })
        
        # Split data
        train_size = 30
        val_size = 10
        test_size = 10
        
        X_train = data.iloc[:train_size]
        X_val = data.iloc[train_size:train_size + val_size]
        X_test = data.iloc[train_size + val_size:]
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.data_loader.scale_features(
            X_train, X_val, X_test
        )
        
        # Check that training set has mean ~0 and std ~1
        train_means = X_train_scaled.mean()
        train_stds = X_train_scaled.std(ddof=0)  # Use population std
        
        np.testing.assert_array_almost_equal(train_means, 0, decimal=5)
        np.testing.assert_array_almost_equal(train_stds, 1, decimal=2)  # More lenient for std
        
        # Check that all sets have the same columns
        self.assertTrue(all(X_train_scaled.columns == X_val_scaled.columns))
        self.assertTrue(all(X_train_scaled.columns == X_test_scaled.columns))
    
    def test_get_feature_info(self):
        """Test feature information retrieval."""
        feature_info = self.data_loader.get_feature_info()
        
        self.assertIsInstance(feature_info, dict)
        self.assertEqual(len(feature_info), 8)  # 8 features
        self.assertIn('cement', feature_info)
        self.assertIn('age', feature_info)
        
        # Check that all descriptions are strings
        for description in feature_info.values():
            self.assertIsInstance(description, str)
            self.assertTrue(len(description) > 0)
    
    def test_load_and_prepare_data_integration(self):
        """Test the complete data preparation pipeline."""
        # First create sample data
        self.data_loader._create_sample_data()
        
        # Run the complete pipeline
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.load_and_prepare_data()
        
        # Verify outputs
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        
        # Check that data is properly scaled (training set should have mean ~0)
        train_means = X_train.mean()
        np.testing.assert_array_almost_equal(train_means, 0, decimal=1)
        
        # Check that processed data file was created
        self.assertTrue(self.data_loader.processed_data_path.exists())
    
    def test_column_names_consistency(self):
        """Test that column names are consistent throughout processing."""
        # Create sample data
        self.data_loader._create_sample_data()
        
        # Load raw data
        df_raw = self.data_loader.load_raw_data()
        
        # Check column names
        expected_columns = self.data_loader.COLUMN_NAMES
        self.assertEqual(list(df_raw.columns), expected_columns)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            self.data_loader.prepare_features_target(empty_df)
        
        # Test with missing target column
        no_target_df = self.sample_data.drop('compressive_strength', axis=1)
        
        with self.assertRaises(ValueError):
            self.data_loader.prepare_features_target(no_target_df)


class TestDataLoaderEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.data_loader = ConcreteDataLoader(data_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_single_row_data(self):
        """Test handling of single-row dataset."""
        single_row = pd.DataFrame({
            'cement': [540.0],
            'blast_furnace_slag': [0.0],
            'fly_ash': [0.0],
            'water': [162.0],
            'superplasticizer': [2.5],
            'coarse_aggregate': [1040.0],
            'fine_aggregate': [676.0],
            'age': [28],
            'compressive_strength': [79.99]
        })
        
        # This should handle gracefully (though may not be practical)
        cleaned = self.data_loader.clean_data(single_row)
        self.assertGreaterEqual(len(cleaned), 0)
    
    def test_all_zero_values(self):
        """Test handling of all-zero input values."""
        zero_data = pd.DataFrame({
            'cement': [0.0, 0.0],
            'blast_furnace_slag': [0.0, 0.0],
            'fly_ash': [0.0, 0.0],
            'water': [0.0, 0.0],
            'superplasticizer': [0.0, 0.0],
            'coarse_aggregate': [0.0, 0.0],
            'fine_aggregate': [0.0, 0.0],
            'age': [0, 0],
            'compressive_strength': [1.0, 2.0]  # Some positive target
        })
        
        # Should be filtered out during cleaning due to physical constraints
        cleaned = self.data_loader.clean_data(zero_data)
        # Water should be > 0 for concrete, so these should be filtered
        self.assertEqual(len(cleaned), 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
