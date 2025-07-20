"""
Data loading and preprocessing module for concrete strength prediction.

This module handles downloading the UCI concrete dataset,
cleaning the data, and preparing it for machine learning.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcreteDataLoader:
    """Handles loading and preprocessing of concrete strength data."""
    
    # UCI Dataset URL
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    
    # Column names for the dataset
    COLUMN_NAMES = [
        "cement", "blast_furnace_slag", "fly_ash", "water",
        "superplasticizer", "coarse_aggregate", "fine_aggregate", 
        "age", "compressive_strength"
    ]
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_path = self.data_dir / "Concrete_Data.xls"
        self.processed_data_path = self.data_dir / "concrete_processed.csv"
        self.scaler = StandardScaler()
        
    def download_data(self) -> None:
        """Download the UCI concrete dataset if not already present."""
        if self.raw_data_path.exists():
            logger.info(f"Data already exists at {self.raw_data_path}")
            return
            
        logger.info(f"Downloading data from {self.DATA_URL}")
        try:
            response = requests.get(self.DATA_URL, timeout=30)
            response.raise_for_status()
            
            with open(self.raw_data_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Data downloaded successfully to {self.raw_data_path}")
            
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            # Create sample data for development if download fails
            self._create_sample_data()
    
    def _create_sample_data(self) -> None:
        """Create sample data for development purposes."""
        logger.info("Creating sample data for development")
        np.random.seed(42)
        
        # Generate realistic concrete data
        n_samples = 1000
        cement = np.random.normal(281, 104, n_samples)
        blast_furnace_slag = np.random.normal(73, 86, n_samples)
        fly_ash = np.random.normal(54, 64, n_samples)
        water = np.random.normal(181, 21, n_samples)
        superplasticizer = np.random.normal(6, 6, n_samples)
        coarse_aggregate = np.random.normal(972, 77, n_samples)
        fine_aggregate = np.random.normal(773, 80, n_samples)
        age = np.random.randint(1, 365, n_samples)
        
        # Generate target based on realistic relationships
        compressive_strength = (
            0.12 * cement + 
            0.08 * blast_furnace_slag + 
            0.06 * fly_ash -
            0.15 * water +
            0.5 * superplasticizer +
            0.02 * coarse_aggregate +
            0.02 * fine_aggregate +
            0.1 * np.log(age + 1) +
            np.random.normal(0, 5, n_samples)
        )
        
        # Ensure positive values
        cement = np.maximum(cement, 0)
        blast_furnace_slag = np.maximum(blast_furnace_slag, 0)
        fly_ash = np.maximum(fly_ash, 0)
        water = np.maximum(water, 100)
        superplasticizer = np.maximum(superplasticizer, 0)
        coarse_aggregate = np.maximum(coarse_aggregate, 500)
        fine_aggregate = np.maximum(fine_aggregate, 500)
        compressive_strength = np.maximum(compressive_strength, 5)
        
        sample_data = pd.DataFrame({
            'cement': cement,
            'blast_furnace_slag': blast_furnace_slag,
            'fly_ash': fly_ash,
            'water': water,
            'superplasticizer': superplasticizer,
            'coarse_aggregate': coarse_aggregate,
            'fine_aggregate': fine_aggregate,
            'age': age,
            'compressive_strength': compressive_strength
        })
        
        sample_data.to_excel(self.raw_data_path, index=False)
        logger.info(f"Sample data created with {len(sample_data)} records")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from file.
        
        Returns:
            Raw dataframe
        """
        if not self.raw_data_path.exists():
            self.download_data()
            
        try:
            df = pd.read_excel(self.raw_data_path)
            
            # Set proper column names if needed
            if len(df.columns) == len(self.COLUMN_NAMES):
                df.columns = self.COLUMN_NAMES
                
            logger.info(f"Loaded raw data with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Check for missing values
        missing_values = df_clean.isnull().sum()
        if missing_values.any():
            logger.warning(f"Found missing values: {missing_values}")
            df_clean = df_clean.dropna()
        
        # Remove outliers using IQR method
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(df_clean)
            df_clean = df_clean[
                (df_clean[column] >= lower_bound) & 
                (df_clean[column] <= upper_bound)
            ]
            outliers_removed = outliers_before - len(df_clean)
            
            if outliers_removed > 0:
                logger.info(f"Removed {outliers_removed} outliers from {column}")
        
        # Ensure positive values for physical quantities
        physical_columns = [col for col in df_clean.columns if col != 'compressive_strength']
        for col in physical_columns:
            df_clean = df_clean[df_clean[col] >= 0]
        
        # Ensure positive strength values
        df_clean = df_clean[df_clean['compressive_strength'] > 0]
        
        logger.info(f"Data cleaned. Final shape: {df_clean.shape}")
        return df_clean
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable.
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Tuple of (features, target)
        """
        target_column = 'compressive_strength'
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Prepared features with shape {X.shape} and target with shape {y.shape}")
        return X, y
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(
        self, 
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features  
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_val, X_test)
        """
        # Fit scaler on training data only
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Features scaled successfully")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def load_and_prepare_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Complete data loading and preparation pipeline.
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Load and clean data
        df_raw = self.load_raw_data()
        df_clean = self.clean_data(df_raw)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df_clean)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X, y, test_size, val_size, random_state
        )
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        # Save processed data
        processed_data = pd.concat([X_train_scaled, y_train], axis=1)
        processed_data.to_csv(self.processed_data_path, index=False)
        logger.info(f"Processed data saved to {self.processed_data_path}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def get_feature_info(self) -> dict:
        """Get information about the features.
        
        Returns:
            Dictionary with feature descriptions
        """
        return {
            'cement': 'Cement content (kg/m³)',
            'blast_furnace_slag': 'Blast furnace slag content (kg/m³)',
            'fly_ash': 'Fly ash content (kg/m³)', 
            'water': 'Water content (kg/m³)',
            'superplasticizer': 'Superplasticizer content (kg/m³)',
            'coarse_aggregate': 'Coarse aggregate content (kg/m³)',
            'fine_aggregate': 'Fine aggregate content (kg/m³)',
            'age': 'Age of concrete (days)'
        }


def main():
    """Main function for testing the data loader."""
    loader = ConcreteDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_and_prepare_data()
    
    print("Data loading completed successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Display basic statistics
    print("\nFeature statistics (training set):")
    print(X_train.describe())
    
    print("\nTarget statistics:")
    print(y_train.describe())


if __name__ == "__main__":
    main()
