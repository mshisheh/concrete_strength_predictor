"""
Additional data source utilities for downloading datasets from various repositories.

This module extends the data loading capabilities to support multiple data sources
including Kaggle datasets and alternative UCI sources.
"""

import os
import logging
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import requests
from config import Config

logger = logging.getLogger(__name__)


class DataSourceManager:
    """Manages multiple data sources for the concrete strength dataset."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data source manager.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_from_kaggle(self, dataset_name: str, force_download: bool = False) -> Optional[Path]:
        """Download dataset from Kaggle using kaggle API.
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'sinamhd9/concrete-comprehensive-strength')
            force_download: Force re-download even if file exists
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            try:
                import kaggle
            except ImportError:
                logger.warning("Kaggle package not installed. Install with: poetry install --extras kaggle")
                return None
            
            output_path = self.data_dir / "kaggle_concrete_data.zip"
            
            if output_path.exists() and not force_download:
                logger.info(f"Kaggle data already exists at {output_path}")
                return self._extract_kaggle_data(output_path)
            
            logger.info(f"Downloading dataset from Kaggle: {dataset_name}")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(self.data_dir),
                unzip=False
            )
            
            # Find the downloaded zip file
            zip_files = list(self.data_dir.glob("*.zip"))
            if zip_files:
                zip_file = zip_files[0]
                zip_file.rename(output_path)
                return self._extract_kaggle_data(output_path)
            else:
                logger.error("No zip file found after Kaggle download")
                return None
                
        except ImportError:
            logger.warning("Kaggle package not installed. Install with: pip install kaggle")
            return None
        except Exception as e:
            logger.error(f"Failed to download from Kaggle: {e}")
            return None
    
    def _extract_kaggle_data(self, zip_path: Path) -> Optional[Path]:
        """Extract and return the concrete data from Kaggle zip file.
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            Path to extracted CSV file
        """
        try:
            extract_dir = self.data_dir / "kaggle_extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Look for CSV files
            csv_files = list(extract_dir.glob("*.csv"))
            if csv_files:
                csv_file = csv_files[0]
                # Copy to standard location
                standard_path = self.data_dir / "concrete_data_kaggle.csv"
                csv_file.rename(standard_path)
                logger.info(f"Kaggle data extracted to {standard_path}")
                return standard_path
            else:
                logger.error("No CSV file found in Kaggle download")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract Kaggle data: {e}")
            return None
    
    def download_from_alternative_uci(self) -> Optional[Path]:
        """Download from alternative UCI mirrors if main source fails.
        
        Returns:
            Path to downloaded file or None if failed
        """
        alternative_urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
            "https://raw.githubusercontent.com/selva86/datasets/master/Concrete_Data.csv"
        ]
        
        for url in alternative_urls:
            try:
                logger.info(f"Trying alternative UCI source: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Determine file extension
                if url.endswith('.xls'):
                    filename = "Concrete_Data_alt.xls"
                elif url.endswith('.csv'):
                    filename = "Concrete_Data_alt.csv"
                elif url.endswith('.zip'):
                    filename = "concrete_uci_alt.zip"
                else:
                    filename = "concrete_data_alt.txt"
                
                file_path = self.data_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded from alternative UCI source: {file_path}")
                
                # If it's a zip file, extract it
                if filename.endswith('.zip'):
                    return self._extract_uci_zip(file_path)
                
                return file_path
                
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                continue
        
        logger.error("All alternative UCI sources failed")
        return None
    
    def _extract_uci_zip(self, zip_path: Path) -> Optional[Path]:
        """Extract UCI zip file and return the data file.
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            Path to extracted data file
        """
        try:
            extract_dir = self.data_dir / "uci_extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Look for data files
            data_files = list(extract_dir.glob("*.xls")) + list(extract_dir.glob("*.csv"))
            if data_files:
                data_file = data_files[0]
                standard_path = self.data_dir / f"concrete_data_uci_alt.{data_file.suffix[1:]}"
                data_file.rename(standard_path)
                logger.info(f"UCI zip data extracted to {standard_path}")
                return standard_path
            else:
                logger.error("No data file found in UCI zip")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract UCI zip: {e}")
            return None
    
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available datasets.
        
        Returns:
            Dictionary with dataset information
        """
        datasets = {}
        
        # Check for existing files
        for file_path in self.data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                datasets[file_path.stem] = {
                    'path': file_path,
                    'format': 'csv',
                    'shape': df.shape,
                    'columns': list(df.columns)
                }
            except Exception:
                pass
        
        for file_path in self.data_dir.glob("*.xls*"):
            try:
                df = pd.read_excel(file_path)
                datasets[file_path.stem] = {
                    'path': file_path,
                    'format': 'excel',
                    'shape': df.shape,
                    'columns': list(df.columns)
                }
            except Exception:
                pass
        
        return datasets
    
    def validate_dataset(self, file_path: Path) -> bool:
        """Validate that a dataset has the expected structure for concrete data.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Check basic requirements
            if df.shape[0] < 100:  # Should have at least 100 samples
                return False
            
            if df.shape[1] < 8:  # Should have at least 8 features + target
                return False
            
            # Check for numeric data
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 8:
                return False
            
            logger.info(f"Dataset {file_path} validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate dataset {file_path}: {e}")
            return False


def setup_kaggle_credentials():
    """Helper function to set up Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if not kaggle_file.exists():
        print("Kaggle credentials not found. Please:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to API section and click 'Create New API Token'")
        print("3. Save the downloaded kaggle.json file to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    return True


if __name__ == "__main__":
    # Example usage
    manager = DataSourceManager()
    
    # Try to download from Kaggle
    if setup_kaggle_credentials():
        kaggle_path = manager.download_from_kaggle("sinamhd9/concrete-comprehensive-strength")
        if kaggle_path and manager.validate_dataset(kaggle_path):
            print(f"Successfully downloaded and validated Kaggle dataset: {kaggle_path}")
    
    # Try alternative UCI sources
    uci_path = manager.download_from_alternative_uci()
    if uci_path and manager.validate_dataset(uci_path):
        print(f"Successfully downloaded and validated UCI dataset: {uci_path}")
    
    # Show available datasets
    datasets = manager.get_available_datasets()
    print(f"\nAvailable datasets: {list(datasets.keys())}")
