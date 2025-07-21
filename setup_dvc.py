#!/usr/bin/env python3
"""
DVC Integration Setup for Concrete Strength Project

This script will set up DVC (Data Version Control) for data versioning
to complement the MLflow model versioning already in place.
"""

import subprocess
import sys
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DVCSetup:
    """Handle DVC setup and configuration."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.dvc_config_file = self.project_root / "dvc_config.yml"
    
    def check_dvc_installed(self) -> bool:
        """Check if DVC is installed."""
        try:
            result = subprocess.run(["dvc", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"DVC is installed: {result.stdout.strip()}")
                return True
            else:
                logger.warning("DVC is not installed")
                return False
        except FileNotFoundError:
            logger.warning("DVC is not installed")
            return False
    
    def install_dvc(self):
        """Install DVC using pip."""
        try:
            logger.info("Installing DVC...")
            subprocess.run([sys.executable, "-m", "pip", "install", "dvc[s3]"], check=True)
            logger.info("DVC installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install DVC: {e}")
            raise
    
    def initialize_dvc(self):
        """Initialize DVC in the project."""
        try:
            logger.info("Initializing DVC...")
            subprocess.run(["dvc", "init"], cwd=self.project_root, check=True)
            logger.info("DVC initialized successfully")
        except subprocess.CalledProcessError as e:
            if "already exists" in str(e):
                logger.info("DVC already initialized")
            else:
                logger.error(f"Failed to initialize DVC: {e}")
                raise
    
    def setup_data_tracking(self):
        """Set up data tracking with DVC."""
        data_files = [
            "data/Concrete_Data.xls",
            "data/concrete_processed.csv"
        ]
        
        for data_file in data_files:
            file_path = self.project_root / data_file
            if file_path.exists():
                try:
                    logger.info(f"Adding {data_file} to DVC tracking...")
                    subprocess.run(["dvc", "add", str(file_path)], cwd=self.project_root, check=True)
                    logger.info(f"{data_file} added to DVC tracking")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to add {data_file} to DVC: {e}")
            else:
                logger.warning(f"Data file {data_file} not found")
    
    def create_dvc_pipeline(self):
        """Create a DVC pipeline for data processing."""
        pipeline_yaml = {
            'stages': {
                'data_preprocessing': {
                    'cmd': 'python -c "from concrete.data_loader import ConcreteDataLoader; loader = ConcreteDataLoader(); df = loader.load_raw_data(); clean_df = loader.clean_data(df); clean_df.to_csv(\'data/concrete_processed.csv\', index=False)"',
                    'deps': ['data/Concrete_Data.xls', 'concrete/data_loader.py'],
                    'outs': ['data/concrete_processed.csv']
                },
                'data_splitting': {
                    'cmd': 'python -c "from concrete.data_loader import ConcreteDataLoader; loader = ConcreteDataLoader(); loader.load_and_prepare_data()"',
                    'deps': ['data/concrete_processed.csv', 'concrete/data_loader.py'],
                    'outs': ['data/splits/']
                }
            }
        }
        
        pipeline_file = self.project_root / "dvc.yaml"
        with open(pipeline_file, 'w') as f:
            yaml.dump(pipeline_yaml, f, default_flow_style=False)
        
        logger.info("DVC pipeline created")
    
    def setup_remote_storage(self, remote_type: str = "local", remote_path: str = "dvc_remote"):
        """Set up DVC remote storage."""
        try:
            logger.info(f"Setting up {remote_type} remote storage...")
            
            if remote_type == "local":
                remote_path = self.project_root / remote_path
                remote_path.mkdir(exist_ok=True)
                subprocess.run([
                    "dvc", "remote", "add", "-d", "storage", str(remote_path)
                ], cwd=self.project_root, check=True)
            
            logger.info("Remote storage configured")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set up remote storage: {e}")
    
    def run_setup(self):
        """Run the complete DVC setup."""
        logger.info("Starting DVC setup...")
        
        # Check if DVC is installed, install if not
        if not self.check_dvc_installed():
            self.install_dvc()
        
        # Initialize DVC
        self.initialize_dvc()
        
        # Set up data tracking
        self.setup_data_tracking()
        
        # Create DVC pipeline
        self.create_dvc_pipeline()
        
        # Set up remote storage
        self.setup_remote_storage()
        
        logger.info("DVC setup completed!")
        
        # Update config to mark DVC as enabled
        if self.dvc_config_file.exists():
            with open(self.dvc_config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            config['data_versioning']['enabled'] = True
            
            with open(self.dvc_config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

def main():
    """Main function to set up DVC."""
    setup = DVCSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
