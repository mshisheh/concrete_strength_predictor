"""
Model training module for concrete strength prediction.

This module handles training multiple ML models, hyperparameter tuning,
and model selection using MLflow for experiment tracking.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from concrete.data_loader import ConcreteDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcreteTrainer:
    """Handles training and evaluation of concrete strength prediction models."""
    
    def __init__(self, models_dir: str = "models", experiment_name: str = "concrete_strength", 
                 model_name: str = "concrete_strength_model"):
        """Initialize the trainer.
        
        Args:
            models_dir: Directory to save trained models
            experiment_name: MLflow experiment name
            model_name: MLflow model registry name
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.experiment_name = experiment_name
        self.model_name = model_name
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
        # Define model configurations
        self.model_configs = self._get_model_configs()
        
    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations for training.
        
        Returns:
            Dictionary of model configurations
        """
        return {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'lasso': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                }
            },
            'elastic_net': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'svr': {
                'model': SVR(),
                'params': {
                    'kernel': ['rbf', 'poly'],
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 0.2]
                }
            },
            'mlp': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
    
    def train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_grid_search: bool = True,
        cv_folds: int = 5
    ) -> Tuple[Any, Dict[str, float], str]:
        """Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            use_grid_search: Whether to use grid search for hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (best_model, metrics, run_id)
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        base_model = config['model']
        param_grid = config['params']
        
        logger.info(f"Training {model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_training") as run:
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("use_grid_search", use_grid_search)
            mlflow.log_param("cv_folds", cv_folds)
            
            if use_grid_search and param_grid:
                # Use RandomizedSearchCV for efficiency with large parameter spaces
                if len(param_grid) > 3 or any(len(v) > 4 for v in param_grid.values()):
                    search = RandomizedSearchCV(
                        base_model,
                        param_grid,
                        n_iter=20,
                        cv=cv_folds,
                        scoring='neg_mean_squared_error',
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    search = GridSearchCV(
                        base_model,
                        param_grid,
                        cv=cv_folds,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                
                # Log best parameters
                for param, value in search.best_params_.items():
                    mlflow.log_param(f"best_{param}", value)
                    
            else:
                # Train with default parameters
                best_model = base_model
                best_model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_val_pred = best_model.predict(X_val)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                y_train, y_train_pred, y_val, y_val_pred
            )
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            signature = infer_signature(X_train, y_train_pred)
            mlflow.sklearn.log_model(
                best_model,
                "model",  # Changed from f"{model_name}_model" to "model" for consistency
                signature=signature
            )
            
            # Save model locally
            model_path = self.models_dir / f"{model_name}_model.joblib"
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(str(model_path))
            
            logger.info(f"{model_name} training completed. Val RMSE: {metrics['val_rmse']:.4f}")
            
            run_id = run.info.run_id
            
        return best_model, metrics, run_id
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        models_to_train: Optional[List[str]] = None,
        register_best: bool = True
    ) -> Dict[str, Tuple[Any, Dict[str, float], str]]:
        """Train all models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            models_to_train: List of model names to train (None for all)
            register_best: Whether to register the best model to MLflow registry
            
        Returns:
            Dictionary of {model_name: (model, metrics, run_id)}
        """
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())
        
        results = {}
        
        for model_name in models_to_train:
            try:
                model, metrics, run_id = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                results[model_name] = (model, metrics, run_id)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Log comparison results
        self._log_model_comparison(results)
        
        # Register best model to registry
        if register_best and results:
            best_model_name, best_model, best_run_id = self.get_best_model_with_run_id(results)
            try:
                logger.info(f"Attempting to register best model: {best_model_name} with run_id: {best_run_id}")
                version = self.register_best_model(
                    best_model_name, 
                    best_model, 
                    best_run_id,
                    stage="Staging",
                    description=f"Best performing model from training run with {len(results)} models"
                )
                logger.info(f"Best model {best_model_name} registered as version {version}")
                return results
            except Exception as e:
                logger.error(f"Failed to register best model: {e}")
                # Continue with normal flow even if registration fails
        
        return results
    
    def _calculate_metrics(
        self,
        y_train_true: pd.Series,
        y_train_pred: np.ndarray,
        y_val_true: pd.Series,
        y_val_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics.
        
        Args:
            y_train_true: True training target values
            y_train_pred: Predicted training target values
            y_val_true: True validation target values
            y_val_pred: Predicted validation target values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            # Training metrics
            'train_rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
            'train_mae': mean_absolute_error(y_train_true, y_train_pred),
            'train_r2': r2_score(y_train_true, y_train_pred),
            
            # Validation metrics
            'val_rmse': np.sqrt(mean_squared_error(y_val_true, y_val_pred)),
            'val_mae': mean_absolute_error(y_val_true, y_val_pred),
            'val_r2': r2_score(y_val_true, y_val_pred),
        }
        
        return metrics
    
    def _log_model_comparison(self, results: Dict[str, Tuple[Any, Dict[str, float], str]]) -> None:
        """Log model comparison results.
        
        Args:
            results: Dictionary of model results with run_ids
        """
        with mlflow.start_run(run_name="model_comparison"):
            comparison_data = []
            
            for model_name, (model, metrics, run_id) in results.items():
                comparison_data.append({
                    'model': model_name,
                    'val_rmse': metrics['val_rmse'],
                    'val_mae': metrics['val_mae'],
                    'val_r2': metrics['val_r2'],
                    'train_rmse': metrics['train_rmse'],
                    'train_mae': metrics['train_mae'],
                    'train_r2': metrics['train_r2'],
                    'run_id': run_id
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('val_rmse')
            
            # Log the best model info
            best_model_name = comparison_df.iloc[0]['model']
            best_val_rmse = comparison_df.iloc[0]['val_rmse']
            
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_val_rmse", best_val_rmse)
            
            # Save comparison results
            comparison_path = self.models_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            mlflow.log_artifact(str(comparison_path))
            
            logger.info("Model comparison completed")
            logger.info(f"Best model: {best_model_name} (Val RMSE: {best_val_rmse:.4f})")
    
    def get_best_model(self, results: Dict[str, Tuple[Any, Dict[str, float]]]) -> Tuple[str, Any]:
        """Get the best performing model based on validation RMSE.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            Tuple of (model_name, model)
        """
        if not results:
            raise ValueError("No trained models available")
        
        best_model_name = min(
            results.keys(),
            key=lambda name: results[name][1]['val_rmse']
        )
        
        best_model = results[best_model_name][0]
        
        logger.info(f"Best model: {best_model_name}")
        return best_model_name, best_model
    
    def get_best_model_with_run_id(self, results: Dict[str, Tuple[Any, Dict[str, float], str]]) -> Tuple[str, Any, str]:
        """Get the best performing model with run_id based on validation RMSE.
        
        Args:
            results: Dictionary of model results with run_ids
            
        Returns:
            Tuple of (model_name, model, run_id)
        """
        if not results:
            raise ValueError("No trained models available")
        
        best_model_name = min(
            results.keys(),
            key=lambda name: results[name][1]['val_rmse']
        )
        
        best_model = results[best_model_name][0]
        best_run_id = results[best_model_name][2]
        
        logger.info(f"Best model: {best_model_name} (run_id: {best_run_id})")
        return best_model_name, best_model, best_run_id

    def save_best_model(
        self,
        model_name: str,
        model: Any,
        filename: str = "best_model.joblib"
    ) -> Path:
        """Save the best model.
        
        Args:
            model_name: Name of the model
            model: Trained model
            filename: Filename to save the model
            
        Returns:
            Path to saved model
        """
        model_path = self.models_dir / filename
        
        model_info = {
            'model': model,
            'model_name': model_name,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        joblib.dump(model_info, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> Tuple[str, Any]:
        """Load a saved model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Tuple of (model_name, model)
        """
        model_info = joblib.load(model_path)
        
        if isinstance(model_info, dict):
            return model_info['model_name'], model_info['model']
        else:
            # Legacy format - just the model
            return "unknown", model_info
        
    def register_best_model(
        self,
        model_name: str,
        model: Any,
        run_id: str,
        stage: str = "Staging"
    ) -> str:
        """Register the best model to MLflow Model Registry.
        
        Args:
            model_name: Name of the model algorithm
            model: Trained model object
            run_id: MLflow run ID where the model was logged
            stage: Model stage (Staging, Production, Archived)
            
        Returns:
            Model version string
        """
        try:
            # Register model
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name
            )
            
            model_version = result.version
            logger.info(f"Model registered: {self.model_name} version {model_version}")
            
            # Transition to specified stage
            if stage != "None":
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=model_version,
                    stage=stage,
                    archive_existing_versions=False
                )
                logger.info(f"Model version {model_version} transitioned to {stage}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def promote_model_to_production(self, version: str = None) -> str:
        """Promote a model version to Production stage.
        
        Args:
            version: Specific version to promote. If None, promotes latest Staging version.
            
        Returns:
            Version that was promoted
        """
        try:
            if version is None:
                # Get latest version in Staging
                staging_versions = self.client.get_latest_versions(
                    name=self.model_name,
                    stages=["Staging"]
                )
                if not staging_versions:
                    raise ValueError("No models in Staging stage to promote")
                version = staging_versions[0].version
            
            # Archive current production models
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            
            logger.info(f"Model version {version} promoted to Production")
            return version
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def load_model_from_registry(self, stage: str = "Production") -> Tuple[Any, Dict[str, Any]]:
        """Load model from MLflow Model Registry.
        
        Args:
            stage: Model stage to load (Production, Staging, etc.)
            
        Returns:
            Tuple of (model, model_info)
        """
        try:
            # Get latest version in specified stage
            latest_versions = self.client.get_latest_versions(
                name=self.model_name,
                stages=[stage]
            )
            
            if not latest_versions:
                raise ValueError(f"No model found in {stage} stage")
            
            latest_version = latest_versions[0]
            model_uri = f"models:/{self.model_name}/{stage}"
            
            # Load model
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get model metadata
            model_info = {
                'model_name': latest_version.name,
                'version': latest_version.version,
                'stage': latest_version.current_stage,
                'run_id': latest_version.run_id,
                'description': latest_version.description,
                'creation_timestamp': latest_version.creation_timestamp,
                'last_updated_timestamp': latest_version.last_updated_timestamp
            }
            
            logger.info(f"Loaded model: {self.model_name} v{latest_version.version} from {stage}")
            return model, model_info
            
        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            raise
    
    def get_model_versions(self) -> List[Dict[str, Any]]:
        """Get all versions of the registered model.
        
        Returns:
            List of model version information
        """
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            version_info = []
            
            for version in versions:
                version_info.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'creation_timestamp': version.creation_timestamp,
                    'last_updated_timestamp': version.last_updated_timestamp,
                    'description': version.description
                })
            
            return sorted(version_info, key=lambda x: int(x['version']), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []

    def get_best_model_from_registry(self) -> Tuple[Any, Dict[str, Any]]:
        """Get the best performing model from MLflow registry based on validation RMSE.
        
        Returns:
            Tuple of (model, model_info) for the best performing model
        """
        try:
            # Get all model versions
            versions = self.get_model_versions()
            
            if not versions:
                raise ValueError("No models found in registry")
            
            # Get metrics for each version from their original runs
            best_version = None
            best_rmse = float('inf')
            
            for version in versions:
                try:
                    run_id = version['run_id']
                    run = self.client.get_run(run_id)
                    val_rmse = run.data.metrics.get('val_rmse', float('inf'))
                    
                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        best_version = version
                        
                except Exception as e:
                    logger.warning(f"Could not get metrics for version {version['version']}: {e}")
                    continue
            
            if best_version is None:
                raise ValueError("No valid model versions found with metrics")
            
            # Load the best model
            stage = best_version['stage']
            model, model_info = self.load_model_from_registry(stage)
            
            logger.info(f"Best model from registry: v{best_version['version']} "
                       f"(Val RMSE: {best_rmse:.4f}, Stage: {stage})")
            
            return model, model_info
            
        except Exception as e:
            logger.error(f"Failed to get best model from registry: {e}")
            raise

    def promote_best_model_to_production(self) -> str:
        """Find the best model in the registry and promote it to Production.
        
        Returns:
            Version that was promoted
        """
        try:
            # Get all model versions and find the best one
            versions = self.get_model_versions()
            
            if not versions:
                raise ValueError("No models found in registry")
            
            # Find the best model by validation RMSE
            best_version = None
            best_rmse = float('inf')
            
            for version in versions:
                try:
                    run_id = version['run_id']
                    run = self.client.get_run(run_id)
                    val_rmse = run.data.metrics.get('val_rmse', float('inf'))
                    
                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        best_version = version
                        
                except Exception as e:
                    logger.warning(f"Could not get metrics for version {version['version']}: {e}")
                    continue
            
            if best_version is None:
                raise ValueError("No valid model versions found with metrics")
            
            # Promote the best model to Production
            version_num = best_version['version']
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version_num,
                stage="Production",
                archive_existing_versions=True
            )
            
            logger.info(f"Best model v{version_num} (Val RMSE: {best_rmse:.4f}) promoted to Production")
            return version_num
            
        except Exception as e:
            logger.error(f"Failed to promote best model: {e}")
            raise

def main():
    """Main function for training models."""
    # Load data
    data_loader = ConcreteDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_prepare_data()
    
    # Initialize trainer
    trainer = ConcreteTrainer()
    
    # Train a subset of models for quick testing (you can train all by removing this list)
    quick_models = ['linear_regression', 'ridge', 'random_forest']
    
    # Train models
    results = trainer.train_all_models(
        X_train, y_train, X_val, y_val,
        models_to_train=quick_models
    )
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model(results)
    
    # Save best model
    trainer.save_best_model(best_model_name, best_model)
    
    print("Training completed successfully!")
    print(f"Best model: {best_model_name}")
    
    # Print results summary
    print("\nResults Summary:")
    for model_name, (model, metrics) in results.items():
        print(f"{model_name}: Val RMSE = {metrics['val_rmse']:.4f}, Val R² = {metrics['val_r2']:.4f}")


if __name__ == "__main__":
    main()
