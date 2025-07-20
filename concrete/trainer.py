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

from concrete.data_loader import ConcreteDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcreteTrainer:
    """Handles training and evaluation of concrete strength prediction models."""
    
    def __init__(self, models_dir: str = "models", experiment_name: str = "concrete_strength"):
        """Initialize the trainer.
        
        Args:
            models_dir: Directory to save trained models
            experiment_name: MLflow experiment name
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.experiment_name = experiment_name
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
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
    ) -> Tuple[Any, Dict[str, float]]:
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
            Tuple of (best_model, metrics)
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        base_model = config['model']
        param_grid = config['params']
        
        logger.info(f"Training {model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_training"):
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
                f"{model_name}_model",
                signature=signature
            )
            
            # Save model locally
            model_path = self.models_dir / f"{model_name}_model.joblib"
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(str(model_path))
            
            logger.info(f"{model_name} training completed. Val RMSE: {metrics['val_rmse']:.4f}")
            
        return best_model, metrics
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """Train all models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            models_to_train: List of model names to train (None for all)
            
        Returns:
            Dictionary of {model_name: (model, metrics)}
        """
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())
        
        results = {}
        
        for model_name in models_to_train:
            try:
                model, metrics = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                results[model_name] = (model, metrics)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Log comparison results
        self._log_model_comparison(results)
        
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
    
    def _log_model_comparison(self, results: Dict[str, Tuple[Any, Dict[str, float]]]) -> None:
        """Log model comparison results.
        
        Args:
            results: Dictionary of model results
        """
        with mlflow.start_run(run_name="model_comparison"):
            comparison_data = []
            
            for model_name, (model, metrics) in results.items():
                comparison_data.append({
                    'model': model_name,
                    'val_rmse': metrics['val_rmse'],
                    'val_mae': metrics['val_mae'],
                    'val_r2': metrics['val_r2'],
                    'train_rmse': metrics['train_rmse'],
                    'train_mae': metrics['train_mae'],
                    'train_r2': metrics['train_r2']
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
