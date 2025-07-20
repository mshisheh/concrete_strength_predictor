"""
Model evaluation module for concrete strength prediction.

This module handles comprehensive model evaluation including
performance metrics, validation, and statistical analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.model_selection import cross_val_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcreteEvaluator:
    """Handles comprehensive evaluation of concrete strength prediction models."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the evaluator.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        metrics = {
            f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}mae': mean_absolute_error(y_true, y_pred),
            f'{prefix}mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            f'{prefix}r2': r2_score(y_true, y_pred),
            f'{prefix}explained_variance': explained_variance_score(y_true, y_pred),
            f'{prefix}max_error': np.max(np.abs(y_true - y_pred)),
            f'{prefix}mean_error': np.mean(y_true - y_pred),
            f'{prefix}std_error': np.std(y_true - y_pred)
        }
        
        return metrics
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            X_train: Training features (optional)
            y_train: Training target (optional)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {}
        
        # Test set evaluation
        y_test_pred = model.predict(X_test)
        test_metrics = self.calculate_regression_metrics(y_test, y_test_pred, 'test')
        results.update(test_metrics)
        
        # Training set evaluation (if provided)
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            train_metrics = self.calculate_regression_metrics(y_train, y_train_pred, 'train')
            results.update(train_metrics)
            
            # Calculate overfitting metrics
            results['overfitting_rmse'] = train_metrics['train_rmse'] - test_metrics['test_rmse']
            results['overfitting_r2'] = train_metrics['train_r2'] - test_metrics['test_r2']
        
        # Validation set evaluation (if provided)
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_metrics = self.calculate_regression_metrics(y_val, y_val_pred, 'val')
            results.update(val_metrics)
        
        # Cross-validation (if training data provided)
        if X_train is not None and y_train is not None:
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5, scoring='neg_mean_squared_error'
                )
                results['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
                results['cv_rmse_std'] = np.sqrt(cv_scores.std())
                
                cv_r2_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5, scoring='r2'
                )
                results['cv_r2_mean'] = cv_r2_scores.mean()
                results['cv_r2_std'] = cv_r2_scores.std()
                
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
        
        # Prediction intervals (simple approach)
        residuals = y_test - y_test_pred
        results['prediction_interval_95'] = np.percentile(np.abs(residuals), 95)
        results['prediction_interval_90'] = np.percentile(np.abs(residuals), 90)
        
        logger.info(f"Model evaluation completed. Test RMSE: {results['test_rmse']:.4f}")
        
        return results
    
    def compare_models(
        self,
        models_dict: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sort_by: str = 'test_rmse'
    ) -> pd.DataFrame:
        """Compare multiple models on test data.
        
        Args:
            models_dict: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test target
            sort_by: Metric to sort by
            
        Returns:
            Comparison dataframe
        """
        comparison_results = []
        
        for model_name, model in models_dict.items():
            try:
                results = self.evaluate_model(model, X_test, y_test)
                results['model_name'] = model_name
                comparison_results.append(results)
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        if not comparison_results:
            raise ValueError("No models could be evaluated")
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Sort by specified metric (ascending for error metrics, descending for R²)
        ascending = sort_by not in ['test_r2', 'val_r2', 'train_r2', 'cv_r2_mean']
        comparison_df = comparison_df.sort_values(sort_by, ascending=ascending)
        
        return comparison_df
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        importance_type: str = 'auto'
    ) -> pd.DataFrame:
        """Extract feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            importance_type: Type of importance ('auto', 'permutation')
            
        Returns:
            Feature importance dataframe
        """
        importance_data = []
        
        # Try to get built-in feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_type = 'built_in'
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(model.coef_)
            importance_type = 'coefficients'
        else:
            logger.warning("Model doesn't have built-in feature importance")
            return pd.DataFrame()
        
        for i, feature in enumerate(feature_names):
            importance_data.append({
                'feature': feature,
                'importance': importances[i],
                'importance_type': importance_type
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def calculate_prediction_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """Calculate detailed prediction errors for analysis.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dataframe with error analysis
        """
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        relative_errors = errors / y_true * 100
        
        error_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'error': errors,
            'abs_error': abs_errors,
            'relative_error': relative_errors,
            'squared_error': errors ** 2
        })
        
        # Add error categories
        error_df['error_magnitude'] = pd.cut(
            abs_errors,
            bins=[0, 5, 10, 20, np.inf],
            labels=['Small (<5)', 'Medium (5-10)', 'Large (10-20)', 'Very Large (>20)']
        )
        
        return error_df
    
    def generate_evaluation_report(
        self,
        model: Any,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test target
            X_train: Training features (optional)
            y_train: Training target (optional)
            save_path: Path to save the report
            
        Returns:
            Complete evaluation report
        """
        logger.info(f"Generating evaluation report for {model_name}")
        
        report = {
            'model_name': model_name,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'test_set_size': len(X_test),
            'feature_count': X_test.shape[1]
        }
        
        # Model evaluation
        evaluation_results = self.evaluate_model(
            model, X_test, y_test, X_train, y_train
        )
        report['metrics'] = evaluation_results
        
        # Feature importance
        try:
            feature_importance = self.get_feature_importance(
                model, X_test.columns.tolist()
            )
            report['feature_importance'] = feature_importance.to_dict('records')
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            report['feature_importance'] = None
        
        # Prediction errors analysis
        y_pred = model.predict(X_test)
        error_analysis = self.calculate_prediction_errors(y_test.values, y_pred)
        
        report['error_summary'] = {
            'error_distribution': error_analysis['error_magnitude'].value_counts().to_dict(),
            'worst_predictions': error_analysis.nlargest(5, 'abs_error')[
                ['y_true', 'y_pred', 'abs_error']
            ].to_dict('records'),
            'best_predictions': error_analysis.nsmallest(5, 'abs_error')[
                ['y_true', 'y_pred', 'abs_error']
            ].to_dict('records')
        }
        
        # Save report if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def load_and_evaluate_best_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_path: str = None
    ) -> Dict[str, Any]:
        """Load and evaluate the best saved model.
        
        Args:
            X_test: Test features
            y_test: Test target
            model_path: Path to model file (if None, looks for best_model.joblib)
            
        Returns:
            Evaluation results
        """
        if model_path is None:
            model_path = self.models_dir / "best_model.joblib"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load model
        model_info = joblib.load(model_path)
        
        if isinstance(model_info, dict):
            model = model_info['model']
            model_name = model_info['model_name']
        else:
            model = model_info
            model_name = "loaded_model"
        
        # Evaluate
        results = self.evaluate_model(model, X_test, y_test)
        results['model_name'] = model_name
        
        logger.info(f"Evaluated {model_name}. Test RMSE: {results['test_rmse']:.4f}")
        
        return results


def main():
    """Main function for model evaluation."""
    from concrete.data_loader import ConcreteDataLoader
    from concrete.trainer import ConcreteTrainer
    
    # Load data
    data_loader = ConcreteDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_prepare_data()
    
    # Initialize evaluator
    evaluator = ConcreteEvaluator()
    
    # Try to load best model and evaluate
    try:
        results = evaluator.load_and_evaluate_best_model(X_test, y_test)
        print("Best model evaluation:")
        print(f"Test RMSE: {results['test_rmse']:.4f}")
        print(f"Test R²: {results['test_r2']:.4f}")
        print(f"Test MAE: {results['test_mae']:.4f}")
        
    except FileNotFoundError:
        print("No trained model found. Please run training first.")


if __name__ == "__main__":
    main()
