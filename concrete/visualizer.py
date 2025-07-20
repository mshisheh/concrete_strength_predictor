"""
Visualization module for concrete strength prediction.

This module creates comprehensive visualizations for data exploration,
model evaluation, and feature analysis using matplotlib, seaborn, plotly, and SHAP.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: poetry install --extras viz")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ConcreteVisualizer:
    """Handles all visualizations for concrete strength prediction project."""
    
    def __init__(self, output_dir: str = "plots"):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color palette
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
    
    def plot_data_distribution(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot distribution of all features and target.
        
        Args:
            df: Dataframe with features and target
            save_path: Path to save the plot
        """
        n_features = len(df.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, column in enumerate(df.columns):
            ax = axes[i]
            
            # Histogram with KDE
            df[column].hist(bins=30, alpha=0.7, ax=ax, color=self.colors['primary'])
            ax2 = ax.twinx()
            df[column].plot.kde(ax=ax2, color=self.colors['danger'], linewidth=2)
            
            ax.set_title(f'Distribution of {column}', fontsize=12, fontweight='bold')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax2.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot correlation matrix heatmap.
        
        Args:
            df: Dataframe with features and target
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .5},
            fmt='.2f'
        )
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def plot_target_vs_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'compressive_strength',
        save_path: Optional[str] = None
    ) -> None:
        """Plot target variable against each feature.
        
        Args:
            df: Dataframe with features and target
            target_col: Name of target column
            save_path: Path to save the plot
        """
        feature_cols = [col for col in df.columns if col != target_col]
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(feature_cols):
            ax = axes[i]
            
            # Scatter plot with regression line
            sns.scatterplot(
                data=df, x=feature, y=target_col,
                alpha=0.6, ax=ax, color=self.colors['primary']
            )
            sns.regplot(
                data=df, x=feature, y=target_col,
                scatter=False, ax=ax, color=self.colors['danger']
            )
            
            # Calculate correlation
            corr = df[feature].corr(df[target_col])
            ax.set_title(f'{feature} vs {target_col}\n(r = {corr:.3f})', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Target vs features plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        save_path: Optional[str] = None
    ) -> None:
        """Plot predicted vs actual values.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            title: Plot title
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        ax1.scatter(y_true, y_pred, alpha=0.6, color=self.colors['primary'])
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{title} - Scatter Plot')
        ax1.grid(True, alpha=0.3)
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # Residual plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color=self.colors['secondary'])
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{title} - Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions vs actual plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance.
        
        Args:
            importance_df: Dataframe with feature importance
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # Horizontal bar plot
        bars = plt.barh(importance_df['feature'], importance_df['importance'],
                       color=self.colors['primary'], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.xlabel('Importance')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'test_rmse',
        save_path: Optional[str] = None
    ) -> None:
        """Plot model comparison.
        
        Args:
            comparison_df: Dataframe with model comparison results
            metric: Metric to compare
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Sort by metric
        ascending = metric not in ['test_r2', 'val_r2', 'train_r2']
        comparison_df = comparison_df.sort_values(metric, ascending=ascending)
        
        # Bar plot
        bars = plt.bar(comparison_df['model_name'], comparison_df[metric],
                      color=self.colors['primary'], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison - {metric.upper()}', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        train_sizes: List[int],
        metric_name: str = "RMSE",
        save_path: Optional[str] = None
    ) -> None:
        """Plot learning curves.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            train_sizes: Training set sizes
            metric_name: Name of the metric
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_scores, 'o-', color=self.colors['primary'],
                label=f'Training {metric_name}')
        plt.plot(train_sizes, val_scores, 'o-', color=self.colors['secondary'],
                label=f'Validation {metric_name}')
        
        plt.xlabel('Training Set Size')
        plt.ylabel(metric_name)
        plt.title(f'Learning Curves - {metric_name}', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves plot saved to {save_path}")
        
        plt.show()
    
    def create_shap_plots(
        self,
        model: Any,
        X: pd.DataFrame,
        save_dir: Optional[str] = None
    ) -> None:
        """Create SHAP explanation plots.
        
        Args:
            model: Trained model
            X: Feature data
            save_dir: Directory to save plots
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP plots. Install with: poetry install --extras viz")
            return
            
        try:
            # Create explainer
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False)
            if save_dir:
                plt.savefig(f"{save_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # Waterfall plot for first instance
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(shap_values[0], show=False)
            if save_dir:
                plt.savefig(f"{save_dir}/shap_waterfall.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # Feature importance plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            if save_dir:
                plt.savefig(f"{save_dir}/shap_importance.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("SHAP plots created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create SHAP plots: {e}")
    
    def create_interactive_dashboard(
        self,
        df: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Create interactive dashboard with Plotly.
        
        Args:
            df: Dataframe with features and target
            y_true: True target values (optional)
            y_pred: Predicted target values (optional)
            save_path: Path to save HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Feature Distributions', 'Correlation Heatmap',
                'Predictions vs Actual', 'Residual Analysis'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Feature distribution (first feature as example)
        first_feature = df.columns[0]
        fig.add_trace(
            go.Histogram(x=df[first_feature], name=first_feature, nbinsx=30),
            row=1, col=1
        )
        
        # Correlation heatmap
        corr_matrix = df.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ),
            row=1, col=2
        )
        
        # Predictions vs actual (if available)
        if y_true is not None and y_pred is not None:
            fig.add_trace(
                go.Scatter(
                    x=y_true, y=y_pred,
                    mode='markers',
                    name='Predictions',
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ),
                row=2, col=1
            )
            
            # Residuals
            residuals = y_true - y_pred
            fig.add_trace(
                go.Scatter(
                    x=y_pred, y=residuals,
                    mode='markers',
                    name='Residuals',
                    opacity=0.6
                ),
                row=2, col=2
            )
            
            # Zero line for residuals
            fig.add_trace(
                go.Scatter(
                    x=[y_pred.min(), y_pred.max()], y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(dash='dash', color='red')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Concrete Strength Prediction Dashboard",
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text=first_feature, row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        if y_true is not None and y_pred is not None:
            fig.update_xaxes(title_text="Actual Values", row=2, col=1)
            fig.update_yaxes(title_text="Predicted Values", row=2, col=1)
            fig.update_xaxes(title_text="Predicted Values", row=2, col=2)
            fig.update_yaxes(title_text="Residuals", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()


def main():
    """Main function for testing visualizations."""
    from concrete.data_loader import ConcreteDataLoader
    
    # Load data
    data_loader = ConcreteDataLoader()
    df_raw = data_loader.load_raw_data()
    df_clean = data_loader.clean_data(df_raw)
    
    # Initialize visualizer
    visualizer = ConcreteVisualizer()
    
    # Create basic plots
    print("Creating data visualizations...")
    
    visualizer.plot_data_distribution(df_clean)
    visualizer.plot_correlation_matrix(df_clean)
    visualizer.plot_target_vs_features(df_clean)
    
    print("Visualization testing completed!")


if __name__ == "__main__":
    main()
