"""
Simplified Streamlit web application for concrete strength prediction.

This app uses MLflow Model Registry by default and provides a clean, production-ready interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import os
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from concrete.data_loader import ConcreteDataLoader
from concrete.trainer import ConcreteTrainer
from concrete.evaluator import ConcreteEvaluator
from concrete.visualizer import ConcreteVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MLflow
mlflow.set_tracking_uri("file:./mlruns")

class ModelManager:
    """Manages model loading from MLflow Registry with fallback to local files."""
    
    def __init__(self):
        self.client = MlflowClient()
        self.model_name = "concrete_strength_model"
        
    def load_model(self, force_reload: bool = False):
        """Load model from MLflow Registry (Staging preferred) or local file."""
        try:
            # Try to load from MLflow Registry
            try:
                # First try Staging
                model = mlflow.sklearn.load_model(f"models:/{self.model_name}/Staging")
                logger.info("Loaded model from MLflow Registry (Staging)")
                return model, "MLflow Registry (Staging)"
            except Exception:
                # Then try Production
                try:
                    model = mlflow.sklearn.load_model(f"models:/{self.model_name}/Production")
                    logger.info("Loaded model from MLflow Registry (Production)")
                    return model, "MLflow Registry (Production)"
                except Exception:
                    logger.warning("No models found in MLflow Registry")
                    
            # Fallback to local file
            model_path = "models/best_model.joblib"
            if os.path.exists(model_path):
                model_info = joblib.load(model_path)
                if isinstance(model_info, dict):
                    model = model_info['model']
                else:
                    model = model_info
                logger.info("Loaded model from local file")
                return model, "Local File"
            else:
                return None, "No Model Found"
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, f"Error: {str(e)}"
    
    def promote_to_production(self):
        """Promote the current Staging model to Production."""
        try:
            # Get the latest Staging version
            staging_versions = self.client.get_latest_versions(
                self.model_name, 
                stages=["Staging"]
            )
            
            if not staging_versions:
                return False, "No Staging model found to promote"
            
            version = staging_versions[0].version
            
            # Transition to Production
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production"
            )
            
            return True, f"Successfully promoted model version {version} to Production"
            
        except Exception as e:
            return False, f"Error promoting model: {str(e)}"

def main():
    """Main application function."""
    
    # Initialize components
    model_manager = ModelManager()
    data_loader = ConcreteDataLoader()
    
    # Header
    st.markdown('<h1 class="main-header">🏗️ Concrete Strength Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["🔮 Predict", "📊 Explore Data", "🤖 Train Model"],
            index=0
        )
        
        st.divider()
        
        # Model status
        st.subheader("Model Status")
        model, source = model_manager.load_model()
        
        if model is not None:
            st.success(f"✅ Model Loaded")
            st.info(f"Source: {source}")
            
            # Show promote button only if using Staging
            if source == "MLflow Registry (Staging)":
                if st.button("🚀 Promote to Production"):
                    success, message = model_manager.promote_to_production()
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        else:
            st.error("❌ No Model Available")
            st.info("Please train a model first")
    
    # Main content based on selected page
    if page == "🔮 Predict":
        show_prediction_page(model_manager, model)
    elif page == "📊 Explore Data":
        show_data_exploration_page(data_loader)
    elif page == "🤖 Train Model":
        show_training_page(data_loader)

def show_prediction_page(model_manager, model):
    """Show the prediction interface."""
    st.header("Concrete Strength Prediction")
    
    if model is None:
        st.warning("No model available. Please train a model first.")
        return
    
    st.markdown("Enter the concrete mix parameters to predict compressive strength:")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cement = st.number_input(
                "Cement (kg/m³)",
                min_value=0.0,
                max_value=1000.0,
                value=300.0,
                help="Amount of cement in the concrete mix"
            )
            blast_furnace_slag = st.number_input(
                "Blast Furnace Slag (kg/m³)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                help="Amount of blast furnace slag"
            )
            fly_ash = st.number_input(
                "Fly Ash (kg/m³)",
                min_value=0.0,
                max_value=300.0,
                value=0.0,
                help="Amount of fly ash"
            )
            water = st.number_input(
                "Water (kg/m³)",
                min_value=100.0,
                max_value=300.0,
                value=180.0,
                help="Amount of water in the mix"
            )
        
        with col2:
            superplasticizer = st.number_input(
                "Superplasticizer (kg/m³)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                help="Amount of superplasticizer"
            )
            coarse_aggregate = st.number_input(
                "Coarse Aggregate (kg/m³)",
                min_value=800.0,
                max_value=1500.0,
                value=1000.0,
                help="Amount of coarse aggregate"
            )
            fine_aggregate = st.number_input(
                "Fine Aggregate (kg/m³)",
                min_value=600.0,
                max_value=1000.0,
                value=800.0,
                help="Amount of fine aggregate"
            )
            age = st.number_input(
                "Age (days)",
                min_value=1,
                max_value=365,
                value=28,
                help="Age of the concrete when tested"
            )
        
        submit_button = st.form_submit_button("🔮 Predict Strength")
    
    if submit_button:
        # Prepare input data
        input_data = np.array([[
            cement, blast_furnace_slag, fly_ash, water,
            superplasticizer, coarse_aggregate, fine_aggregate, age
        ]])
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Strength",
                    f"{prediction:.2f} MPa",
                    help="Predicted compressive strength"
                )
            
            with col2:
                # Strength category
                if prediction < 20:
                    category = "Low Strength"
                    color = "🔴"
                elif prediction < 40:
                    category = "Medium Strength"
                    color = "🟡"
                else:
                    category = "High Strength"
                    color = "🟢"
                
                st.metric(
                    "Strength Category",
                    f"{color} {category}",
                    help="Strength classification"
                )
            
            with col3:
                # Typical use cases
                if prediction < 15:
                    use_case = "Non-structural"
                elif prediction < 25:
                    use_case = "Residential"
                elif prediction < 40:
                    use_case = "Commercial"
                else:
                    use_case = "Infrastructure"
                
                st.metric(
                    "Typical Use",
                    use_case,
                    help="Recommended application"
                )
            
            # Additional information
            st.markdown("---")
            st.subheader("Additional Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Standards Reference:**
                - Standard testing age: 28 days
                - Minimum residential: 20 MPa
                - Typical commercial: 25-40 MPa
                - High-performance: >50 MPa
                """)
            
            with col2:
                st.info(f"""
                **Mix Summary:**
                - Total cementitious: {cement + blast_furnace_slag + fly_ash:.0f} kg/m³
                - Water-cement ratio: {water/cement:.2f}
                - Total aggregate: {coarse_aggregate + fine_aggregate:.0f} kg/m³
                """)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def show_data_exploration_page(data_loader):
    """Show data exploration interface."""
    st.header("Data Exploration")
    
    # Load data
    try:
        X, y = data_loader.load_data()
        data = X.copy()
        data['compressive_strength'] = y
        
        # Basic statistics
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Features", len(X.columns))
        with col3:
            st.metric("Avg Strength", f"{y.mean():.1f} MPa")
        with col4:
            st.metric("Max Strength", f"{y.max():.1f} MPa")
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        feature = st.selectbox(
            "Select feature to visualize:",
            options=list(X.columns) + ['compressive_strength']
        )
        
        fig = px.histogram(
            data, 
            x=feature, 
            title=f"Distribution of {feature}",
            nbins=30
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with target
        st.subheader("Feature Correlations with Strength")
        
        correlations = X.corrwith(y).sort_values(ascending=False)
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title="Feature Correlations with Compressive Strength"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Raw Data Sample")
        st.dataframe(data.head(100), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_training_page(data_loader):
    """Show model training interface."""
    st.header("Model Training")
    
    st.markdown("""
    Train a new model and automatically register it to MLflow Registry.
    The best performing model will be registered to the **Staging** stage.
    """)
    
    # Training options
    with st.form("training_form"):
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Fraction of data to use for testing"
            )
            
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
        
        with col2:
            enable_hyperparameter_tuning = st.checkbox(
                "Enable Hyperparameter Tuning",
                value=True,
                help="Use GridSearchCV for hyperparameter optimization"
            )
            
            auto_register = st.checkbox(
                "Auto-register to MLflow",
                value=True,
                help="Automatically register best model to Staging"
            )
        
        submit_training = st.form_submit_button("🚀 Start Training")
    
    if submit_training:
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Load data
                X, y = data_loader.load_data()
                
                # Initialize trainer
                trainer = ConcreteTrainer()
                
                # Train models
                results = trainer.train_and_evaluate(
                    X, y,
                    test_size=test_size,
                    cv_folds=cv_folds,
                    tune_hyperparameters=enable_hyperparameter_tuning
                )
                
                # Register best model if enabled
                if auto_register and results:
                    trainer.register_best_model(stage="Staging")
                
                st.success("✅ Training completed successfully!")
                
                # Display results
                if results:
                    st.subheader("Training Results")
                    
                    # Convert results to DataFrame for display
                    results_df = pd.DataFrame([
                        {
                            'Model': name,
                            'RMSE': metrics['rmse'],
                            'R²': metrics['r2'],
                            'MAE': metrics['mae'],
                            'MAPE': metrics['mape']
                        }
                        for name, metrics in results.items()
                    ])
                    
                    results_df = results_df.sort_values('RMSE')
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Highlight best model
                    best_model = results_df.iloc[0]['Model']
                    st.info(f"🏆 Best model: **{best_model}** (RMSE: {results_df.iloc[0]['RMSE']:.3f})")
                    
                    if auto_register:
                        st.success("🔄 Best model automatically registered to MLflow Registry (Staging)")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                logger.error(f"Training error: {e}")

if __name__ == "__main__":
    main()
