"""
Streamlit web application for concrete strength prediction.

This app provides an interactive interface for:
- Data exploration and visualization
- Model training and evaluation
- Making predictions on new data
- Viewing model explanations with SHAP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import io
import json
from typing import Dict, Any, Optional

# Import our modules
from concrete.data_loader import ConcreteDataLoader
from concrete.trainer import ConcreteTrainer
from concrete.evaluator import ConcreteEvaluator
from concrete.visualizer import ConcreteVisualizer

# Configure Streamlit page
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the concrete dataset."""
    data_loader = ConcreteDataLoader()
    try:
        df_raw = data_loader.load_raw_data()
        df_clean = data_loader.clean_data(df_raw)
        return df_raw, df_clean, data_loader.get_feature_info()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


@st.cache_data
def load_processed_data():
    """Load processed training data."""
    data_loader = ConcreteDataLoader()
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_prepare_data()
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        st.error(f"Error loading processed data: {e}")
        return None, None, None, None, None, None


@st.cache_resource
def load_trained_model():
    """Load the best trained model."""
    try:
        model_path = Path("models/best_model.joblib")
        if model_path.exists():
            model_info = joblib.load(model_path)
            if isinstance(model_info, dict):
                return model_info['model'], model_info['model_name'], model_info.get('timestamp', '')
            else:
                return model_info, "Loaded Model", ''
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def get_current_model():
    """Get the current model, checking for updates."""
    try:
        model_path = Path("models/best_model.joblib")
        if not model_path.exists():
            return None, None, None
        
        # Get file modification time
        mod_time = model_path.stat().st_mtime
        
        # Always check if we need to update the cache
        if ('model_mod_time' not in st.session_state or 
            st.session_state.model_mod_time != mod_time):
            
            st.session_state.model_mod_time = mod_time
            # Force clear all related caches
            load_trained_model.clear()
            
            # Force reload by calling the function directly without cache
            try:
                model_info = joblib.load(model_path)
                if isinstance(model_info, dict):
                    model, model_name, timestamp = (
                        model_info['model'], 
                        model_info['model_name'], 
                        model_info.get('timestamp', '')
                    )
                else:
                    model, model_name, timestamp = model_info, "Loaded Model", ''
                
                # Update cache manually to ensure consistency
                return model, model_name, timestamp
            except Exception as e:
                st.error(f"Error loading fresh model: {e}")
                return None, None, None
        
        # Use cached version if timestamps match
        return load_trained_model()
    except Exception as e:
        st.error(f"Error checking model: {e}")
        return None, None, None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("<h1 class='main-header'>🏗️ Concrete Strength Predictor</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts concrete compressive strength using machine learning.
    Explore the data, train models, and make predictions with confidence intervals.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Exploration", "🔬 Model Training", 
         "📈 Model Evaluation", "🎯 Make Predictions", "📋 About"]
    )
    
    # Model status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Current Model")
    model, model_name, timestamp = get_current_model()
    if model is not None:
        st.sidebar.success(f"✅ {model_name}")
        if timestamp:
            # Show just the date part for brevity
            date_part = timestamp.split('T')[0] if 'T' in timestamp else timestamp
            st.sidebar.caption(f"📅 {date_part}")
    else:
        st.sidebar.warning("⚠️ No model")
    
    # Add model cache refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Force Refresh Model"):
        # Clear all caches and session state
        load_trained_model.clear()
        if 'model_mod_time' in st.session_state:
            del st.session_state.model_mod_time
        
        # Clear all streamlit caches
        st.cache_data.clear()
        st.cache_resource.clear()
        
        st.sidebar.success("All caches cleared!")
        st.rerun()
    
    # Debug info for troubleshooting
    if st.sidebar.checkbox("🔍 Show Debug Info"):
        st.sidebar.markdown("**Debug Information:**")
        model_path = Path("models/best_model.joblib")
        if model_path.exists():
            mod_time = model_path.stat().st_mtime
            st.sidebar.text(f"File mod time: {mod_time}")
            session_time = st.session_state.get('model_mod_time', 'Not set')
            st.sidebar.text(f"Session time: {session_time}")
            st.sidebar.text(f"Match: {mod_time == session_time if session_time != 'Not set' else False}")
    
    # Load data
    df_raw, df_clean, feature_info = load_data()
    
    if df_clean is None:
        st.error("Failed to load data. Please check your connection and try again.")
        return
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(df_clean)
    elif page == "📊 Data Exploration":
        show_data_exploration_page(df_raw, df_clean, feature_info)
    elif page == "🔬 Model Training":
        show_model_training_page()
    elif page == "📈 Model Evaluation":
        show_model_evaluation_page()
    elif page == "🎯 Make Predictions":
        show_prediction_page(feature_info)
    elif page == "📋 About":
        show_about_page()


def show_home_page(df: pd.DataFrame):
    """Display the home page with overview statistics."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    
    with col2:
        st.metric("Features", df.shape[1] - 1)
    
    with col3:
        st.metric("Avg Strength", f"{df['compressive_strength'].mean():.1f} MPa")
    
    with col4:
        st.metric("Max Strength", f"{df['compressive_strength'].max():.1f} MPa")
    
    st.markdown("---")
    
    # Quick overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Strength Distribution")
        fig = px.histogram(
            df, x='compressive_strength',
            nbins=30, title="Compressive Strength Distribution",
            labels={'compressive_strength': 'Compressive Strength (MPa)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Feature Overview")
        
        # Show basic statistics
        stats_df = df.describe().round(2)
        st.dataframe(stats_df)
    
    # Recent activity or model status
    st.markdown("---")
    st.subheader("🎯 Model Status")
    
    model, model_name, timestamp = get_current_model()
    if model is not None:
        st.success(f"✅ Trained model available: {model_name}")
        if timestamp:
            st.info(f"📅 Model trained: {timestamp}")
        st.info("You can make predictions using the trained model!")
    else:
        st.warning("⚠️ No trained model found. Please train a model first.")


def show_data_exploration_page(df_raw: pd.DataFrame, df_clean: pd.DataFrame, feature_info: Dict):
    """Display data exploration page."""
    
    st.header("📊 Data Exploration")
    
    # Data overview tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Info", "📈 Distributions", "🔗 Correlations", "🧹 Data Quality"])
    
    with tab1:
        st.subheader("Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Raw Data:**")
            st.write(f"- Shape: {df_raw.shape}")
            st.write(f"- Missing values: {df_raw.isnull().sum().sum()}")
            
            st.write("**Clean Data:**")
            st.write(f"- Shape: {df_clean.shape}")
            st.write(f"- Samples removed: {len(df_raw) - len(df_clean)}")
        
        with col2:
            st.write("**Feature Descriptions:**")
            for feature, description in feature_info.items():
                st.write(f"- **{feature}**: {description}")
        
        st.subheader("Sample Data")
        st.dataframe(df_clean.head(10))
    
    with tab2:
        st.subheader("Feature Distributions")
        
        # Select feature to visualize
        feature = st.selectbox("Select a feature:", df_clean.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                df_clean, x=feature,
                nbins=30, title=f"Distribution of {feature}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df_clean, y=feature, title=f"Box Plot of {feature}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader(f"Statistics for {feature}")
        stats = df_clean[feature].describe()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{stats['mean']:.2f}")
        with col2:
            st.metric("Std Dev", f"{stats['std']:.2f}")
        with col3:
            st.metric("Min", f"{stats['min']:.2f}")
        with col4:
            st.metric("Max", f"{stats['max']:.2f}")
    
    with tab3:
        st.subheader("Feature Correlations")
        
        # Correlation matrix
        corr_matrix = df_clean.corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Strongest correlations with target
        target_corr = corr_matrix['compressive_strength'].abs().sort_values(ascending=False)
        target_corr = target_corr.drop('compressive_strength')  # Remove self-correlation
        
        st.subheader("Strongest Correlations with Target")
        for feature, corr in target_corr.head(5).items():
            st.write(f"**{feature}**: {corr:.3f}")
    
    with tab4:
        st.subheader("Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values:**")
            missing = df_raw.isnull().sum()
            if missing.sum() == 0:
                st.success("✅ No missing values found")
            else:
                st.dataframe(missing[missing > 0])
        
        with col2:
            st.write("**Data Types:**")
            st.dataframe(df_clean.dtypes)
        
        # Outlier analysis
        st.subheader("Outlier Analysis")
        
        feature_for_outliers = st.selectbox(
            "Select feature for outlier analysis:", 
            df_clean.columns,
            key="outlier_feature"
        )
        
        Q1 = df_clean[feature_for_outliers].quantile(0.25)
        Q3 = df_clean[feature_for_outliers].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[
            (df_clean[feature_for_outliers] < lower_bound) | 
            (df_clean[feature_for_outliers] > upper_bound)
        ]
        
        st.write(f"Outliers in {feature_for_outliers}: {len(outliers)} ({len(outliers)/len(df_clean)*100:.1f}%)")


def show_model_training_page():
    """Display model training page."""
    
    st.header("🔬 Model Training")
    
    st.info("Train multiple machine learning models and compare their performance.")
    
    # Training configuration
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test set size", 0.1, 0.3, 0.2, 0.05)
        
        with col2:
            val_size = st.slider("Validation set size", 0.1, 0.3, 0.2, 0.05)
        
        with col3:
            random_state = st.number_input("Random state", 1, 100, 42)
    
    # Model selection
    available_models = {
        'Linear Regression': 'linear_regression',
        'Ridge Regression': 'ridge',
        'Lasso Regression': 'lasso',
        'Elastic Net': 'elastic_net',
        'Random Forest': 'random_forest',
        'Gradient Boosting': 'gradient_boosting',
        'XGBoost': 'xgboost',
        'Support Vector Regression': 'svr',
        'Neural Network': 'mlp'
    }
    
    selected_models = st.multiselect(
        "Select models to train:",
        options=list(available_models.keys()),
        default=['Linear Regression', 'Ridge Regression', 'Random Forest', 'XGBoost'],
        help="Select one or more models to train and compare"
    )
    
    if not selected_models:
        st.warning("Please select at least one model to train.")
        return
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        if not selected_models:
            st.error("Please select at least one model.")
            return
        
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Load data
                data_loader = ConcreteDataLoader()
                X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_prepare_data(
                    test_size=test_size, val_size=val_size, random_state=random_state
                )
                
                # Initialize trainer
                trainer = ConcreteTrainer()
                
                # Train selected models
                models_to_train = [available_models[model] for model in selected_models]
                results = trainer.train_all_models(
                    X_train, y_train, X_val, y_val,
                    models_to_train=models_to_train
                )
                
                # Display results
                st.success("✅ Training completed!")
                
                # Results comparison
                comparison_data = []
                for model_name, (model, metrics) in results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Val RMSE': metrics['val_rmse'],
                        'Val R²': metrics['val_r2'],
                        'Val MAE': metrics['val_mae'],
                        'Train RMSE': metrics['train_rmse'],
                        'Train R²': metrics['train_r2']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('Val RMSE')
                
                st.subheader("📊 Model Comparison")
                st.dataframe(comparison_df)
                
                # Best model
                best_model_name, best_model = trainer.get_best_model(results)
                st.success(f"🎯 Best model: {best_model_name}")
                
                # Save best model
                trainer.save_best_model(best_model_name, best_model)
                
                # Aggressive cache clearing to ensure fresh model load
                load_trained_model.clear()
                st.cache_data.clear()
                st.cache_resource.clear()
                
                # Clear session state to force fresh check
                if 'model_mod_time' in st.session_state:
                    del st.session_state.model_mod_time
                
                st.success("✅ Best model saved successfully! You can now use it for predictions.")
                st.success("🔄 Model cache refreshed - the app will now use the newly trained model.")
                
                # Show which model was actually saved
                st.info(f"🎯 Saved model: **{best_model_name}**")
                
                # Visualization
                fig = px.bar(
                    comparison_df, x='Model', y='Val RMSE',
                    title="Model Comparison - Validation RMSE"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Training failed: {e}")


def show_model_evaluation_page():
    """Display model evaluation page."""
    
    st.header("📈 Model Evaluation")
    
    # Load model and data
    model, model_name, timestamp = get_current_model()
    
    if model is None:
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    st.success(f"✅ Loaded model: {model_name}")
    if timestamp:
        st.info(f"📅 Model trained: {timestamp}")
    
    # Load test data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
        
        if X_test is None:
            st.error("Failed to load test data.")
            return
        
        # Initialize evaluator
        evaluator = ConcreteEvaluator()
        
        # Evaluate model
        with st.spinner("Evaluating model..."):
            results = evaluator.evaluate_model(
                model, X_test, y_test, X_train, y_train, X_val, y_val
            )
        
        # Display metrics
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test RMSE", f"{results['test_rmse']:.3f}")
        
        with col2:
            st.metric("Test R²", f"{results['test_r2']:.3f}")
        
        with col3:
            st.metric("Test MAE", f"{results['test_mae']:.3f}")
        
        with col4:
            st.metric("Test MAPE", f"{results['test_mape']:.1f}%")
        
        # Predictions vs Actual
        st.subheader("🎯 Predictions vs Actual")
        
        y_pred = model.predict(X_test)
        
        fig = px.scatter(
            x=y_test, y=y_pred,
            title="Predictions vs Actual Values",
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}
        )
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(dash="dash", color="red")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual plot
        st.subheader("📉 Residual Analysis")
        
        residuals = y_test - y_pred
        
        fig = px.scatter(
            x=y_pred, y=residuals,
            title="Residual Plot",
            labels={'x': 'Predicted Values', 'y': 'Residuals'}
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            st.subheader("🎯 Feature Importance")
            
            importance_df = evaluator.get_feature_importance(
                model, X_test.columns.tolist()
            )
            
            if not importance_df.empty:
                fig = px.bar(
                    importance_df.head(10),
                    x='importance', y='feature',
                    orientation='h',
                    title="Top 10 Most Important Features"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        with st.expander("📋 Detailed Metrics"):
            metrics_df = pd.DataFrame([
                {"Metric": k, "Value": f"{v:.4f}"} for k, v in results.items()
                if isinstance(v, (int, float))
            ])
            st.dataframe(metrics_df)
            
    except Exception as e:
        st.error(f"Evaluation failed: {e}")


def show_prediction_page(feature_info: Dict):
    """Display prediction page."""
    
    st.header("🎯 Make Predictions")
    
    # Load model
    model, model_name, timestamp = get_current_model()
    
    if model is None:
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    st.success(f"✅ Using model: {model_name}")
    if timestamp:
        st.info(f"📅 Model trained: {timestamp}")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "Upload CSV File"]
    )
    
    if input_method == "Manual Input":
        show_manual_input_prediction(model, feature_info)
    else:
        show_batch_prediction(model)


def show_manual_input_prediction(model: Any, feature_info: Dict):
    """Show manual input prediction interface."""
    
    st.subheader("🔧 Enter Concrete Mix Properties")
    
    # Create input widgets for each feature
    inputs = {}
    
    col1, col2 = st.columns(2)
    
    features = list(feature_info.keys())
    mid_point = len(features) // 2
    
    with col1:
        for feature in features[:mid_point]:
            inputs[feature] = st.number_input(
                f"{feature.replace('_', ' ').title()} (kg/m³)" if feature != 'age' else f"{feature.title()} (days)",
                min_value=0.0,
                value=100.0 if feature != 'age' else 28.0,
                help=feature_info[feature]
            )
    
    with col2:
        for feature in features[mid_point:]:
            inputs[feature] = st.number_input(
                f"{feature.replace('_', ' ').title()} (kg/m³)" if feature != 'age' else f"{feature.title()} (days)",
                min_value=0.0,
                value=100.0 if feature != 'age' else 28.0,
                help=feature_info[feature]
            )
    
    # Predict button
    if st.button("🔮 Predict Strength", type="primary"):
        try:
            # Prepare input data
            input_data = pd.DataFrame([inputs])
            
            # Load data loader to get scaler
            data_loader = ConcreteDataLoader()
            # We need to fit the scaler on training data
            X_train, _, _, _, _, _ = data_loader.load_and_prepare_data()
            
            # Scale the input
            input_scaled = pd.DataFrame(
                data_loader.scaler.transform(input_data),
                columns=input_data.columns
            )
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.success(f"🎯 Predicted Compressive Strength: **{prediction:.2f} MPa**")
            
            # Interpretation
            if prediction < 20:
                st.warning("⚠️ Low strength concrete - suitable for non-structural applications")
            elif prediction < 40:
                st.info("ℹ️ Moderate strength concrete - suitable for residential construction")
            elif prediction < 60:
                st.success("✅ High strength concrete - suitable for commercial construction")
            else:
                st.success("🏆 Very high strength concrete - suitable for heavy-duty applications")
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                st.dataframe(input_data.T.rename(columns={0: "Value"}))
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def show_batch_prediction(model: Any):
    """Show batch prediction interface."""
    
    st.subheader("📁 Upload CSV File for Batch Predictions")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with concrete mix properties"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            input_df = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Uploaded Data")
            st.dataframe(input_df.head())
            
            if st.button("🔮 Predict All", type="primary"):
                with st.spinner("Making predictions..."):
                    # Load data loader to get scaler
                    data_loader = ConcreteDataLoader()
                    X_train, _, _, _, _, _ = data_loader.load_and_prepare_data()
                    
                    # Scale the input
                    input_scaled = pd.DataFrame(
                        data_loader.scaler.transform(input_df),
                        columns=input_df.columns
                    )
                    
                    # Make predictions
                    predictions = model.predict(input_scaled)
                    
                    # Add predictions to dataframe
                    result_df = input_df.copy()
                    result_df['predicted_strength'] = predictions
                    
                    st.subheader("📈 Results")
                    st.dataframe(result_df)
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="concrete_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("📊 Prediction Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Count", len(predictions))
                    
                    with col2:
                        st.metric("Mean Strength", f"{predictions.mean():.2f} MPa")
                    
                    with col3:
                        st.metric("Min Strength", f"{predictions.min():.2f} MPa")
                    
                    with col4:
                        st.metric("Max Strength", f"{predictions.max():.2f} MPa")
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")


def show_about_page():
    """Display about page."""
    
    st.header("📋 About This Application")
    
    st.markdown("""
    ## 🏗️ Concrete Strength Predictor
    
    This application uses machine learning to predict the compressive strength of concrete
    based on its constituent materials and age.
    
    ### 📊 Dataset
    - **Source**: UCI Machine Learning Repository
    - **Features**: 8 input variables (cement, water, aggregates, etc.)
    - **Target**: Compressive strength (MPa)
    - **Samples**: ~1000 concrete test records
    
    ### 🤖 Machine Learning Models
    - Linear Regression
    - Ridge & Lasso Regression
    - Random Forest
    - Gradient Boosting
    - Support Vector Regression
    - Neural Networks
    
    ### 🛠️ Technologies Used
    - **Framework**: Streamlit
    - **ML Library**: scikit-learn
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Experiment Tracking**: MLflow
    - **Model Interpretation**: SHAP
    - **Data Processing**: Pandas, NumPy
    
    ### 📈 Features
    - **Data Exploration**: Interactive visualizations and statistics
    - **Model Training**: Compare multiple ML algorithms
    - **Model Evaluation**: Comprehensive performance metrics
    - **Predictions**: Single and batch prediction capabilities
    - **Interpretability**: SHAP explanations for model decisions
    
    ### 🎯 Use Cases
    - **Quality Control**: Predict strength before testing
    - **Mix Design**: Optimize concrete formulations
    - **Cost Reduction**: Minimize expensive physical testing
    - **Research**: Understand concrete strength relationships
    
    ### 📝 How to Use
    1. **Explore Data**: Navigate to Data Exploration to understand the dataset
    2. **Train Models**: Use Model Training to train and compare algorithms
    3. **Evaluate**: Check model performance in Model Evaluation
    4. **Predict**: Make predictions on new concrete mixes
    
    ### ⚠️ Disclaimer
    This tool is for educational and research purposes. Always validate predictions
    with physical testing for critical applications.
    """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Data Preprocessing
        - Outlier removal using IQR method
        - Feature scaling with StandardScaler
        - Train/validation/test split (60%/20%/20%)
        
        ### Model Selection
        - Cross-validation for hyperparameter tuning
        - Grid search for optimal parameters
        - Performance evaluation on unseen test data
        
        ### Metrics
        - RMSE (Root Mean Square Error)
        - MAE (Mean Absolute Error)
        - R² (Coefficient of Determination)
        - MAPE (Mean Absolute Percentage Error)
        """)


if __name__ == "__main__":
    main()
