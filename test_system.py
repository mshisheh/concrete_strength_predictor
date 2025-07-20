#!/usr/bin/env python3
"""
Simple test script to verify the concrete strength prediction system works.
"""

from concrete.data_loader import ConcreteDataLoader
from concrete.trainer import ConcreteTrainer
from concrete.evaluator import ConcreteEvaluator

def test_system():
    """Test the complete system functionality."""
    print("🏗️ Testing Concrete Strength Prediction System...")
    
    # Test 1: Data Loading
    print("\n1. Testing data loading...")
    loader = ConcreteDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_and_prepare_data()
    print(f"✅ Data loaded successfully!")
    print(f"   - Training set: {X_train.shape}")
    print(f"   - Validation set: {X_val.shape}")
    print(f"   - Test set: {X_test.shape}")
    print(f"   - Target range: {y_train.min():.2f} - {y_train.max():.2f} MPa")
    
    # Test 2: Model Training
    print("\n2. Testing model training...")
    trainer = ConcreteTrainer()
    
    # Train just a simple model for testing
    results = []
    try:
        # Train Linear Regression (fastest)
        model, metrics = trainer.train_single_model(
            "linear_regression", 
            X_train, y_train, X_val, y_val
        )
        result = {
            'model_name': 'linear_regression',
            'model': model,
            **metrics
        }
        results.append(result)
        print(f"✅ Linear Regression trained successfully!")
        print(f"   - Training RMSE: {result['train_rmse']:.2f}")
        print(f"   - Validation RMSE: {result['val_rmse']:.2f}")
        
        # Train Random Forest
        model, metrics = trainer.train_single_model(
            "random_forest", 
            X_train, y_train, X_val, y_val, use_grid_search=False  # Skip grid search for speed
        )
        result = {
            'model_name': 'random_forest',
            'model': model,
            **metrics
        }
        results.append(result)
        print(f"✅ Random Forest trained successfully!")
        print(f"   - Training RMSE: {result['train_rmse']:.2f}")
        print(f"   - Validation RMSE: {result['val_rmse']:.2f}")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False
    
    # Test 3: Model Evaluation
    print("\n3. Testing model evaluation...")
    try:
        evaluator = ConcreteEvaluator()
        best_result = min(results, key=lambda x: x['val_rmse'])
        
        # Evaluate on test set
        test_metrics = evaluator.evaluate_model(
            best_result['model'], 
            X_test, y_test
        )
        
        print(f"✅ Model evaluation completed!")
        print(f"   - Best model: {best_result['model_name']}")
        print(f"   - Test RMSE: {test_metrics['test_rmse']:.2f}")
        print(f"   - Test R²: {test_metrics['test_r2']:.3f}")
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The system is working correctly.")
    return True

if __name__ == "__main__":
    test_system()
