<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Concrete Strength Predictor - Copilot Instructions

This is a professional machine learning project for predicting concrete compressive strength. When working on this project, please follow these guidelines:

## Project Context
- **Domain**: Civil Engineering / Materials Science
- **Task**: Regression (predicting concrete compressive strength in MPa)
- **Dataset**: UCI Concrete Compressive Strength Dataset
- **Tech Stack**: Python, scikit-learn, Streamlit, MLflow, Docker

## Code Style & Standards
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write comprehensive docstrings for all classes and functions
- Use Poetry for dependency management
- Follow the existing project structure and naming conventions

## ML/Data Science Best Practices
- Always use proper train/validation/test splits
- Scale features using StandardScaler before model training
- Use cross-validation for hyperparameter tuning
- Track experiments with MLflow
- Include comprehensive model evaluation metrics (RMSE, R², MAE, MAPE)
- Implement proper error handling for data loading and model operations

## Domain-Specific Knowledge
- Concrete strength is measured in MPa (Megapascals)
- All material quantities are in kg/m³ except age (days)
- Typical concrete strength ranges from 10-80 MPa
- Age significantly affects strength (28 days is standard testing age)
- Physical constraints: all material quantities must be non-negative

## File Organization
- **concrete/data_loader.py**: Data loading, cleaning, preprocessing
- **concrete/trainer.py**: Model training, hyperparameter tuning
- **concrete/evaluator.py**: Model evaluation, metrics calculation
- **concrete/visualizer.py**: Data visualization, plotting
- **concrete/app.py**: Streamlit web application
- **tests/**: Unit tests for all modules

## Testing Guidelines
- Write unit tests for all core functionality
- Use pytest for testing framework
- Include edge cases and error conditions
- Aim for >80% code coverage
- Mock external dependencies (file I/O, network requests)

## Documentation
- Update README.md for any new features
- Include code examples in docstrings
- Document API changes and new configuration options
- Keep changelog updated for releases

## MLOps Practices
- Use MLflow for experiment tracking
- Version control models and datasets
- Implement proper CI/CD pipeline
- Include Docker support for deployment
- Monitor model performance over time

## Security & Quality
- Validate all user inputs
- Handle file uploads securely
- Use environment variables for sensitive configuration
- Regular dependency updates for security patches

## Visualization Guidelines
- Use Plotly for interactive charts in Streamlit
- Use matplotlib/seaborn for static analysis plots
- Include proper axis labels and titles
- Ensure colorblind-friendly color schemes
- Make plots responsive and mobile-friendly

When suggesting code improvements or new features, prioritize:
1. Code maintainability and readability
2. Performance and scalability
3. User experience in the Streamlit app
4. Reproducibility of ML experiments
5. Proper error handling and logging
