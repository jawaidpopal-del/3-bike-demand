"""
Predictions and Inference for Seoul Bike Demand Forecasting.

This module handles:
- Loading trained models
- Making predictions on new data
- Analyzing prediction results
- Saving predictions to file
"""

import pandas as pd
import numpy as np
import pickle
import os


def load_trained_model(model_path: str) -> object:
    """Load a trained model from pickle file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded: {model_path}")
    return model


def load_scaler(scaler_path: str) -> object:
    """Load the scaler used during training."""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded: {scaler_path}")
    return scaler


def load_test_data(test_file: str) -> tuple:
    """Load test data and separate features from target."""
    df = pd.read_csv(test_file)
    X_test = df.drop('Rented Bike Count', axis=1)
    y_test = df['Rented Bike Count']
    print(f"✓ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    return X_test, y_test


def make_predictions(model, X_test: pd.DataFrame, model_name: str) -> np.ndarray:
    """Make predictions using trained model."""
    print(f"\n⏳ Making predictions with {model_name}...")
    predictions = model.predict(X_test)
    print(f"✓ Predictions complete: {len(predictions)} predictions")
    return predictions


def calculate_prediction_errors(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculate prediction errors and statistics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    residuals = y_true - y_pred
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'residuals': residuals,
        'mean_residual': residuals.mean(),
        'std_residual': residuals.std(),
        'min_residual': residuals.min(),
        'max_residual': residuals.max(),
    }


def print_prediction_summary(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> dict:
    """Print detailed prediction summary."""
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS: {model_name.upper()}")
    print(f"{'='*60}")
    
    errors = calculate_prediction_errors(y_true, y_pred)
    
    print(f"\nPrediction Accuracy:")
    print(f"  MAE (Mean Absolute Error):  {errors['mae']:.2f} bikes")
    print(f"  RMSE (Root Mean Squared Error): {errors['rmse']:.2f} bikes")
    print(f"  R² Score:                   {errors['r2']:.4f}")
    
    print(f"\nResidual Analysis (Prediction - Actual):")
    print(f"  Mean Residual:              {errors['mean_residual']:.2f} bikes")
    print(f"  Std Dev of Residuals:       {errors['std_residual']:.2f} bikes")
    print(f"  Min Residual:               {errors['min_residual']:.2f} bikes")
    print(f"  Max Residual:               {errors['max_residual']:.2f} bikes")
    
    return errors


def print_sample_predictions(y_true: pd.Series, y_pred: np.ndarray, n_samples: int = 10) -> None:
    """Print sample predictions vs actual values."""
    print(f"\nSample Predictions (first {n_samples} test samples):")
    print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<10}")
    print(f"{'-'*54}")
    
    for i in range(min(n_samples, len(y_true))):
        actual = y_true.iloc[i]
        predicted = y_pred[i]
        error = actual - predicted
        error_pct = (error / actual * 100) if actual != 0 else 0
        
        print(f"{i:<8} {actual:<12.0f} {predicted:<12.0f} {error:<12.0f} {error_pct:<10.1f}%")


def save_predictions(y_test: pd.Series, y_pred: np.ndarray, 
                    output_file: str, model_name: str) -> None:
    """Save predictions to CSV file."""
    results_df = pd.DataFrame({
        'actual_demand': y_test.values,
        'predicted_demand': y_pred,
        'error': y_test.values - y_pred,
        'error_percentage': ((y_test.values - y_pred) / y_test.values * 100)
    })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Predictions saved: {output_file}")


def analyze_prediction_distribution(y_pred: np.ndarray) -> None:
    """Analyze distribution of predictions."""
    print(f"\nPrediction Distribution:")
    print(f"  Mean:     {y_pred.mean():.2f} bikes")
    print(f"  Median:   {np.median(y_pred):.2f} bikes")
    print(f"  Std Dev:  {y_pred.std():.2f} bikes")
    print(f"  Min:      {y_pred.min():.2f} bikes")
    print(f"  Max:      {y_pred.max():.2f} bikes")
    print(f"  Range:    {y_pred.max() - y_pred.min():.2f} bikes")


def main(data_dir: str = 'data/processed/',
         models_dir: str = 'models/',
         predictions_dir: str = 'predictions/') -> None:
    """Run complete prediction pipeline."""
    print("\n" + "="*60)
    print("SEOUL BIKE DEMAND - PREDICTIONS & INFERENCE")
    print("="*60)
    
    # Load test data
    print(f"\n{'='*60}")
    print("STEP 1: LOADING TEST DATA")
    print(f"{'='*60}")
    test_file = os.path.join(data_dir, 'test.csv')
    X_test, y_test = load_test_data(test_file)
    
    # Load models
    print(f"\n{'='*60}")
    print("STEP 2: LOADING TRAINED MODELS")
    print(f"{'='*60}")
    gb_model_path = os.path.join(models_dir, 'gradient_boosting_model.pkl')
    rf_model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    
    gb_model = load_trained_model(gb_model_path)
    rf_model = load_trained_model(rf_model_path)
    
    # Make predictions with both models
    print(f"\n{'='*60}")
    print("STEP 3: MAKING PREDICTIONS")
    print(f"{'='*60}")
    
    gb_predictions = make_predictions(gb_model, X_test, 'Gradient Boosting')
    rf_predictions = make_predictions(rf_model, X_test, 'Random Forest')
    
    # Evaluate predictions
    print(f"\n{'='*60}")
    print("STEP 4: EVALUATING PREDICTIONS")
    print(f"{'='*60}")
    
    gb_errors = print_prediction_summary(y_test, gb_predictions, 'Gradient Boosting')
    print_sample_predictions(y_test, gb_predictions, n_samples=10)
    
    rf_errors = print_prediction_summary(y_test, rf_predictions, 'Random Forest')
    print_sample_predictions(y_test, rf_predictions, n_samples=10)
    
    # Analyze prediction distributions
    print(f"\n{'='*60}")
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nGradient Boosting Predictions:")
    analyze_prediction_distribution(gb_predictions)
    
    print(f"\nRandom Forest Predictions:")
    analyze_prediction_distribution(rf_predictions)
    
    print(f"\nActual Test Data:")
    analyze_prediction_distribution(y_test.values)
    
    # Save predictions
    print(f"\n{'='*60}")
    print("STEP 5: SAVING PREDICTIONS")
    print(f"{'='*60}")
    
    gb_pred_file = os.path.join(predictions_dir, 'gradient_boosting_predictions.csv')
    rf_pred_file = os.path.join(predictions_dir, 'random_forest_predictions.csv')
    
    save_predictions(y_test, gb_predictions, gb_pred_file, 'Gradient Boosting')
    save_predictions(y_test, rf_predictions, rf_pred_file, 'Random Forest')
    
    # Model comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Metric':<25} {'Gradient Boosting':<20} {'Random Forest':<20}")
    print(f"{'-'*65}")
    print(f"{'MAE':<25} {gb_errors['mae']:<20.2f} {rf_errors['mae']:<20.2f}")
    print(f"{'RMSE':<25} {gb_errors['rmse']:<20.2f} {rf_errors['rmse']:<20.2f}")
    print(f"{'R² Score':<25} {gb_errors['r2']:<20.4f} {rf_errors['r2']:<20.4f}")
    
    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    if gb_errors['mae'] < rf_errors['mae']:
        print(f"✓ Gradient Boosting has better test performance (lower MAE)")
        print(f"  Recommended model: gradient_boosting_model.pkl")
    else:
        print(f"✓ Random Forest has better test performance (lower MAE)")
        print(f"  Recommended model: random_forest_model.pkl")
    
    print(f"\nNext Steps:")
    print(f"  1. Review predictions in {predictions_dir}")
    print(f"  2. Analyze residual patterns for improvement opportunities")
    print(f"  3. Deploy best model to production")
    print(f"  4. Monitor prediction performance on new data")
    print(f"  5. Retrain periodically with new data")
    
    print(f"\n{'='*60}")
    print("PREDICTION COMPLETE ✓")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
