# Generation Prompts for 3-bike-demand

This file documents the AI-assisted development prompts used to generate each component of the bike-demand forecasting project for reproducibility and transparency.

---

## 1. Exploratory Data Analysis (`src/explore_data.py`)

### Prompt

```
Create a comprehensive exploratory data analysis (EDA) script for the Seoul Bike Demand dataset.

The script should implement the following functions as a professional machine learning engineer would:

1. Load the raw dataset from CSV (handle encoding if needed)
2. Examine dataset structure (shape, dtypes, first rows, column names)
3. Check data quality (missing values, duplicates)
4. Analyze the target variable "Rented Bike Count":
   - Descriptive statistics (mean, median, std, min, max, quartiles)
   - Distribution shape (skewness, kurtosis)
5. Analyze numerical features with summary statistics
6. Analyze categorical features with value counts
7. Explore temporal patterns:
   - Average demand by hour of day (peak and low hours)
   - Average demand by season
   - Average demand by holiday vs. non-holiday
8. Calculate feature correlations with the target variable
9. Identify high correlations between features (multicollinearity)
10. Detect outliers using the IQR method
11. Provide key insights and recommendations for data preparation and modeling

Output should be:
- Well-formatted, easy-to-read terminal output
- Clear section headers
- Concise statistics and insights
- Actionable recommendations for next steps

Keep the code simple, readable, and well-documented with Google-style docstrings.
```

### Purpose

Understand the dataset structure, identify patterns, detect data quality issues, and guide feature engineering decisions for machine learning modeling.

---

## How to Reproduce

To regenerate this script with the same or similar output:

1. Use the prompt above with an AI assistant (Claude, ChatGPT, Copilot, etc.)
2. Run it against the Seoul Bike Sharing Demand dataset
3. The output should provide identical insights about temporal patterns, feature correlations, and data quality

---

---

## 2. Data Preparation and Feature Engineering (`src/prepare_data.py`)

### Prompt

```
Create a data preparation and feature engineering script for the Seoul Bike Demand dataset.

The script should implement the following functions:

1. Load raw dataset from CSV
2. Handle multicollinearity:
   - Drop 'Dew point temperature' (correlation with temperature = 0.913)
   - Drop 'Date' column (replace with engineered features)
3. Create temporal features from 'Hour':
   - is_peak_hour: 1 if hour is 18-22 (6-10 PM, peak demand), 0 otherwise
   - is_night: 1 if hour is 0-6 (midnight-6 AM, low demand), 0 otherwise
   - is_working_day: binary encoding of 'Functioning Day' column
4. Encode categorical variables:
   - Convert 'Holiday' to binary: 'is_holiday' (1=Holiday, 0=No Holiday)
   - One-hot encode 'Seasons' (drop first to avoid multicollinearity)
5. Scale numerical features using StandardScaler:
   - Hour, Temperature(°C), Humidity(%), Wind speed, Visibility, Solar Radiation, Rainfall, Snowfall
6. Create temporal train/test split (chronological, no random shuffle):
   - Train: first 80% of data
   - Test: last 20% of data
   - Reason: Avoids data leakage in time series
7. Save processed data:
   - Train and test sets as CSV files
   - Scaler object as pickle file (for later prediction)
8. Include print statements throughout to show progress and statistics

Output should:
- Show each processing step clearly
- Display feature count at each stage
- Explain why each transformation is necessary
- Print final data shape and statistics
- Save train.csv, test.csv, and scaler.pkl to data/processed/

Keep the code simple, readable, and well-documented with Google-style docstrings.
```

### Purpose

Transform raw data into machine learning-ready format with engineered features, proper scaling, and temporal train/test split to ensure model reproducibility and prevent data leakage.

---

## 3. Model Training and Evaluation (`src/train.py`)

### Prompt

```
Create a model training and evaluation script for the Seoul Bike Demand regression task.

The script should implement the following:

1. Load processed data:
   - Load train.csv and test.csv from data/processed/
   - Load scaler.pkl for reference
2. Separate features (X) and target (y):
   - Target: 'Rented Bike Count'
   - Features: all other columns
3. Train two regression models:
   a. Gradient Boosting (primary model):
      - Use GradientBoostingRegressor
      - n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42
   b. Random Forest (baseline):
      - Use RandomForestRegressor
      - n_estimators=100, max_depth=20, random_state=42
4. Evaluate both models on train and test sets:
   - Calculate MAE (Mean Absolute Error)
   - Calculate RMSE (Root Mean Squared Error)
   - Calculate R² Score
   - Print results for both train and test sets
   - Check for overfitting (ratio of test MAE to train MAE)
5. Display feature importance:
   - Show top 10 important features for each model
   - Use visual bars (█) for clarity
6. Compare models:
   - Show side-by-side comparison of test performance
   - Recommend the better model
7. Save trained models:
   - Save Gradient Boosting model as models/gradient_boosting_model.pkl
   - Save Random Forest model as models/random_forest_model.pkl
8. Include comprehensive print statements:
   - Show each processing step
   - Display hyperparameters
   - Show training progress
   - Display all metrics clearly
   - Provide recommendations for next steps

Output should be:
- Clear section headers for each step
- Readable metrics and comparisons
- Feature importance visualization
- Model comparison summary
- Recommendations for production deployment

Keep the code simple, readable, and well-documented with Google-style docstrings.
```

### Purpose

Train machine learning models on prepared data, evaluate performance on train and test sets, compare models, and save the best model for inference and production deployment.

---

## 4. Predictions and Inference (`src/predict.py`)

### Prompt

```
Create a predictions and inference script for the Seoul Bike Demand forecasting models.

The script should implement the following:

1. Load trained models:
   - Load gradient_boosting_model.pkl
   - Load random_forest_model.pkl
   - Load scaler.pkl (for reference)
2. Load test data:
   - Load test.csv from data/processed/
   - Separate features (X_test) and target (y_test)
3. Make predictions:
   - Use Gradient Boosting model to predict
   - Use Random Forest model to predict
4. Evaluate predictions:
   - Calculate MAE (Mean Absolute Error)
   - Calculate RMSE (Root Mean Squared Error)
   - Calculate R² Score
   - Analyze residuals (actual - predicted)
   - Print residual statistics
5. Display sample predictions:
   - Show first 10 predictions with actual values
   - Include prediction error and error percentage for each
6. Analyze prediction distributions:
   - Show distribution statistics for each model's predictions
   - Compare with actual test data distribution
7. Save predictions:
   - Save Gradient Boosting predictions to predictions/gradient_boosting_predictions.csv
   - Save Random Forest predictions to predictions/random_forest_predictions.csv
   - Include columns: actual_demand, predicted_demand, error, error_percentage
8. Compare models:
   - Side-by-side comparison of MAE, RMSE, R²
   - Recommend better model based on metrics
9. Include comprehensive print statements:
   - Show each step progress
   - Display all metrics clearly
   - Provide actionable recommendations
   - Suggest next steps (deployment, monitoring, retraining)

Output should be:
- Clear section headers for each step
- Readable metrics and comparisons
- Sample predictions with interpretable errors
- Distribution analysis
- Model recommendations for production

Keep the code simple, readable, and well-documented with Google-style docstrings.
```

### Purpose

Load trained models, make predictions on test data, evaluate prediction quality, analyze residuals and distributions, and save results for stakeholder review and production deployment.

---

## Reproducibility and Deployment

This complete pipeline (EDA → Preparation → Training → Predictions) can be reproduced by:

1. Running the documented prompts with any AI assistant
2. Using the same Seoul Bike Sharing Dataset
3. Following the same hyperparameters and preprocessing steps
4. All scripts include print statements for transparency and debugging

For production deployment:
- Use the trained models in `models/` directory
- Preprocess new data using the same steps in `prepare_data.py`
- Use `predict.py` as template for batch or real-time predictions
- Monitor prediction accuracy and retrain periodically
---

## 5. Local LLM Integration (`src/llm_service.py`)

### Prompt

Create a simple Python service that connects the bike demand application to a local Ollama LLM.

The service should:
- Connect to the local Ollama server.
- Use the `llama3.2:1b` model.
- Accept the machine learning prediction as an input.
- Ask the LLM to explain the prediction in simple language.
- Provide one practical recommendation for someone using the Seoul bike-sharing system.
- The LLM must not calculate, change, or replace the machine learning prediction.
- Keep the LLM interaction separate from the main application code.
- Handle LLM request failures so the application can continue showing the ML prediction.

### Purpose

Add a local LLM capability that complements the machine learning prediction by explaining the result and providing a recommendation.

---

## 6. Streamlit Application Integration (`app.py`)

### Prompt

Create a simple Streamlit application for the Seoul Bike Demand Predictor.

The application should:
- Allow the user to enter the required bike demand conditions.
- Load the trained Gradient Boosting model.
- Load the saved scaler.
- Prepare the user's input using the same features and transformations used during model training.
- Display the predicted bike rental demand.
- Send the prediction to the local Ollama LLM for an explanation and recommendation.
- Keep the ML prediction and LLM interaction as separate components.
- Display the ML prediction even if the local LLM is unavailable.
- Show a clear error message when invalid input is entered.
- Keep the interface simple and easy to use.

### Purpose

Create the user-facing application that connects the machine learning prediction with the local LLM explanation.

---

## 7. Validation and Failure Handling

### Prompt

Review the bike demand application and make sure it handles the following situations:

1. Normal operation where the ML model produces a prediction and the local LLM provides an explanation.
2. Invalid user input where the application displays a clear error message.
3. Local LLM unavailable where the ML prediction is still displayed and the user is informed that the LLM is unavailable.

Do not allow an LLM failure to prevent the machine learning prediction from being displayed.

### Purpose

Validate the application's reliability and confirm that the ML prediction and LLM components are separated correctly.