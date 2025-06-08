import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from feature_engineering import engineer_features

class ReimbursementPredictor:
    """
    Predicts reimbursement using a single model with engineered features.
    """
    def __init__(self, model_class=GradientBoostingRegressor, model_params=None):
        self.model_class = model_class
        self.model_params = model_params
        self.model = None
        self.trained_columns = []

        if self.model_class == GradientBoostingRegressor and self.model_params is None:
            # Default to previously optimized hyperparameters for GBR
            self.model_params = {
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_leaf': 1,
                'min_samples_split': 5,
                'n_estimators': 100,
                'subsample': 0.7,
                'random_state': 42
            }
        elif self.model_params is None:
            self.model_params = {} # Default to empty dict for other models to use their defaults

    def fit(self, data: pd.DataFrame):
        """
        Trains a model using engineered features and specified hyperparameters.
        """
        print(f"Training {self.model_class.__name__} with engineered features...")
        
        # 1. Engineer features and prepare the dataset
        processed_data = engineer_features(data)
        self.trained_columns = [col for col in processed_data.columns if col != 'expected_output']
        
        X = processed_data[self.trained_columns]
        y = processed_data['expected_output']

        # Initialize the chosen model with its parameters
        self.model = self.model_class(**self.model_params)
        
        # Fit the model
        self.model.fit(X, y)
        print(f"{self.model_class.__name__} training complete. Using parameters: {self.model.get_params()}")

    def predict(self, new_trips: pd.DataFrame) -> np.ndarray:
        """
        Predicts reimbursement using the trained model.
        """
        if not self.trained_columns or not self.model:
            raise RuntimeError("Model has not been trained. Please call .fit() first.")
            
        # 1. Engineer features for the new data
        processed_trips = engineer_features(new_trips)
        final_trips = processed_trips.reindex(columns=self.trained_columns, fill_value=0)

        # 2. Get predictions from the model
        predictions = self.model.predict(final_trips)
        
        return predictions

    def save_model(self, directory: str = 'models'):
        """
        Saves the trained model and necessary metadata to disk.
        """
        if not self.trained_columns:
            raise RuntimeError("No models to save. Please train the models first using .fit()")
            
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        if self.model is not None:
            model_path = os.path.join(directory, 'model.joblib')
            joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'trained_columns': self.trained_columns,
            'model_class': self.model_class.__name__,
            'model_params': self.model_params
        }
        
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

# --- Main execution block ---
if __name__ == "__main__":
    # 1. Load the training data from JSON
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Convert JSON data to DataFrame
    data = []
    for case in cases:
        row = case['input'].copy()
        row['expected_output'] = case['expected_output']
        data.append(row)
    
    df = pd.DataFrame(data)
    
    df['expected_output_bins'] = pd.qcut(df['expected_output'], q=5, labels=False, duplicates='drop')
    stratify_column = df['expected_output_bins']

    # Split into train and test sets (80/20 split)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_column)
    
    # --- Gradient Boosting Regressor ---
    print("\n--- Training Gradient Boosting Regressor ---")
    # ReimbursementPredictor defaults to GradientBoostingRegressor, 
    # or we can be explicit as done here.
    gbr_predictor = ReimbursementPredictor(model_class=GradientBoostingRegressor) 
    gbr_predictor.fit(train_df)
    gbr_predictor.save_model(directory='models/gbr_model')

    # Get predictions and evaluate for GBR
    test_features = test_df.drop('expected_output', axis=1).copy()
    gbr_predicted_amounts = gbr_predictor.predict(test_features)

    mae_gbr = mean_absolute_error(test_df['expected_output'], gbr_predicted_amounts)
    rmse_gbr = np.sqrt(mean_squared_error(test_df['expected_output'], gbr_predicted_amounts))
    r2_gbr = r2_score(test_df['expected_output'], gbr_predicted_amounts)

    print("\n--- GBR Model Evaluation on Test Set ---")
    print(f"Mean Absolute Error: ${mae_gbr:.2f}")
    print(f"Root Mean Squared Error: ${rmse_gbr:.2f}")
    print(f"R² Score: {r2_gbr:.4f}")

    # 5. Display some example predictions (using GBR results)
    print("\n--- Example Predictions (using GBR) ---")
    example_df = test_df.copy() # Use .copy() to avoid SettingWithCopyWarning
    example_df['predicted_reimbursement'] = gbr_predicted_amounts
    for i, row in example_df.head(5).iterrows():
        print(
            f"Trip: {row['trip_duration_days']} days, {row['miles_traveled']} miles, "
            f"${row['total_receipts_amount']:.2f} receipts -> "
            f"Predicted: ${row['predicted_reimbursement']:.2f}, "
            f"Actual: ${row['expected_output']:.2f}"
        )
