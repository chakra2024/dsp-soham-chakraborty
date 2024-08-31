import sys
import os
# Assuming house_prices is in the parent directory of the notebook
sys.path.append(os.path.abspath('..'))
from house_prices.preprocess import preprocess_data # type: ignore
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split

def build_model(data: pd.DataFrame) -> dict:
    """
    Builds and trains a linear regression model on the provided data.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing features and the target label.

    Returns:
    - Dict[str, float]: A dictionary containing the RMSE of the model on the test set.
    """
    # Select useful features
    label_col = 'SalePrice'
    useful_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'ExterQual']
    continuous_features = ['GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'YearRemodAdd']
    ordinal_feature = 'ExterQual'
    exterior_quality_dict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}

    df = data[useful_features + [label_col]]

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)
    
    # Preprocess data
    train_preprocessed, scaler = preprocess_data(train_df, continuous_features, ordinal_feature, exterior_quality_dict)
    test_preprocessed, _ = preprocess_data(test_df, continuous_features, ordinal_feature, exterior_quality_dict)
    
    # Train the model
    model = LinearRegression()
    model.fit(train_preprocessed, train_df[label_col])
    
    # Evaluate the model
    y_pred = model.predict(test_preprocessed)
    rmse = np.sqrt(mean_squared_error(test_df[label_col], y_pred))
    
    # Save the model and scaler
    joblib.dump(model, 'C:/Users/SOHAM/dsp-soham-chakraborty/models/model.joblib')
    joblib.dump(scaler, 'C:/Users/SOHAM/dsp-soham-chakraborty/models/scaler.joblib')
    
    return {'rmse': rmse}
