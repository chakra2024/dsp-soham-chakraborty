import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from house_prices.preprocess import preprocess_data
import joblib
import os
MODEL_PATH = 'C:/Users/SOHAM/dsp-soham-chakraborty/models/'

def build_model(data: pd.DataFrame) -> dict: 
    # Split data to avoid leakage
    train_df, test_df = train_test_split(data, test_size=0.33, 
    random_state=42)
    # Persist the trained model
    label_col = 'SalePrice'
    useful_features = ['Foundation', 'KitchenQual', 'TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']
    # Select features and label for training data
    train_df = train_df[useful_features + [label_col]]
    # Preprocess train data
    X_train = preprocess_data(train_df.drop(columns=[label_col]), is_train=True)
    y_train = train_df[label_col]
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(model, os.path.join(MODEL_PATH, 'model.joblib'))
    # Model evaluation
    test_df = test_df[useful_features + [label_col]]
    # Preprocess test data
    X_test = preprocess_data(test_df.drop(columns=[label_col]), is_train=False)                                                       
    y_test = test_df[label_col]
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {'rmse': rmse}
