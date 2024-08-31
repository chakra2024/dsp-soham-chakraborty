import pandas as pd
import numpy as np
import joblib
import sys
import os
# Assuming house_prices is in the parent directory of the notebook
sys.path.append(os.path.abspath('..'))
from house_prices.preprocess import preprocess_data # type: ignore
from sklearn.impute import SimpleImputer

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Makes predictions on the input data using the pre-trained model.

    Parameters:
    - input_data (pd.DataFrame): The input DataFrame containing the features to predict on.

    Returns:
    - List[float]: A list of predicted values.
    """
    # Features used during model training
    continuous_features = ['GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'YearRemodAdd']
    ordinal_feature = 'ExterQual'
    exterior_quality_dict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}

    # Load scaler and model
    scaler = joblib.load('C:/Users/SOHAM/dsp-soham-chakraborty/models/scaler.joblib')
    model = joblib.load('C:/Users/SOHAM/dsp-soham-chakraborty/models/model.joblib')


    # Preprocess ordinal feature
    input_data[ordinal_feature] = input_data[ordinal_feature].apply(lambda x: exterior_quality_dict.get(x, np.nan))

    # Ensure the test data has the same features as the training data
    input_data_subset = input_data[continuous_features + [ordinal_feature]]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    input_data_imputed = pd.DataFrame(imputer.fit_transform(input_data_subset), columns=input_data_subset.columns)

    # Apply the scaler to the continuous features
    input_scaled = scaler.transform(input_data_imputed[continuous_features])

    # Combine scaled continuous features and ordinal feature into a DataFrame
    input_preprocessed = pd.concat([
        pd.DataFrame(input_scaled, columns=continuous_features),
        input_data_imputed[ordinal_feature].reset_index(drop=True)
    ], axis=1)

    # Make predictions
    predictions = model.predict(input_preprocessed)
    return predictions