import pandas as pd
import numpy as np
import joblib
from house_prices.preprocess import preprocess_data

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    continuous_features = ['GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'YearRemodAdd']
    ordinal_feature = 'ExterQual'
    exterior_quality_dict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
    
    scaler = joblib.load('models/scaler.joblib')
    model = joblib.load('models/model.joblib')
    
    # Preprocess data
    input_data[ordinal_feature] = input_data[ordinal_feature].apply(lambda x: exterior_quality_dict.get(x, np.nan))
    input_scaled = scaler.transform(input_data[continuous_features])
    input_preprocessed = pd.concat([
        pd.DataFrame(input_scaled, columns=continuous_features),
        input_data[ordinal_feature].reset_index(drop=True)
    ], axis=1)
    
    predictions = model.predict(input_preprocessed)
    return predictions
