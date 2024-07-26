import pandas as pd
import numpy as np
from house_prices.preprocess import preprocess_data
import joblib
import os

MODEL_PATH = 'C:/Users/SOHAM/dsp-soham-chakraborty/models/'

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    
    # Load the trained model
    model = joblib.load(os.path.join(MODEL_PATH, 'model.joblib'))
    # Preprocess input data
    X_input = preprocess_data(input_data, is_train=False)
    # Make predictions
    predictions = model.predict(X_input)

    return predictions
