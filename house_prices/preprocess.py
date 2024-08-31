import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame, continuous_features: list, ordinal_feature: str, exterior_quality_dict: dict):
    
    """
    Preprocesses the input DataFrame by applying ordinal encoding to the specified
    ordinal feature and scaling the continuous features.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the features.
    - continuous_features (List[str]): List of continuous feature names to be scaled.
    - ordinal_feature (str): The name of the ordinal feature to be encoded.
    - exterior_quality_dict (Dict[str, int]): A dictionary mapping ordinal feature values 
      to integers.

    Returns:
    - Tuple[pd.DataFrame, StandardScaler]: A tuple containing the preprocessed DataFrame 
      and the fitted scaler.
    """
    # Ordinal Encoding
    df[ordinal_feature] = df[ordinal_feature].apply(lambda x: exterior_quality_dict.get(x, np.nan))
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler to the continuous features
    scaler.fit(df[continuous_features])
    
     # Transform the continuous features using the fitted scaler
    df_scaled = scaler.transform(df[continuous_features])

    # Combine preprocessed features
    df_preprocessed = pd.concat([
        pd.DataFrame(df_scaled, columns=continuous_features),
        df[ordinal_feature].reset_index(drop=True)
    ], axis=1)
    
    return df_preprocessed, scaler
