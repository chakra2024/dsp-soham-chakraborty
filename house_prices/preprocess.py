import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame, continuous_features: list, ordinal_feature: str, exterior_quality_dict: dict):
    # Ordinal Encoding
    df[ordinal_feature] = df[ordinal_feature].apply(lambda x: exterior_quality_dict.get(x, np.nan))
    
    # Scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[continuous_features])
    
    # Combine preprocessed features
    df_preprocessed = pd.concat([
        pd.DataFrame(df_scaled, columns=continuous_features),
        df[ordinal_feature].reset_index(drop=True)
    ], axis=1)
    
    return df_preprocessed, scaler
