import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

MODEL_PATH = 'C:/Users/SOHAM/dsp-soham-chakraborty/models/'

def preprocess_data(df: pd.DataFrame, is_train: bool = True):
   
    # Feature groups
    continuous_features = ['TotRmsAbvGrd', 'YrSold', '1stFlrSF']
    # categorical_features = ['Foundation', 'KitchenQual']
    features_to_one_hot_encode = ['Foundation']
    features_to_ordinal_encode = ['KitchenQual']
    if is_train:
        # One Hot Encoding
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoder.fit(df[features_to_one_hot_encode])
        df_one_hot_encoded = one_hot_encoder.transform(df[features_to_one_hot_encode])
        # Save the encoder
        joblib.dump(one_hot_encoder, os.path.join(MODEL_PATH,'one_hot_encoder.joblib'))
        # Ordinal Encoding
        kitchen_quality_dict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}
        df['KitchenQual'] = df['KitchenQual'].apply(lambda x: kitchen_quality_dict[x])
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        df[continuous_features] = imputer.fit_transform(df[continuous_features])
        # Save the imputer
        joblib.dump(imputer, os.path.join(MODEL_PATH, 'imputer.joblib'))
        # Scaling
        scaler = StandardScaler()
        scaler.fit(df[continuous_features])
        df_scaled = scaler.transform(df[continuous_features])
        # Save the scaler
        joblib.dump(scaler, os.path.join(MODEL_PATH, 'scaler.joblib'))
    else:
        # Load the encoder, imputer, and scaler
        one_hot_encoder = joblib.load(os.path.join(MODEL_PATH,'one_hot_encoder.joblib'))
        imputer = joblib.load(os.path.join(MODEL_PATH, 'imputer.joblib'))
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))
        # One Hot Encoding
        df_one_hot_encoded = one_hot_encoder.transform(df[features_to_one_hot_encode])
        # Ordinal Encoding
        kitchen_quality_dict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}
        df['KitchenQual'] = df['KitchenQual'].apply(lambda x: kitchen_quality_dict[x])
        # Impute missing values
        df[continuous_features] = imputer.transform(df[continuous_features])
        # Scaling
        df_scaled = scaler.transform(df[continuous_features])

    # Convert encoded and scaled data back to DataFrame
    df_one_hot_encoded_df = pd.DataFrame(df_one_hot_encoded, 
                                         columns=one_hot_encoder.get_feature_names_out())
    df_scaled_df = pd.DataFrame(df_scaled, columns=continuous_features)

    # Combine all processed features
    df_processed = pd.concat([df_one_hot_encoded_df, 
                              df[features_to_ordinal_encode], df_scaled_df], axis=1)

    return df_processed
