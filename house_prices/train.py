import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from house_prices.preprocess import preprocess_data

def build_model(data: pd.DataFrame) -> dict:
    # Select useful features
    label_col = 'SalePrice'
    useful_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'ExterQual']
    continuous_features = ['GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'YearRemodAdd']
    ordinal_feature = 'ExterQual'
    exterior_quality_dict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}

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
    
    # Save model and transformers
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return {'rmse': rmse}
