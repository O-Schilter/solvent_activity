import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle

def load_data(split_index):
    """Load training and validation data."""
    df_train = pd.read_parquet(f'../data/splits/split_{split_index}/train_database_IAC_ln_clean.parquet', engine='fastparquet')
    df_val = pd.read_parquet(f'../data/splits/split_{split_index}/val_database_IAC_ln_clean.parquet', engine='fastparquet')
    return df_train, df_val

def load_scaler(split_index):
    """Load the standard scaler for a given split."""
    with open(f'../data/splits/split_{split_index}/standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def prepare_features(df_train, df_val, feature_col):
    """Prepare features from train and validation set."""
    return np.stack(df_train[feature_col].values), np.stack(df_val[feature_col].values)

def train_model(model, X_train, y_train_scaled):
    """Train a model."""
    model.fit(X_train, y_train_scaled)
    return model

def evaluate_model(model, X_val, y_val, scaler):
    """Evaluate the model and return metrics."""
    y_pred_scaled = model.predict(X_val)
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    return {
        'MSE': mean_squared_error(y_val, y_pred_unscaled),
        'R2': r2_score(y_val, y_pred_unscaled),
        'MAE': mean_absolute_error(y_val, y_pred_unscaled),
    }

def run_models():
    results = []
    for i in range(1, 6):
        df_train, df_val = load_data(i)
        scaler = load_scaler(i)
        y_train, y_val = df_train['Literature'], df_val['Literature']
        y_train_scaled = scaler.transform(y_train.values.reshape(-1, 1)).flatten()

        
        for feature_col in ['conc_morgan_fp', 'conc_torsion_fp', 'conc_descirptors']:
            X_train, X_val = prepare_features(df_train, df_val, feature_col)
            
            rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            trained_rf = train_model(rf_regressor, X_train, y_train_scaled)
            rf_metrics = evaluate_model(trained_rf, X_val, y_val, scaler)
            
            xgb_regressor = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1)
            trained_xgb = train_model(xgb_regressor, X_train, y_train_scaled)
            xgb_metrics = evaluate_model(trained_xgb, X_val, y_val, scaler)
            
            results.append({
                'Split': i,
                'Feature': feature_col,
                'Model': 'RandomForest',
                'MSE': rf_metrics['MSE'],
                'R2': rf_metrics['R2'],
                'MAE': rf_metrics['MAE']
            })
            results.append({
                'Split': i,
                'Feature': feature_col,
                'Model': 'XGBoost',
                'MSE': xgb_metrics['MSE'],
                'R2': xgb_metrics['R2'],
                'MAE': xgb_metrics['MAE']
            })
    
    results_df = pd.DataFrame(results)
    average_metrics = results_df.groupby(['Model', 'Feature']).mean().drop(columns=['Split'])

    return results_df, average_metrics


if __name__ == "__main__":
    model_results, average_metrics = run_models()
    print(model_results)
    print(average_metrics)
