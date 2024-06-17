import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


class CV:
    def __init__(self, df, num_splits):
        """
        Initializes the CV class with the given DataFrame and number of splits.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            num_splits (int): The number of splits for cross-validation.
        """
        self.df = df
        self.num_splits = num_splits

    def load_index(self):
        """
        Loads the training and validation indices for each fold from a CSV file.

        This method populates the train_index_ls and val_index_ls attributes
        with the indices of the training and validation sets for each split.
        """
        self.train_index_ls = []
        self.val_index_ls = []

        for split_index in range(self.num_splits):
            df_train, df_val = load_data(split_index)
            train_idx = df_train["Unnamed: 0"].to_numpy()
            val_idx = df_val["Unnamed: 0"].to_numpy()
            train_indices = self.df.index[self.df["Unnamed: 0"].isin(train_idx)].tolist()
            val_indices = self.df.index[self.df["Unnamed: 0"].isin(val_idx)].tolist()
            self.train_index_ls.append(train_indices)
            self.val_index_ls.append(val_indices)

    def __iter__(self):
        """
        Iterates over the splits, yielding a tuple of training and validation indices.

        Yields:
            tuple: A tuple containing two lists - training indices and validation indices
                   for each split.
        """
        for split_index in range(self.num_splits):
            yield self.train_index_ls[split_index], self.val_index_ls[split_index]

    def __len__(self):
        """
        Returns the number of splits.

        Returns:
            int: The number of splits for cross-validation.
        """
        return self.num_splits


def get_hyperparameter_distributions(model_name):
    """Define hyperparameter distributions for each model."""
    params = {}
    if model_name == "RandomForest":
        params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    elif model_name == "XGBoost":
        params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
    elif model_name == "GaussianProcess":
        params = {"alpha": [1e-10, 1e-8]}
    elif model_name == "LinearRegression":
        params = {"fit_intercept": [True, False]}
    elif model_name == "Lasso":
        params = {"alpha": [0.1, 0.5, 1.0, 1.5, 2.0]}
    elif model_name == "Ridge":
        params = {"alpha": [0.1, 0.5, 1.0, 1.5, 2.0]}
    return params


def load_data(split_index):
    """Load training and validation data."""
    df_train = pd.read_parquet(
        f"./data/splits/split_{split_index}/train_database_IAC_ln_clean.parquet",
        engine="fastparquet",
    )
    df_val = pd.read_parquet(
        f"./data/splits/split_{split_index}/val_database_IAC_ln_clean.parquet",
        engine="fastparquet",
    )
    return df_train, df_val


def load_scaler(scaling):
    """Load the standard scaler for a given split."""
    if scaling == "y_standard":
        scaler_name = "standard_scaler"
    elif scaling == "y_minmax":
        scaler_name = "min_max_scaler"
    elif scaling == "Literature":
        return False

    with open(f"./data/splits/split_0/{scaler_name}.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler


def save_model(model, model_name, feature_col, scaling):
    """Save the trained model and its parameters."""
    model_path = f"models/{model_name}-{feature_col}-{scaling}.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save model parameters to JSON
    params_path = f"models/{model_name}-{feature_col}-{scaling}_params.json"
    with open(params_path, "w") as f:
        json.dump(model.get_params(), f)


def evaluate_model(model, X_val, y_val, scaler):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_val)
    if scaler:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_val = scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
        
    #if exponential function fails, write large or small values
    try:
        mse_exp = mean_squared_error(np.exp(y_val), np.exp(y_pred))
        r2_exp = r2_score(np.exp(y_val), np.exp(y_pred))
        mae_exp = mean_absolute_error(np.exp(y_val), np.exp(y_pred))
    except:
        mse_exp = 1e6
        r2_exp = -1e6
        mae_exp =  1e6
        
    return {
        "MSE": mean_squared_error(y_val, y_pred),
        "R2": r2_score(y_val, y_pred),
        "MAE": mean_absolute_error(y_val, y_pred),
        "MSE_exp": mse_exp,
        "R2_exp": r2_exp,
        "MAE_exp": mae_exp,
    }


def save_metrics(metrics, model_name, feature_col, scaling):
    """Save metrics to a CSV file."""
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.sort_values(by="mean_test_R2", ascending=False)
    
    # Add descriptors and scaling attributes.
    metrics_df["model_name"] = model_name
    metrics_df["feature_col"] = feature_col
    metrics_df["scaling"] = scaling

    metrics_path = f"models/{model_name}-{feature_col}-{scaling}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    best_row = metrics_df.head(1)
    return best_row


def optimize_models(model_name, model):
    # Runs Hyperparameter optimization
    best_rows = []
    
    # get hyperparameters
    params = get_hyperparameter_distributions(model_name)

    # loads and combines all data
    df_train, df_val = load_data(1)
    df = pd.concat([df_train, df_val])
    
    # CV class returns the index of all 5 cross validation Splits
    cv_index_loader = CV(df, 5)
    cv_index_loader.load_index()

    # Itterates over all molecular descriptors/FP
    for feature_col in [
        "conc_morgan_fp",
        "conc_torsion_fp",
        "conc_descirptors",
        "conc_morfeus_descirptors",
    ]:
        # itterating over all three scalings, Literature equals unscaled 
        for scaling in ["y_minmax", "Literature", "y_standard"]:
            
            if scaling in ["y_minmax", "y_standard"]:
                scaler = load_scaler(scaling)
            else:
                scaler = False
                
            # Construct Gridsearch and fits over data
            grid = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=cv_index_loader,
                refit="R2",
                scoring=lambda estimator, X, y: evaluate_model(estimator, X, y, scaler),
                verbose=3,
                n_jobs=-1,
            )

            x_input = np.stack(df[feature_col].values)

            grid.fit(x_input, df[scaling])

            best_model = grid.best_estimator_
            metrics = grid.cv_results_

            # Save the best model and its parameters
            best_row = save_metrics(metrics, model_name, feature_col, scaling)
            save_model(best_model, model_name, feature_col, scaling)
            best_rows.append(best_row)
    return best_rows


if __name__ == "__main__":
    best_models = []
    best_models.extend(optimize_models("XGBoost", xgb.XGBRegressor()))

    best_models.extend(optimize_models("GaussianProcess", GaussianProcessRegressor()))
    best_models.extend(optimize_models("Ridge", Ridge()))
    best_models.extend(optimize_models("LinearRegression", LinearRegression()))
    best_models.extend(optimize_models("Lasso", Lasso()))
    best_models.extend(optimize_models("RandomForest", RandomForestRegressor()))

    df_results = pd.concat(best_models)
    print(df_results)

    df_results.to_csv("./models/models_metrics.csv")
