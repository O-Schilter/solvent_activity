import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from solvent_activity.bert_utils.dataset import RegDataset
from solvent_activity.bert_utils.tokenization import (
    BasicSmilesTokenizer,
    SmilesTokenizer,
)


def evaluate_model(y_pred, y_val, scaler):
    """Evaluate the model and return metrics."""
    if scaler:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_val = scaler.inverse_transform(y_val).flatten()

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



def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_scaler(scaler_key):
    scaler_mapping = {
        "y_minmax": "./data/splits/split_0/min_max_scaler.pkl",
        "y_standard": "./data/splits/split_0/standard_scaler.pkl",
        "Literature": False,
    }

    scaler_path = scaler_mapping.get(scaler_key)
    if scaler_path:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return scaler
    else:
        return False


def eval_baseline(args):
    def convert_to_list(value):
        if isinstance(value, str):
            return [float(i) for i in value.strip("[]").split(",")]
        else:
            return value

    converters = {
        "conc_morgan_fp": convert_to_list,
        "conc_torsion_fp": convert_to_list,
        "conc_descirptors": convert_to_list,
        "conc_morfeus_descirptors": convert_to_list,
    }

    df_test = pd.read_csv(args.test_file, converters=converters)
    models_folder = args.baseline_models_folder
    baseline_metrics = []

    for model_file in os.listdir(models_folder):
        if model_file.endswith(".pkl"):
            # Parse the model file name
            match = re.match(r"^(.*)-(.*)-(.*)\.pkl$", model_file)
            if not match:
                continue

            model_architecture, feature_column, scaler_key = match.groups()
            model_path = os.path.join(models_folder, model_file)
            model = load_model(model_path)
            scaler = load_scaler(scaler_key)
            X_test = df_test[feature_column]
            y_test = df_test["Literature"].values.reshape(-1, 1)
            if scaler:
                y_test = scaler.transform(y_test)

            y_pred = model.predict(X_test.to_list())

            eval_metrics = evaluate_model(y_pred, y_test, scaler)
            eval_metrics["model_architecture"] = model_architecture
            eval_metrics["feature_column"] = feature_column
            eval_metrics["scaler"] = scaler_key

            baseline_metrics.append(eval_metrics)
    return baseline_metrics


def main(args):
    baseline_metrics = eval_baseline(args)
    eval_metrics_bert = eval_bert(args)
    baseline_metrics.append(eval_metrics_bert)
    df_result = pd.DataFrame(baseline_metrics)
    df_result.to_csv("metrics_on_test.csv")


def eval_bert(args):

    tokenizer = SmilesTokenizer(
        vocab_file=args.bert_vocab_path,
        basic_tokenizer=BasicSmilesTokenizer(),
        remove_mapping=True,
    )
    # Load train and test set
    print("Loading data from:", args.test_file)
    df_test = pd.read_csv(args.test_file)

    # Build iterable dataset
    test_set = RegDataset(df_test[args.rxn_column], df_test[args.objective], tokenizer)
    reg_model = BertForSequenceClassification.from_pretrained(
        args.bert_model_path, num_labels=1
    )
    reg_model.eval()
    print(reg_model)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    pred_ls = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            pred = (
                reg_model(
                    batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
                )["logits"]
                .cpu()
                .numpy()
            )
            pred_ls.extend(pred.tolist())

    # Evaluate the model

    scaler = load_scaler("y_standard")
    y_test = df_test["Literature"].values.reshape(-1, 1)
    if scaler:
        y_test = scaler.transform(y_test)

    eval_metrics = evaluate_model(
        np.array([i[0] for i in pred_ls]), y_test, scaler=scaler
    )
    eval_metrics["model_architecture"] = "bert"
    eval_metrics["feature_column"] = "smiles"
    eval_metrics["scaler"] = "y_standard"
    return eval_metrics


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert_model_path",
        type=str,
        default="/Users/oli/projects/solvent_activity/reg_models_standard/checkpoint-15000",
    )
    parser.add_argument("--baseline_models_folder", type=str, default="./models/")

    parser.add_argument(
        "--test_file", type=str, default="./data/splits/test_database_IAC_ln_clean.csv"
    )
    parser.add_argument(
        "--bert_vocab_path", type=str, default="./data/splits/vocab.txt"
    )


    parser.add_argument("--rxn_column", type=str, default="solvent_solute_smiles")
    parser.add_argument("--objective", type=str, default="Literature")
    parser.add_argument("--scaling", action="store_true")

    args = parser.parse_args()

    main(args)
