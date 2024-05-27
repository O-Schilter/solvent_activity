import torch
from torch import nn
import pickle
import pandas as pd
import numpy as np

import evaluate
import argparse
from tokenization import SmilesTokenizer, BasicSmilesTokenizer
from transformers import TrainingArguments, Trainer


from rdkit import RDLogger

from dataset import RegDataset
from transformers import BertForSequenceClassification
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
metric = evaluate.load("accuracy")


def metrics_regression(eval_pred):
    y_pred, y_true = eval_pred
    
    if scaler:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1))

    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    mse_ = mean_squared_error(
        y_true=y_true, y_pred=y_pred, multioutput='raw_values')
    r2_ = r2_score(y_true, y_pred, multioutput='raw_values')
    mae_ = mean_absolute_error(y_true, y_pred, multioutput='raw_values')

    metrics = {'mse': mse,
               'r2': r2,
               'mae': mae}

    for i in range(len(mse_)):
        metrics['mse_'+str(i)] = mse_[i]
        metrics['r2_'+str(i)] = r2_[i]
        metrics['mae_'+str(i)] = mae_[i]

    return metrics



def main(args):
    SMILES_VOCAB_FILE = args.vocab_path

    print('Using old tokenizer with no atom maps')
    tokenizer = SmilesTokenizer(
        vocab_file=SMILES_VOCAB_FILE, basic_tokenizer=BasicSmilesTokenizer(),
        remove_mapping=True
    )
    RDLogger.DisableLog('rdApp.*')


    # Load train and test set
    print("Loading data from:", args.data_path)
    df_train = pd.read_parquet(
        args.data_path + "train_database_IAC_ln_clean.parquet", engine="fastparquet"
    )
    df_val = pd.read_parquet(
        args.data_path + "val_database_IAC_ln_clean.parquet", engine="fastparquet"
    )

    

    # if args.scaling:
    #     df_train[args.objective] = scaler.fit_transform(df_train[args.objective].values.reshape(-1, 1))
    #     df_val[args.objective] = scaler.transform(df_val[args.objective].values.reshape(-1, 1))
        
        
    # Build iterable dataset
    training_set = RegDataset(
        df_train[args.rxn_column], df_train[args.objective], tokenizer)
    
    val_set = RegDataset(df_val[args.rxn_column],
                         df_val[args.objective], tokenizer)

    reg_model = BertForSequenceClassification.from_pretrained(
        args.model_path, num_labels=1)

    reg_model.dropout = nn.Dropout(0.2)
    reg_model.config.classifier_dropout = 0.2
    reg_model.config.problem_type = "regression"
    reg_model.num_labels = 1
    # reg_model.classifier = torch.nn.Linear(reg_model.config.hidden_size, len(args.objective))
    
    training_args = TrainingArguments(
        report_to= "tensorboard",
        learning_rate=args.learning_rate,
        output_dir=args.output_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_first_step=True,
        do_train=True,
        do_eval=True,
        logging_steps=50,
        save_steps=5000,
        eval_steps=500,
        warmup_steps=200,
        overwrite_output_dir=True,
        save_total_limit=2,
        fp16=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # resume_from_checkpoint='./models/solvent_classification_USPTO_suzuki/checkpoint-509000/',
        # model_name_or_path="path/to/checkpoint"
        num_train_epochs=100,
        use_mps_device=True
        # max_steps = 250
    )

    trainer = Trainer(
        model=reg_model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=val_set,
        compute_metrics=metrics_regression
    )
    
    print(reg_model.config.problem_type)
    print(args.base_name)
    trainer.train()


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", type=str, default="")
    parser.add_argument('--model_path', type=str, default='./')
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--vocab_path", type=str, default="./")
    parser.add_argument("--data_path", type=str, default="./")
    parser.add_argument("--rxn_column", type=str, default="solvent_solute_smiles")
    parser.add_argument('--objective', type=str, default='Literature')
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--mode", default="new_maps", help="options are new_maps, old_maps, no_maps")

    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    global scaler 
    scaler = False
    if args.scaling:
        print('is scaled')
        scaler = pd.read_pickle(args.data_path+ 'standard_scaler.pkl')
    
    main(args)