import argparse

import pandas as pd
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from solvent_activity.bert_utils.dataset import MlmDataset
from solvent_activity.bert_utils.tokenization import (
    BasicSmilesTokenizer,
    SmilesTokenizer,
)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def main(args):
    tokenizer = SmilesTokenizer(
        vocab_file=args.vocab_path,
        basic_tokenizer=BasicSmilesTokenizer(),
        remove_mapping=True,
    )

    # Load train and test set
    print("Loading data from:", args.data_path)
    df_train = pd.read_parquet(
        args.data_path + "train_database_IAC_ln_clean.parquet", engine="fastparquet"
    )
    df_val = pd.read_parquet(
        args.data_path + "val_database_IAC_ln_clean.parquet", engine="fastparquet"
    )

    # Build iterable dataset
    training_set = MlmDataset(df_train[args.rxn_column], tokenizer)
    val_set = MlmDataset(df_val[args.rxn_column], tokenizer)

    config = BertConfig()

    config.intermediate_size = 512
    config.hidden_size = 256
    config.num_attention_heads = 4
    config.num_hidden_layers = 12
    config.hidden_dropout_prob = args.dropout
    config.vocab_size = tokenizer.vocab_size
    print(config)

    mlm_model = BertForMaskedLM(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        # pad_to_multiple_of=8 if pad_to_multiple_of_8 else None
    )

    logging_steps = 100  # how often to log to wandb
    save_steps = 5000  # nb update steps before 2 checkpoint saves
    eval_steps = 20000  # nb update steps between 2 evaluations
    warmup_steps = 10000  # nb steps of linear warmup from 0 to learning_rate

    run_name = (
        f"lr_{args.learning_rate}_bs_{args.batch_size}_nep_{str(args.train_epochs)}"
    )

    training_args = TrainingArguments(
        report_to="tensorboard",
        run_name=run_name,
        learning_rate=args.learning_rate,
        output_dir=args.output_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_first_step=True,
        do_train=True,
        do_eval=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        overwrite_output_dir=True,  # careful
        save_total_limit=2,
        fp16=False,
        dataloader_num_workers=args.num_cpus,
        dataloader_pin_memory=True,
        # model_name_or_path="path/to/checkpoint"
        num_train_epochs=args.train_epochs,
        # use_cpu=torch.cuda.is_available(),
        use_mps_device=True,  # on mac , should be default
    )

    print(f"Training args {training_args}")

    trainer = Trainer(
        model=mlm_model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=val_set,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outs/")

    parser.add_argument(
        "--vocab_path", type=str, default=".", help="location of vocab file to use"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=".",
        help="data directory to read parquet files from",
    )
    parser.add_argument("--rxn_column", type=str, default="solvent_solute_smiles")

    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--train_epochs", type=int, default=250)
    parser.add_argument("--num_cpus", type=int, default=1)

    args = parser.parse_args()
    main(args)
