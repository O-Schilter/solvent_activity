import numpy as np
import pandas as pd
from tqdm import tqdm

from solvent_activity.bert_utils.tokenization import BasicSmilesTokenizer

LS_SPECIAL_TOKENS = [
    "[PAD]",
    "[unused1]",
    "[unused2]",
    "[unused3]",
    "[unused4]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]


def save_vocab(vocab_tokens, special_tokens, file_path):
    with open(file_path, "w") as f:
        for token in special_tokens:
            f.write(token)
            f.write("\n")

        for token in vocab_tokens:
            f.write(token)
            f.write("\n")
    print("vocab save as:", file_path)
    print("# vocab tokens:", len(vocab_tokens))


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "./data/splits/split_1/train_database_IAC_ln_clean.parquet"

    OUTPUT_PATH = "./data/splits/"

    SMILES_COLUMN = "solvent_solute_smiles"

    # load data
    df_data = pd.read_parquet(DATA_PATH)

    smiles = df_data[SMILES_COLUMN].to_list()

    tokenizer = BasicSmilesTokenizer(remove_maps=True)

    tokens = np.unique(np.concatenate([tokenizer.tokenize(rx) for rx in tqdm(smiles)]))

    save_vocab(tokens, LS_SPECIAL_TOKENS, OUTPUT_PATH + "vocab.txt")
