# Solvent Activity

This repository contains code for processing data, training baseline models, and training BERT models for predicting solvent activity.

## Setup

1. Clone the repository:
```
git clone https://github.com/O-Schilter/solvent_activity.git
```
Create a conda environment and install the required packages:
```
conda create -n solvent-activity python=3.10
conda activate solvent-activity
pip install -r requirements.txt
```
## Data Processing

To create the data for training and evaluation, run the following script:
```
python ./solvent_activity/data_processing.py
```
This script will process the raw data and create the necessary files in the `./data/splits/` folder for the subsequent steps.


## Baseline Models
To train the baseline regression models, run the following script:
```
python ./solvent_activity/reg_baseline.py
```

This script will train and save the baseline models in the `./models/` folder. As in this step the hyperparameter optimization takes part it requires some time to be run.

## BERT Models

### Creating Vocabulary
Before training the BERT models, you need to create a vocabulary file. Run the following script:
```
python ./solvent_activity/create_vocab.py
```
This script will create a vocabulary file based on the dataset in the file defined `OUTPUT_PATH` which is by default the `./data/splits` folder

### Training BERT for Masked Language Modeling (MLM)
To train the BERT model for the Masked Language Modeling (MLM) task, run the following script with the appropriate arguments:
```
python ./solvent_activity/mlm_bert.py --output_path "outs/" --vocab_path "./data/splits/vocab.txt" --data_path "./data/splits/split_0"
```
The available arguments are:

--output_path: Path to save the trained model (default: "outs/")
--vocab_path: Location of the vocabulary file (default: ".")
--data_path: Directory to read the parquet data files from (default: ".")

### Fine-tuning BERT for Regression
After training the BERT model for MLM, you can fine-tune it for the regression task by running the following script with the appropriate arguments:
```
python /solvent_activity/reg_bert.py  --model_path "./outs/" --output_path "./reg_models/" --vocab_path "./data/splits/vocab.txt" --data_path  "./data/splits/split_0"
```
The available arguments are:

--model_path: Path to the pre-trained BERT model (default: "./")
--output_path: Path to save the fine-tuned model (default: "./")
--vocab_path: Path to the vocabulary file (default: "./")
--data_path: Path to the data directory (default: "./")

## Evaluation
To evaluate the trained models on the holdout test set, run the following script:
```
./solvent_activity/eval_on_test.py
```

This script will load the trained baseline models and evaluate their performance on the test set.