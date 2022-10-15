# T-REX_TextualRecEXp

# Usage
The base workflow is:
1. Dataset creation
2. Dataset **reprocessing** and **formatting** with respect to selected model
3. Training(or hyperparameter search) of the **TransE embedding model**
4. Training(or hyperparameter search) of the **Recommender model**
5. Test of the quality of the recommender model

## Dataset creation
For each dataset run from the top level folder
```bash
export DATASET_NAME=...
python3 map_dataset.py --data $DATASET_NAME --model MODEL_NAME
```

Available DATASET_NAME values **{lfm1m, ml1m}**
Available MODEL_NAME values **{cafe, pgpr, ucpr}**
## Dataset preprocessing and formatting
Then, each dataset has to be processed and formatted according to the specifications of each model.
To achieve this, run from within each folder models/MODEL_NAME
```bash
python3 preprocess.py --data $DATASET_NAME
```
## Train embedding model
```bash
python3 train_transe.py --dataset $DATASET_NAME
```
### (Optional) Hyperparameter optimization of the embedding model 
In order to properly track all configurations, additional logging functionalities are provided by means of wandb.
To perform wandb logging, use the below arguments, otherwise run transe_gridsearch.py without any additional command line arguments.
```bash
python3 transe_gridsearch.py --wandb --wandb_entity YOUR_WANDB_ACCOUNT_NAME 
```
Note:
In order to use wandb, you have to already be logged in from the command line.
It can be done by simply running, with your API_KEY obtained by register for free to their service.
```bash
wandb login [OPTIONS] [KEY]
```
## Train Recommender
```bash
python3 train.py --dataset $DATASET_NAME 
```
#### (Optional)Hyperparameter optimization of the recommender model 
```bash
python3 gridsearch.py --dataset $DATASET_NAME 
```

## Test 
```bash
python3 test.py --dataset $DATASET_NAME 
```
