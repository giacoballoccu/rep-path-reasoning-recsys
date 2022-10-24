# T-REX_TextualRecEXp


# Overview
The base workflow is:
1. Dataset creation
2. Dataset **preprocessing** and **formatting** with respect to the selected model
3. Training(or hyperparameter search) of the **TransE embedding model**
4. Training(or hyperparameter search) of the **Recommender model**
5. Test of the quality of the recommender model
6. Evaluation of the metrics

## Dependencies and Installation
Required **pytorch** and **tensorflow**, both with gpu version.
Clone the repo, enter into it.
All other dependencies can then be installed by means of the env_setup.sh script located at the top level of the project
```bash
bash env_setup.sh
```
The above environment setup includes download of the datasets and their preliminary setup.

Alternatively, the data can be downloaded from the following drive:
```bash
https://drive.google.com/file/d/1VUVkU1RLaJWUVqReox6cT6N9dcQ6nTd7/view?usp=sharing
```
The setup also downloads automatically the pretrained weights for the embeddings.

# Usage
To **facilitate the use of the codebase**, a set of scripts are provided to seamlessly execute all stages of the machine learning pipeline described above.
The top level of the project contains all such scripts.

### 1. Bulk dataset creation
```bash
./build_datasets.sh
```
### 2. Bulk dataset preprocessing
```python
python3 prepare_datasets.py
```
### 3. Models
Available DATASET_NAME values **{lfm1m, ml1m}**
Available MODEL_NAME values:
- path based methods **{cafe, pgpr, ucpr}**
- knowledge aware methods **{kgat, cfkg, cke}**
- embedding methods **{transe}**

The following variables can be used. Alternatively the provided dataset and model names can be manually written as command line arguments to the scripts that follow (thus substituting the $x expression with the actual name)
```bash
export DATASET_NAME=...
export MODEL_NAME=...
```
#### 3.1 Training

```python
python3 train.py --model $MODEL_NAME --dataset $DATASET_NAME
```
#### 3.2 Hyper parameter tuning

```python
python3 gridsearch.py --model $MODEL_NAME --dataset $DATASET_NAME
```

#### 4. Test
Note, the test phase does not support testing of the embedding models, although their metrics are gathered during training.
The main purpose of the test phase is to evaluate the metrics on each embedding model.
```python
python3 test.py --model $MODEL_NAME --dataset $DATASET_NAME
```


#### 5. Evaluation
As last step, with the computed top10 or the paths predicted by the models, evaluation can be run to obtain the evaluation metrics.

Note that this step requires training of the models and at least a testing phase for path reasoning models.

This is because the path-based ones require pre-computation of paths before the topK items can be obtained. 

After paths are computed, creation of the topK is computationally inexpensive.

Knowledge aware models generate the topK by default after each epoch of training.

```python
python3 evaluate.py --model $MODEL_NAME --dataset $DATASET_NAME
```


#### 6. Hyper parameters
# lfm1m
## Ucpr
## pgpr
## cafe

# ml1m
## kgat 
## cke
## cfkg
