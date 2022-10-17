import argparse
import json
from models.UCPR.utils import *
import wandb
import sys
import numpy as np
import shutil
import os
from sklearn.model_selection import ParameterGrid
import subprocess
TRAIN_FILE_NAME = 'preprocess/train_transe.py'

def load_metrics(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        metrics = json.load(f)
    return metrics
def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f)
def save_cfg(configuration, filepath):
    with open(filepath, 'w') as f:
        json.dump(configuration, f)     
def metrics_average(metrics):
    avg_metrics = dict()
    for k, v in metrics.items():
        avg_metrics[k] = sum(v)/max(len(v),1)
    return avg_metrics


def save_best(best_metrics, test_metrics, grid):
    dataset_name = grid["dataset"]
    if best_metrics is None:
        save_metrics(test_metrics, f'{BEST_TRANSE_TEST_METRICS_FILE_PATH[dataset_name]}')
        save_cfg(grid, f'{BEST_TRANSE_CFG_FILE_PATH[dataset_name] }')
        shutil.rmtree(BEST_CFG_DIR[dataset_name])
        shutil.copytree(TMP_DIR[dataset_name], BEST_CFG_DIR[dataset_name] )
        return 


    if best_metrics[TRANSE_OPT_METRIC] > test_metrics[TRANSE_OPT_METRIC]:
        save_metrics(test_metrics, f'{BEST_TRANSE_TEST_METRICS_FILE_PATH[dataset_name]}')
        save_cfg(grid, f'{BEST_TRANSE_CFG_FILE_PATH[dataset_name] }')
        shutil.rmtree(BEST_CFG_DIR[dataset_name])
        shutil.copytree(TMP_DIR[dataset_name], BEST_CFG_DIR[dataset_name] )


def makedirs(dataset_name):
    os.makedirs(BEST_CFG_DIR[dataset_name], exist_ok=True)
    os.makedirs(CFG_DIR[dataset_name], exist_ok=True)



def main(args):


    chosen_hyperparam_grid = {'batch_size': [64],
         'dataset': ['lfm1m','ml1m'],
         'embed_size': [100],
         'epochs': [1],
         'gpu': ['0'],
         'l2_lambda': [0],
         'lr': [0.5],
         'max_grad_norm': [5.0],
         'name': ['train_transe_model'],
         'num_neg_samples': [5],
         'seed': [123],
         'steps_per_checkpoint': [200],
         'weight_decay': [0]}
    hparam_grids = ParameterGrid(chosen_hyperparam_grid)
    print('num_experiments: ', len(hparam_grids))

    for configuration in hparam_grids:
        dataset_name = configuration["dataset"]
        makedirs(dataset_name)
        if args.wandb:
            wandb.init(project=f'{MODEL_NAME}_TRANSE_{dataset_name}',
                           entity=args.wandb_entity, config=configuration)    
        #'''
        CMD = ["python3", TRAIN_FILE_NAME]

        for k,v in configuration.items():
            CMD.extend( [f'--{k}', f'{v}'] )
        print('Executing job: ',configuration)
        subprocess.call(CMD)
        
        print(TRANSE_CFG_FILE_PATH[dataset_name])
        save_cfg(configuration, TRANSE_CFG_FILE_PATH[dataset_name])        
        test_metrics = load_metrics(TRANSE_TEST_METRICS_FILE_PATH[dataset_name])
        best_metrics = load_metrics(BEST_TRANSE_TEST_METRICS_FILE_PATH[dataset_name])
        save_best(best_metrics, test_metrics, configuration)
    
        if args.wandb:
            wandb.log(test_metrics[TRANSE_OPT_METRIC])





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")
    parser.add_argument(
        "--wandb_entity",
        required="--wandb" in sys.argv,
        type=str,
        help="Entity name to push to the wandb logged data, in case args.wandb is specified.",
    )    
    args = parser.parse_args()
    main(args)
                 
    