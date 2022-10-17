import argparse
import json
from models.PGPR.utils import *
import wandb
import sys
import numpy as np
import shutil
import os
from sklearn.model_selection import ParameterGrid
import subprocess
TRAIN_FILE_NAME = 'train_agent.py'
TEST_FILE_NAME = 'test_agent.py'

def load_metrics(filepath):
    with open(filepath) as f:
        metrics = json.load(f)
    return metrics
def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f)
def save_cfg(configuration, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f)     
def metrics_average(metrics):
    avg_metrics = dict()
    for k, v in metrics.items():
        avg_metrics[k] = sum(v)/max(len(v),1)
    return avg_metrics

def save_best(best_metrics, test_metrics, grid):
    dataset_name = grid["dataset"]

    if test_metrics[OPTIM_HPARAMS_METRIC] > best_metrics[OPTIM_HPARAMS_METRIC]:
        save_metrics(test_metrics, f'{BEST_TEST_METRICS_FILE_PATH[dataset_name]}')
        save_cfg(grid, f'{BEST_CFG_FILE_PATH[dataset_name] }')
        shutil.rmtree(BEST_CFG_DIR[dataset_name])
        shutil.copytree(TMP_DIR[dataset_name], BEST_CFG_DIR[dataset_name] )





def main(args):


    chosen_hyperparam_grid = {"act_dropout": [0], 
    "batch_size": [32], 
    "dataset": ["lfm1m", "ml1m"], 
    "do_validation": [True], 
    "ent_weight":[ 0.001], 
    "epochs": [50], 
    "gamma": [0.99], 
    "gpu": ["0"], 
    "hidden": [[512, 256]], 
    "lr": [0.0001], 
    "max_acts": [250], 
    "max_path_len": [3], 
    "name": ["train_agent"], 
    "seed": [123], 
    "state_history": [1], 
    "wandb": [True], 
    "wandb_entity": ['t-rex-recom']}

    test_args ={'dataset','seed','gpu','epochs','max_acts','max_acts', 'max_path_len','gamma','state_history',
                'hidden','add_products','top_k','run_path', 'run_eval', 'save_paths'}




    hparam_grids = ParameterGrid(chosen_hyperparam_grid)
    print('num_experiments: ', len(hparam_grids))

    for configuration in hparam_grids:
        dataset_name = configuration["dataset"]
        makedirs(dataset_name)
        if args.wandb:
            wandb.init(project=f'{MODEL_NAME}_{dataset_name}',
                           entity=args.wandb_entity, config=configuration)    
            
         

        CMD = ["python3", TRAIN_FILE_NAME]

        for k,v in configuration.items():
                if k == 'wandb':
                    CMD.extend([f'--{k}'])
                elif isinstance(v,list):
                    cmd_args = [f'--{k}'] + [f" {val} " for val in v]
                    CMD.extend( cmd_args )
                else:
                    CMD.extend( [f'--{k}', f'{v}'] )            
           
        print('Executing job: ',configuration)
        subprocess.call(CMD)
        
        # cafe and ucpr have the same command line args, pgpr does not, so the call below will have to be 
        # modified accordingly
        print('Done training, testing phase')
        CMD = ["python3", TEST_FILE_NAME]
        for k,v in configuration.items():
            if k in test_args:
                if k == 'wandb':
                    CMD.extend([f'--{k}'])
                elif isinstance(v,list):
                    cmd_args = [f'--{k}'] + [f" {val} " for val in v]
                    CMD.extend( cmd_args )
                else:
                    CMD.extend( [f'--{k}', f'{v}'] )
        subprocess.call(CMD)

        save_cfg(configuration, CFG_FILE_PATH[dataset_name])        
        test_metrics = load_metrics(TEST_METRICS_FILE_PATH[dataset_name])
        best_metrics = load_metrics(BEST_TEST_METRICS_FILE_PATH[dataset_name])
        save_best(best_metrics, test_metrics, configuration)
    
        if args.wandb:
            wandb.log(test_metrics)





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
                 
    