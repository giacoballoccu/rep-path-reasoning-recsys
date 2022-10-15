import argparse
import json
from UCPR.utils import *
import wandb
import sys
import numpy as np
import shutil
import os

TRAIN_FILE_NAME = 'src/train.py'
TEST_FILE_NAME = 'src/test.py'

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
    best_avg = metrics_average(best_metrics)
    avg = metrics_average(test_metrics)

    if avg[OPTIM_HPARAMS_METRIC] > best_avg[OPTIM_HPARAMS_METRIC]:
        save_metrics(test_metrics, f'{BEST_TEST_METRICS_FILE_PATH[dataset_name]}')
        save_cfg(grid, f'{BEST_CFG_FILE_PATH[dataset_name] }')
        shutil.rmtree(BEST_CFG_DIR[dataset_name])
        shutil.copytree(TMP_DIR[dataset_name], BEST_CFG_DIR[dataset_name] )





def main(args):


    chosen_hyperparam_grid = {'batch_size': 64,
         'dataset': 'lfm1m',
         'embed_size': 100,
         'epochs': 30,
         'gpu': '0',
         'l2_lambda': 0,
         'lr': 0.5,
         'max_grad_norm': 5.0,
         'name': 'train_transe_model',
         'num_neg_samples': 5,
         'seed': 123,
         'steps_per_checkpoint': 200,
         'weight_decay': 0}
    hparam_grids = ParameterGrid(chosen_hyperparam_grid)
    print('num_experiments: ', len(hparam_grids))

    for grid in hparam_grids:
        dataset_name = grid["dataset"]
        makedirs(dataset_name)
        if args.wandb:
            wandb.init(project=f'{MODEL_NAME}_{dataset_name}',
                           entity=args.wandb_entity, config=grid)    
            
         

        CMD = ["python3", TRAIN_FILE_NAME]

        for k,v in grid.items():
            CMD.extend( [f'--{k}', f'{v}'] )
        print('Executing job: ',grid)
        call(CMD)
        
        # cafe and ucpr have the same command line args, pgpr does not, so the call below will have to be 
        # modified accordingly
        print('Done training, testing phase')
        CMD = ["python3", TEST_FILE_NAME]
        for k,v in grid.items():
            CMD.extend( [f'--{k}', f'{v}'] )
        call(CMD)

        save_cfg(configuration, CFG_FILE_PATH[dataset_name])        
        test_metrics = load_metrics(TEST_METRICS_FILE_PATH[dataset_name])
        best_metrics = load_metrics(BEST_TEST_METRICS_FILE_PATH[dataset_name])
        save_best(best_metrics, test_metrics, grid)
    
        if args.wandb:
            wandb.log(metrics_average(test_metrics))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")
    parser.add_argument(
        "--wandb_entity",
        required="--wandb" in sys.argv,
        type=str,
        help="Entity name to push to the wandb logged data, in case args.wandb is specified.",
    )    

    main(args)
                 
    