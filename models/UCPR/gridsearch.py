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
from tqdm import tqdm
TRAIN_FILE_NAME = 'src/train.py'
TEST_FILE_NAME = 'src/test.py'

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
        shutil.rmtree(BEST_CFG_DIR[dataset_name])
        shutil.copytree(TMP_DIR[dataset_name], BEST_CFG_DIR[dataset_name] )
        save_metrics(test_metrics, f'{BEST_TEST_METRICS_FILE_PATH[dataset_name]}')
        save_cfg(grid, f'{BEST_CFG_FILE_PATH[dataset_name] }')
        return 

    x = sum(test_metrics[OPTIM_HPARAMS_METRIC][-OPTIM_HPARAMS_LAST_K:])/OPTIM_HPARAMS_LAST_K
    best_x = sum(best_metrics[OPTIM_HPARAMS_METRIC][-OPTIM_HPARAMS_LAST_K:])/OPTIM_HPARAMS_LAST_K
    # if avg total reward is higher than current best
    if x > best_x :
        shutil.rmtree(BEST_CFG_DIR[dataset_name])
        shutil.copytree(TMP_DIR[dataset_name], BEST_CFG_DIR[dataset_name] )
        save_metrics(test_metrics, f'{BEST_TEST_METRICS_FILE_PATH[dataset_name]}')
        save_cfg(grid, f'{BEST_CFG_FILE_PATH[dataset_name] }')





def main(args):


    chosen_hyperparam_grid = { 
    "act_dropout": [0.5], 
    "batch_size": [128],   
    "dataset": ["lfm1m", 'ml1m'], 
    "embed_size": [50, 100, 200], 
    "ent_weight": [0.001],   
    "epochs": [40],  
    "gamma": [0.99], 
    "hidden": [[64, 32], [128, 64]], 
     "l2_lambda": [0], 
     "l2_weight": [1e-06], 
     "lambda_num": [0.5], 
     "lr": [7e-05], 
     "max_acts": [50], 
     "max_path_len": [3], 
     "model": ["UCPR"],  
     "n_memory": [32], 
"p_hop": [1],  
 "sub_batch_size": [1], 


    "add_products": [True], 
    "att_core": [0], 
    #"gp_setting": "6000_800_15_500_250", 
    "gpu": ["0"], 
    #"grad_check": [False],
     "gradient_plot": ["gradient_plot/"], 
     "h0_embbed": [0], 
    
     "item_core": [10], 
     #"kg_emb_grad": [False], 
     "kg_fre_lower": [15],
    "kg_fre_upper": [500],  

     "name": ["train_agent_enti_emb"],  
       
     "run_eval": [True], 
     "run_path": [True], 
     "sam_type": ["alet"],   
     "seed": [52], 
     "sort_by": ["prob"], 
     "state_history": [1], 
     #"state_rg": [False], 
    
     #"test_lstm_up": [True], 
     "topk": [[25, 5, 1]], 
     "topk_list": [[1, 10, 100, 100]],  
     #"tri_pro_rm": [False], 
     #"tri_wd_rm": [False], 
     "user_core": [300], 
     "user_core_th": [6], 
     #"user_o": [False], 
     "wandb": [True if args.wandb else False], 
     "wandb_entity": [args.wandb_entity]}
    hparam_grids = ParameterGrid(chosen_hyperparam_grid)
    print('num_experiments: ', len(hparam_grids))

    for i, configuration in enumerate(tqdm(hparam_grids)):
        dataset_name = configuration["dataset"]
        makedirs(dataset_name)
        if args.wandb:
            wandb.init(project=f'grid_ucpr_{dataset_name}',
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
        print(f'Executing job {i+1}/{len(hparam_grids)}: ',configuration)
        subprocess.call(CMD)
        #,
        #        stdout=subprocess.DEVNULL,
        #        stderr=subprocess.STDOUT)
        
        '''     
        # cafe and ucpr have the same command line args, pgpr does not, so the call below will have to be 
        # modified accordingly
        print('Done training, testing phase')
        CMD = ["python3", TEST_FILE_NAME]
        for k,v in configuration.items():
                if k == 'wandb':
                    CMD.extend([f'--{k}'])
                elif isinstance(v,list):
                    cmd_args = [f'--{k}'] + [f" {val} " for val in v]
                    CMD.extend( cmd_args )
                else:
                    CMD.extend( [f'--{k}', f'{v}'] )   
        subprocess.call(CMD)
        '''

        save_cfg(configuration, CFG_FILE_PATH[dataset_name])        
        test_metrics = load_metrics(TEST_METRICS_FILE_PATH[dataset_name])
        best_metrics = load_metrics(BEST_TEST_METRICS_FILE_PATH[dataset_name])
        save_best(best_metrics, test_metrics, configuration)
    
        #if args.wandb:
        #    wandb.log(test_metrics)





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
                 
    