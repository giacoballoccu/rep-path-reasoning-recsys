import argparse
import wandb
import sys
import os
import subprocess

from utils import *


def main(args):

    train_files = {
            TRANSE : 'models/embeddings/transe/transe_gridsearch.py',
            PGPR : 'models/PGPR/gridsearch.py',
            CAFE : 'models/CAFE/gridsearch.py',
            UCPR : 'models/UCPR/gridsearch.py',
            KGAT : 'models/knowledge_aware/KGAT/gridsearch.py',
            CKE : 'models/knowledge_aware/CKE/gridsearch.py',
            CFKG : 'models/knowledge_aware/CFKG/gridsearch.py',
 
            #BPRMF : 'models/matrix_factorization/BPRMF/gridsearch.py'
            #NFM : 'models/matrix_factorization/NFM/gridsearch.py'
            #FM : 'models/matrix_factorization/FM/gridsearch.py'
    }
    assert args.model in train_files, f'Error, given model name {args.model} not found in available models'
    ensure_dataset_name(args.dataset)
    TRAIN_FILE_NAME = train_files[args.model]
    CMD = ["python3", os.path.basename(TRAIN_FILE_NAME) , "--dataset" , args.dataset]

    if args.wandb:
        CMD.append('--wandb')
        CMD.extend( ['--wandb_entity', args.wandb_entity ] )

    subprocess.call(CMD, cwd=os.path.dirname(TRAIN_FILE_NAME)  )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=LFM1M, help='One of {ml1m, lfm1m}')
    parser.add_argument('--model', type=str, default=PGPR, help='Model to train: {pgpr, cafe, ucpr, cke, cfkg, kgat}')
    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")
    parser.add_argument(
        "--wandb_entity",
        required="--wandb" in sys.argv,
        type=str,
        help="Entity name to push to the wandb logged data, in case args.wandb is specified.",
    )    
    args = parser.parse_args()
    main(args)
                 