import argparse
import wandb
import sys
import os
import subprocess

from utils import *


def main(args):

    test_files = {
            PGPR : 'models/PGPR/test_agent.py',
            CAFE : 'models/CAFE/execute_neural_symbol.py',
            UCPR : 'models/UCPR/test.py',
            KGAT : 'models/knowledge_aware/KGAT/test.py',
            CKE : 'models/knowledge_aware/CKE/test.py',
            CFKG : 'models/knowledge_aware/CFKG/test.py',
 
            BPRMF : 'models/matrix_factorization/BPRMF/test.py',
            NFM : 'models/matrix_factorization/NFM/test.py',
            FM : 'models/matrix_factorization/FM/test.py',
    }
    assert args.model in test_files, 'Error, given model name {args.model} not found in available models'
    assert ensure_dataset_name(args.dataset), f'Error dataset {args.dataset} not found in {DATASETS}'
    TEST_FILE_NAME = test_files[args.model]
    CMD = ["python3", os.path.basename(TEST_FILE_NAME) , "--dataset" , args.dataset]

    if args.wandb:
        CMD.append('--wandb')
        CMD.extend( ['--wandb_entity', args.wandb_entity ] )

    subprocess.call(CMD, cwd=os.path.dirname(TEST_FILE_NAME)  )



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
                 
    