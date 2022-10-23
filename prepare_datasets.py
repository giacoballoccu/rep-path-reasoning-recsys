import argparse
import wandb
import sys
import os
import subprocess

from utils import *


def main():

    preprocess_files = {
            TRANSE: 'models/embeddings/transe/preprocess.py',
            PGPR : 'models/PGPR/preprocess.py',
            CAFE : 'models/CAFE/preprocess.py',
            UCPR : 'models/UCPR/preprocess/preprocess.py',
            KGAT : '',
            CKE : '',
            CFKG : '',
 
            BPRMF : '',
            NFM : '',
            FM : '',
    }
    for dataset_name in DATASETS:
        for model_name in preprocess_files:
            if model_name not in (PATH_REASONING_METHODS+EMBEDDING_METHODS):
                continue
            #    print(f'Model {args.model} already processed')
            #    return 

            CMD = ["python3", os.path.basename(preprocess_files[model_name]) , "--dataset" , dataset_name]
            subprocess.call(CMD, cwd=os.path.dirname(preprocess_files[model_name] )  )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()
                 
    
