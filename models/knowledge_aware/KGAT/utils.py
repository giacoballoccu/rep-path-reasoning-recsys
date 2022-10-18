from easydict import EasyDict as edict
import wandb
import os

ML1M = 'ml1m'
LFM1M = 'lfm1m'
CELL = 'cellphones'
MODEL = 'kgat'
ROOT_DIR = os.environ('TREX_DATA_ROOT') if 'TREX_DATA_ROOT' in os.environ else '../../..'
# Dataset directories.
DATA_DIR = {
    ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed/{MODEL}',
    LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed/{MODEL}',
    CELL: f'{ROOT_DIR}/data/{CELL}/preprocessed/{MODEL}'
}
# Model result directories.
TMP_DIR = {
    ML1M: f'{DATA_DIR[ML1M]}/tmp',
    LFM1M: f'{DATA_DIR[LFM1M]}/tmp',
    CELL: f'{DATA_DIR[CELL]}/tmp',
}



OPTIM_HPARAMS_METRIC = 'valid_ndcg'
VALID_METRICS_FILE_NAME = 'valid_metrics.json'

LOG_DIR = f'{ROOT_DIR}/results/{MODEL}'

CFG_DIR = {
    ML1M: f'{LOG_DIR}/{ML1M}/hparams_cfg',
    LFM1M: f'{LOG_DIR}/{LFM1M}/hparams_cfg',
    CELL: f'{LOG_DIR}/{CELL}/hparams_cfg',
}
BEST_CFG_DIR = {
    ML1M: f'{LOG_DIR}/{ML1M}/best_hparams_cfg',
    LFM1M: f'{LOG_DIR}/{LFM1M}/best_hparams_cfg',
    CELL: f'{LOG_DIR}/{CELL}/best_hparams_cfg',
}
TEST_METRICS_FILE_NAME = 'test_metrics.json'
TEST_METRICS_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{TEST_METRICS_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{TEST_METRICS_FILE_NAME}',
    CELL: f'{CFG_DIR[CELL]}/{TEST_METRICS_FILE_NAME}',
}
BEST_TEST_METRICS_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{TEST_METRICS_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{TEST_METRICS_FILE_NAME}',
    CELL: f'{BEST_CFG_DIR[CELL]}/{TEST_METRICS_FILE_NAME}',
}


CONFIG_FILE_NAME = 'config.json'
CFG_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{CONFIG_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{CONFIG_FILE_NAME}',
    CELL: f'{CFG_DIR[CELL]}/{CONFIG_FILE_NAME}',
}
BEST_CFG_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{CONFIG_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{CONFIG_FILE_NAME}',
    CELL: f'{BEST_CFG_DIR[CELL]}/{CONFIG_FILE_NAME}',
}



def makedirs(dataset_name):
    os.makedirs(BEST_CFG_DIR[dataset_name], exist_ok=True)
    os.makedirs(CFG_DIR[dataset_name], exist_ok=True)






logger = None
class MetricsLogger:
    # attribute names
    WANDB_ENTITY='wandb_entity'
    PROJECT_NAME='project_name'
    WANDB_CONFIG = 'config'
    def __init__(self, wandb_entity=None, project_name=None, config=None):
        self.wandb_entity = wandb_entity 
        # extra care should be taken to call the wandb method only from
        # main process if distributed training is on
        if self.wandb_entity is not None:
            assert wandb_entity is not None, f'Error {MetricsLogger.WANDB_ENTITY} is None, but is required for wandb logging.\n Please provide your account name as value of this member variable'
            assert project_name is not None, f'Error "{MetricsLogger.PROJECT_NAME}" is None, but is required for wandb logging'
            self.wandb_run = wandb.init(project=project_name,
                       entity=wandb_entity, config=config)   
        self.metrics = dict()
    
    def register(self, metric_name):
        self.metrics[metric_name] = []
        setattr(self, metric_name, self.metrics[metric_name])

    def log(self, metric_name, value):
        if metric_name not in self.metrics:
            self.register(metric_name)
        self.metrics[metric_name].append(value)


    def history(self, metric_name, n_samples ):
        # return latest n_samples of metric_name
        return self.metrics[metric_name][-n_samples:]
    def push(self, metric_names):
        if self.wandb_entity is not None:
            to_push = dict()
            for name in metric_names:
                to_push[name] = self.metrics[name][-1]
            wandb.log(to_push)
    def push_model(self, model_filepath, model_name):
        artifact = wandb.Artifact(model_name, type='model')
        artifact.add_file(model_filepath)
        self.wandb_run.log_artifact(artifact)
        self.wandb_run.join()

    def write(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            import json
            import copy
            json.dump(self.metrics, f) 