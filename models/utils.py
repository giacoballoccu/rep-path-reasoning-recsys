import wandb
import os
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

    def write(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            import json
            import copy
            json.dump(self.metrics, f)  
    def close_wandb(self):
        if self.wandb_entity is not None:
            self.wandb_run.finish()