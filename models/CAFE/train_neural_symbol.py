from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import logging
import logging.handlers
import torch
import torch.optim as optim
#from tensorboardX import SummaryWriter
import time

from models.CAFE.knowledge_graph import *
from models.CAFE.data_utils import OnlinePathLoader, OnlinePathLoaderWithMPSplit, KGMask
from models.CAFE.symbolic_model import EntityEmbeddingModel, SymbolicNetwork, create_symbolic_model
from models.CAFE.cafe_utils import *
from easydict import EasyDict as edict
import wandb
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
            wandb.init(project=project_name,
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
    def write(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            import json
            import copy
            json.dump(self.metrics, f)   


def set_logger(logname):
    global logger
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def train(args):
    train_dataloader = OnlinePathLoader(args.dataset, args.batch_size, topk=args.topk_candidates)
    valid_dataloader = OnlinePathLoader(args.dataset, args.batch_size, topk=args.topk_candidates)
    metapaths = train_dataloader.kg.metapaths

    "?????????????????????????????????????????????????????"
    kg_embeds = load_embed(args.dataset) if train else None

    model = create_symbolic_model(args, train_dataloader.kg, train=True, pretrain_embeds=kg_embeds)
    params = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f'Trainable parameters: {params}')
    logger.info('==================================')

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    total_steps = args.epochs * train_dataloader.total_steps
    

    metrics = MetricsLogger(args.wandb_entity, 
                            f'pgpr_{args.dataset}',
                            config=args)
    metrics.register('train_loss')
    metrics.register('train_regloss')
    metrics.register('train_rankloss')

    metrics.register('avg_train_loss')
    metrics.register('avg_train_regloss')
    metrics.register('avg_train_rankloss')

    metrics.register('valid_loss')
    metrics.register('valid_regloss')
    metrics.register('valid_rankloss')     

    metrics.register('avg_valid_loss')
    metrics.register('avg_valid_regloss')
    metrics.register('avg_valid_rankloss')     

    loaders = {'train': train_dataloader,
                'valid': valid_dataloader}

    step_counter = {
                'train': 0,
            'valid':0
    }
    first_iterate = True

    torch.save(model.state_dict(), '{}/symbolic_model_epoch{}.ckpt'.format(args.log_dir, 0))
    start_time = time.time()
    first_iterate = True
    model.train()
    for epoch in range(1, args.epochs + 1):

        splits_to_compute = list(loaders.items())
        if first_iterate:
            first_iterate = False
            splits_to_compute.insert(0, ('valid', valid_dataloader))
        for split_name, dataloader in splits_to_compute:
            if split_name == 'valid':
                model.eval()
            else:
                model.train()
            iter_counter = 0
            ### Start epoch ###
            dataloader.reset()
            while dataloader.has_next():
                # Update learning rate
                if split_name == 'train':
                    lr = args.lr * max(1e-4, 1.0 - steps / total_steps)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr

                # pos_paths: [bs, path_len], neg_paths: [bs, n, path_len]
                mpid, pos_paths, neg_pids = dataloader.get_batch()
                pos_paths = torch.from_numpy(pos_paths).to(args.device)
                neg_pids = torch.from_numpy(neg_pids).to(args.device)

                optimizer.zero_grad()
                reg_loss, rank_loss = model(metapaths[mpid], pos_paths, neg_pids)
                loss = reg_loss + args.rank_weight * rank_loss
                if split_name == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()


                cur_metrics = {f'{split_name}_loss': loss,
                                 f'{split_name}_regloss':reg_loss.item(), 
                                 f'{split_name}_rankloss':rank_loss.item(), 
                                f'{split_name}_iter': step_counter[split_name]}

                for k,v in cur_metrics.items():
                    metrics.log(k, v)
                metrics.push(cur_metrics.keys())
                
                step_counter[split_name] += 1
                iter_counter += 1

            cur_metrics = [f'{split_name}_epoch']
            cur_metrics.extend([f'{split_name}_loss',
                 f'{split_name}_regloss', 
                 f'{split_name}_rankloss'
                ])
            for k in cur_metrics[1:]:
                metrics.log(f'avg_{k}', sum(metrics.history(k, iter_counter))/max(iter_counter,1) )
                
            metrics.log(f'{split_name}_epoch', epoch)
            metrics.log(f'std_{split_name}_reward',np.std(metrics.history( f'{split_name}_reward', iter_counter)) )
            info = ""
            for k in cur_metrics:
                if isinstance(getattr(metrics,k)[-1],float):
                    x = '{:.5f}'.format(getattr(metrics, k)[-1])
                else:
                    x = '{:d}'.format(getattr(metrics, k)[-1])
                info = info + f'| {k}={x} ' 

            metrics.push(cur_metrics)
            logger.info(info)
            if epoch % 10 == 0:
                torch.save(model.state_dict(), '{}/symbolic_model_epoch{}.ckpt'.format(args.log_dir, epoch))

    metrics.write(TMP_DIR[args.dataset])

def main():
    args = parse_args()
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(args.log_dir + '/train_log.txt')
    logger.info(args)
    os.makedirs(TMP_DIR[args.dataset], exist_ok=True)
    with open(os.path.join(TMP_DIR[args.dataset],HPARAMS_FILE), 'w') as f:
        import json
        import copy
        args_dict = dict()
        for x,y in copy.deepcopy(args._get_kwargs()):
            args_dict[x] = y
        if 'device' in args_dict:
            del args_dict['device']
        json.dump(args_dict,f)
       

    train(args)


if __name__ == '__main__':
    main()
