from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from models.UCPR.utils import *
from models.UCPR.src.model.get_model.get_model import *
from models.UCPR.src.parser import parse_args
from models.UCPR.src.para_setting import parameter_path, parameter_path_th
from models.UCPR.src.data_loader import ACDataLoader
from models.UCPR.preprocess.dataset import Dataset
from models.UCPR.preprocess.knowledge_graph import KnowledgeGraph
import time
import json
from easydict import EasyDict as edict
import wandb
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def pretrain_set(args, env):
    if args.load_pretrain_model == True:    
        logger = get_logger(args.log_dir + '/train_log_pretrain.txt')
        args.logger = logger

        print(args.pretrained_dir + '/' + args.sort_by + '_pretrained_md_json_' + args.topk_string + '.txt')
        with open(args.pretrained_dir + '/' + args.sort_by + '_pretrained_md_json_' + args.topk_string + '.txt') as json_file:
            best_model_json = json.load(json_file)

        logger.info(args.pretrained_dir + '/' + args.sort_by + '_pretrained_md_json_' + args.topk_string + '.txt')

        policy_file = best_model_json['pretrained_file']
        pretrain_sd = torch.load(policy_file)

        logger.info("pretrain_model_load")
        logger.info(policy_file)
        # input()

        model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
        model_sd = model.state_dict()

        pretrain_sd = {k: v for k, v in pretrain_sd.items() if k in model_sd}
        para_meter = [k.split('.')[0] for k, v in pretrain_sd.items()]

        model_sd.update(pretrain_sd)
        model.load_state_dict(model_sd)


        for name, child in model.named_children():
            print('name = ', name)
            for param in child.parameters():
                param.requires_grad = True

        if args.dataset in [BEAUTY_CORE, CELL_CORE, CLOTH_CORE]:         
            for name, child in model.named_children():
                print('name = ', name)
                if 'kg' in name:
                    print('name = ', name)
                    for param in child.parameters():
                        param.requires_grad = False

        elif args.dataset in [LFM1M,ML1M]:#MOVIE_CORE, AZ_BOOK_CORE]:
            for name, child in model.named_children():
                print('name = ', name)
                if name in para_meter and 'kg' not in name and 'actor' not in name and 'critic' not in name:
                    print('name = ', name)
                    for param in child.parameters():
                        param.requires_grad = False
                        
        grad_string = ''
        for name, child in model.named_children():
            print('name = ', name)
            for param in child.parameters():
                print(param.requires_grad)
            grad_string += ' name = ' + name  + ' ' + str(param.requires_grad)

        logger.info(grad_string)
        # start_epoch = args.pretrained_st_epoch + 1

    else:
        logger = get_logger(args.log_dir + '/train_log.txt')
        args.logger = logger

        model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)

        grad_string = ''
        for name, child in model.named_children():
            print('name = ', name)
            for param in child.parameters():
                print(param.requires_grad)
            grad_string += ' name = ' + name  + ' ' + str(param.requires_grad)

        logger.info(grad_string)

    core_user_list = args.core_user_list
    #kg_fre_dict = args.kg_fre_dict
    sp_user_filter = args.sp_user_filter

    try:
        kg_user_filter = args.kg_user_filter
        args.kg_user_filter = ''
    except:
        pass

    args.core_user_list = ''
    #args.kg_fre_dict = ''
    args.sp_user_filter = ''
    logger.info(args)
    args.core_user_list = core_user_list
    #args.kg_fre_dict = kg_fre_dict
    args.sp_user_filter = sp_user_filter

    try:
        args.kg_user_filter = kg_user_filter
    except:
        pass

    del core_user_list#, kg_fre_dict

    return model, logger



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

def train(args):

    train_env = KG_Env(args, Dataset(args, set_name='train'), args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    valid_env = KG_Env(args, Dataset(args, set_name='valid'), args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    print('env.output_valid_user() = ', len(train_env.output_valid_user()))
    print('args.batch_size = ', args.batch_size)

    train_dataloader = ACDataLoader(train_env.output_valid_user(), args.batch_size)
    valid_dataloader = ACDataLoader(valid_env.output_valid_user(), args.batch_size)

    model, logger = pretrain_set(args, train_env)
    logger.info('valid user = ')
    # logger.info(env.output_valid_user())
    logger.info(len(train_env.output_valid_user()))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    step=0

 

    metrics = MetricsLogger(args.wandb_entity, 
                            f'ucpr_{args.dataset}',
                            config=args)
    metrics.register('train_loss')
    metrics.register('train_ploss')
    metrics.register('train_vloss')
    metrics.register('train_entropy')
    metrics.register('train_reward')

    metrics.register('avg_train_loss')
    metrics.register('avg_train_ploss')
    metrics.register('avg_train_vloss')
    metrics.register('avg_train_entropy')
    metrics.register('avg_train_reward')
    metrics.register('std_train_reward')

    metrics.register('valid_loss')
    metrics.register('valid_ploss')
    metrics.register('valid_vloss')     
    metrics.register('valid_entropy')
    metrics.register('valid_reward')

    metrics.register('avg_valid_loss')
    metrics.register('avg_valid_ploss')
    metrics.register('avg_valid_vloss')     
    metrics.register('avg_valid_entropy')
    metrics.register('avg_valid_entropy')
    metrics.register('avg_valid_reward')
                 

    loaders = {'train': train_dataloader,
                'valid': valid_dataloader}
    envs = {'train': train_env,
            'valid':valid_env}
    step_counter = {
                'train': 0,
            'valid':0
    }
    first_iterate = True
    for epoch in range(0, args.epochs + 1):
        splits_to_compute = list(loaders.items())
        if first_iterate:
            first_iterate = False
            splits_to_compute.insert(0, ('valid', valid_dataloader))   
        for split_name, dataloader in splits_to_compute:
            if split_name == 'valid':
                model.eval()
            else:
                model.train()
            dataloader.reset()
            env = envs[split_name]
            

            iter_counter = 0
            dataloader.reset()
            while dataloader.has_next():
                batch_uids = dataloader.get_batch()
                ### Start batch episodes ###
                env.reset(epoch, batch_uids, training = True)  # numpy array of [bs, state_dim]
                model.user_triplet_set = env.user_triplet_set
                model.reset(batch_uids)

                while not env._done:
                    batch_act_mask = env.batch_action_mask(dropout=args.act_dropout)  # numpy array of size [bs, act_dim]
                    batch_emb_state = model.generate_st_emb(env._batch_path)
                    batch_next_action_emb = model.generate_act_emb(env._batch_path, env._batch_curr_actions)
                    batch_act_idx = model.select_action(batch_emb_state, batch_next_action_emb, batch_act_mask, args.device)  # int
                    batch_state, batch_reward = env.batch_step(batch_act_idx)
                    model.rewards.append(batch_reward)
                ### End of episodes ###

                for pg in optimizer.param_groups:
                    lr = pg['lr']

                total_reward = np.sum(model.rewards)
                # Update policy
                loss, ploss, vloss, eloss = model.update(optimizer, env, args.device, args.ent_weight, step_counter[split_name])
                cur_metrics = {f'{split_name}_loss':loss,
                                 f'{split_name}_ploss':ploss, 
                                 f'{split_name}_vloss':vloss, 
                                f'{split_name}_entropy':eloss,
                                f'{split_name}_reward':total_reward,
                                f'{split_name}_iter': step_counter[split_name]}

                for k,v in cur_metrics.items():
                    metrics.log(k, v)
                metrics.push(cur_metrics.keys())
                
                step_counter[split_name] += 1
                iter_counter += 1
                
                #if step_counter[split_name] > 0 and step_counter[split_name] % 100 == 0:
                #    #avg_reward = np.mean(total_rewards) / args.batch_size
                #    dataloader.reset()
            cur_metrics = [f'{split_name}_epoch']
            cur_metrics.extend([f'{split_name}_loss',
                 f'{split_name}_ploss', 
                 f'{split_name}_vloss', 
                f'{split_name}_entropy',
                f'{split_name}_reward',
                ])
            for k in cur_metrics[1:]:
                metrics.log(f'avg_{k}', sum(metrics.history(k, iter_counter))/max(iter_counter,1) )
            getattr(metrics, f'avg_{split_name}_reward')[-1] /= args.batch_size 


            
            metrics.log(f'{split_name}_epoch', epoch)
            cur_metrics.append(f'std_{split_name}_reward')
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

        ### END of epoch ###
        if epoch % 5 == 0:
            policy_file = '{}/policy_model_epoch_{}.ckpt'.format(args.save_model_dir, epoch)
            logger.info("Save model to " + policy_file)
            torch.save(model.state_dict(), policy_file)

        cur_tim = time.strftime("%Y%m%d-%H%M%S")
        logger.info("current time = " + str(cur_tim))
        metrics.write(TMP_DIR[args.dataset])


if __name__ == '__main__':
    args = parse_args()
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


    args.training = 1
    args.training = (args.training == 1)
        
    if args.envir == 'p1': 
        para_env = parameter_path_th
        KG_Env = KGEnvironment

    elif args.envir == 'p2':
        para_env = parameter_path
        KG_Env = KGEnvironment

    para_env(args)
    train(args)
