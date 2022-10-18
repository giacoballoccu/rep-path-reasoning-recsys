from __future__ import absolute_import, division, print_function
import warnings

import numpy as np
import torch

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import argparse
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
#from models.PGPR.pgpr_utils import ML1M, TMP_DIR, get_logger, set_random_seed, USER, LOG_DIR, HPARAMS_FILE
from models.PGPR.pgpr_utils import *
from models.PGPR.kg_env import BatchKGEnvironment
from easydict import EasyDict as edict
from collections import defaultdict
import wandb
import sys
from models.utils import MetricsLogger
logger = None

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])



class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.actor = nn.Linear(hidden_sizes[1], act_dim)
        self.critic = nn.Linear(hidden_sizes[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def forward(self, inputs):
        state, act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.5)
        out = self.l2(x)
        x = F.dropout(F.elu(out), p=0.5)

        actor_logits = self.actor(x)
        #actor_logits[1 - act_mask] = -999999.0
        actor_logits[~act_mask] = -999999.0
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def select_action(self, batch_state, batch_act_mask, device):
        state = torch.FloatTensor(batch_state).to(device)  # Tensor [bs, state_dim]
        act_mask = torch.BoolTensor(batch_act_mask).to(device)  # Tensor of [bs, act_dim]

        probs, value = self((state, act_mask))  # act_probs: [bs, act_dim], state_value: [bs, 1]
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False
        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist()

    def update(self, optimizer, device, ent_weight):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0

        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[i]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()


class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_uids = self.uids[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()


def train(args):
    # check how datasets are loaded by BatchKGEnvironment
    train_env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len,
                             state_history=args.state_history)
    valid_env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len,
                             state_history=args.state_history)
    train_uids = list(train_env.kg(USER).keys())
    valid_uids = list(valid_env.kg(USER).keys())
    train_dataloader = ACDataLoader(train_uids, args.batch_size)
    valid_dataloader = ACDataLoader(valid_uids, args.batch_size)


    model = ActorCritic(train_env.state_dim, train_env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model.load_state_dict(model_sd)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metrics = MetricsLogger(args.wandb_entity, 
                            f'pgpr_{args.dataset}',
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
    metrics.register('avg_valid_reward')
    metrics.register('std_valid_reward')
    loaders = {'train': train_dataloader,
                'valid': valid_dataloader}
    envs = {'train': train_env,
            'valid':valid_env}
    step_counter = {
                'train': 0,
            'valid':0
    }
    uids_split = {'train' :train_uids,
                'valid':valid_uids}

    first_iterate = True
    model.train()
    start = 0
    for epoch in range(1, args.epochs + 1):
        splits_to_compute = list(loaders.items())
        if first_iterate:
            first_iterate = False
            splits_to_compute.insert(0, ('valid', valid_dataloader))        
        for split_name, dataloader in splits_to_compute:
            if split_name == 'valid' and epoch%10 != 0:
                continue            
            if split_name == 'valid':
                model.eval()
            else:
                model.train()
            dataloader.reset()
            env = envs[split_name]
            uids = uids_split[split_name]

            iter_counter = 0
            ### Start epoch ###
            dataloader.reset()
            while dataloader.has_next():
                batch_uids = dataloader.get_batch()
                ### Start batch episodes ###
                batch_state = env.reset(batch_uids)  # numpy array of [bs, state_dim]
                done = False
                while not done:
                    batch_act_mask = env.batch_action_mask(dropout=args.act_dropout) # numpy array of size [bs, act_dim]
                    batch_act_idx = model.select_action(batch_state, batch_act_mask, args.device)  # int
                    batch_state, batch_reward, done = env.batch_step(batch_act_idx)
                    model.rewards.append(batch_reward)

                ### End of episodes ###
                if split_name == 'train':
                    lr = args.lr * max(1e-4, 1.0 - float(step_counter[split_name]) / (args.epochs * len(uids) / args.batch_size))
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr

                # Update policy
                total_reward = np.sum(model.rewards)
                loss, ploss, vloss, eloss = model.update(optimizer, args.device, args.ent_weight)
                cur_metrics = {f'{split_name}_loss':loss,
                                 f'{split_name}_ploss':ploss, 
                                 f'{split_name}_vloss':vloss, 
                                f'{split_name}_entropy':eloss,
                                f'{split_name}_reward':total_reward,
                                f'{split_name}_iter': step_counter[split_name]}

                for k,v in cur_metrics.items():
                    metrics.log(k, v)
                #metrics.push(cur_metrics.keys())
                
                step_counter[split_name] += 1
                iter_counter += 1


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
        if epoch % 10 == 0:
            policy_file = '{}/policy_model_epoch_{}.ckpt'.format(args.log_dir, epoch)
            logger.info("Save models to " + policy_file)
            torch.save(model.state_dict(), policy_file)
            metrics.push_model(policy_file, f'{MODEL}_{args.dataset}_{epoch}')
    makedirs(args.dataset)
    metrics.write(TEST_METRICS_FILE_PATH[args.dataset])#os.path.join(TMP_DIR[args.dataset], VALID_METRICS_FILE_NAME))
    metrics.close_wandb()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {ML1M}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float, default=0, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--do_validation', type=bool, default=True, help='Whether to perform validation')
    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")
    parser.add_argument(
        "--wandb_entity",
        required="--wandb" in sys.argv,
        type=str,
        help="Entity name to push to the wandb logged data, in case args.wandb is specified.",
    )  

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

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
  

    args.log_dir = os.path.join(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()
