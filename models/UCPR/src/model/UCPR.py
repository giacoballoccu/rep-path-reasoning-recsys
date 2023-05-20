from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from easydict import EasyDict as edict
from models.UCPR.src.model.lstm_base.model_lstm_mf_emb import AC_lstm_mf_dummy
from models.UCPR.src.model.lstm_base.model_kg import KG_KGE#, RW_KGE
from models.UCPR.src.model.lstm_base.model_kg_pre import KG_KGE_pretrained#, RW_KGE_pretrained
from models.UCPR.src.model.lstm_base.backbone_lstm import EncoderRNN, EncoderRNN_batch, KGState_LSTM, KGState_LSTM_no_rela
from models.UCPR.utils import *


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class UCPR(nn.Module):
    def __init__(self, args, user_triplet_set, rela_2_index, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super(UCPR,self).__init__()#super().__init__(args, user_triplet_set, rela_2_index, act_dim, gamma, hidden_sizes)
        self.args = args        
        self.act_dim = act_dim
        self.device = args.device
        self.gamma = gamma
        self.p_hop = args.p_hop
        self.hidden_sizes = hidden_sizes
        self.n_memory = args.n_memory
        self.rela_2_index = rela_2_index

        self.embed_size = args.embed_size
 
        self.user_triplet_set = user_triplet_set


        self.kg = load_kg(args.dataset)
        self.dataset_name = self.kg.dataset_name




        self.l2_weight = args.l2_weight
        self.sub_batch_size = args.sub_batch_size
        self.device = args.device

        self.scalar = nn.Parameter(torch.Tensor([args.lambda_num]), requires_grad=True)
        print('args.lambda_num = ', args.lambda_num)
        #print(torch.ones(max(self.user_triplet_set) * 2 + 1, 1, self.embed_size).shape)
        #print('qqqq')
        #print(self.dummy_rela)
        self.dummy_rela = torch.ones(max(self.user_triplet_set) * 2 + 1, 1, self.embed_size)
        #print('aaaa',self.dummy_rela)
        self.dummy_rela = nn.Parameter(self.dummy_rela, requires_grad=True).to(self.device)
        #print('sssss')
        self.dummy_rela_emb = nn.Embedding(max(self.user_triplet_set) * 2 + 1, self.embed_size * self.embed_size).to(self.device)


        self.dataset_name = args.dataset
        self.dataset = load_dataset(args.dataset)
        
        if self.args.envir == 'p1':
            self._get_next_node_type = lambda curr_node_type, relation, next_node_id:  \
                        KG_RELATION[self.dataset_name][curr_node_type][relation]            

            if self.args.KGE_pretrained == True: self.kg_emb = RW_KGE_pretrained(args)
            else:  self.kg_emb = RW_KGE(args)

        elif self.args.envir == 'p2':
            self._get_next_node_type = lambda curr_node_type, relation, next_node_id:  \
                        KG_RELATION[self.dataset_name][curr_node_type][relation]
            if self.args.KGE_pretrained == True: self.kg_emb = KG_KGE_pretrained(args)
            else:  self.kg_emb = KG_KGE(args)

            


        self.bulid_mode_user()
        self.bulid_model_rl()
        self.bulid_model_reasoning()


    def bulid_model_rl(self):
        self.state_lstm = KGState_LSTM(self.args, history_len=1)

        self.l1 = nn.Linear(2 * self.embed_size, self.hidden_sizes[1])
        self.l2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])

        self.actor = nn.Linear(self.hidden_sizes[1], self.act_dim)
        self.critic = nn.Linear(self.hidden_sizes[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def bulid_model_reasoning(self):

        self.reasoning_step = self.args.reasoning_step
        self.rn_state_tr_query = []
        self.update_rn_state = []
        self.rn_query_st_tr = []
        self.rh_query = []
        self.o_r_query = []
        self.v_query = []
        self.t_u_query = []
        for i in range(self.reasoning_step):
            self.rn_state_tr_query.append(nn.Linear(self.embed_size * 2, self.embed_size).to(self.device))
            self.update_rn_state.append(nn.Linear(self.embed_size * 2, self.embed_size).to(self.device))
            self.rn_query_st_tr.append(nn.Linear(self.embed_size * (self.p_hop), self.embed_size).to(self.device))

            self.rh_query.append(nn.Linear(self.embed_size, self.embed_size, bias=False).to(self.device))
            self.o_r_query.append(nn.Linear(self.embed_size, self.embed_size, bias=False).to(self.device))
            self.v_query.append(nn.Linear(self.embed_size, self.embed_size, bias=False).to(self.device))
            self.t_u_query.append(nn.Linear(self.embed_size, self.embed_size, bias=False).to(self.device))

        self.rn_state_tr_query = nn.ModuleList(self.rn_state_tr_query)
        self.update_rn_state = nn.ModuleList(self.update_rn_state)
        self.rn_query_st_tr = nn.ModuleList(self.rn_query_st_tr)

        self.rh_query = nn.ModuleList(self.rh_query)
        self.o_r_query = nn.ModuleList(self.o_r_query)
        self.v_query = nn.ModuleList(self.v_query)
        self.t_u_query = nn.ModuleList(self.t_u_query)

        self.rn_cal_state_prop = nn.Linear(self.embed_size, 1, bias=False).to(self.device)#cuda()

    def bulid_mode_user(self):

        self.relation_emb = nn.Embedding(len(self.rela_2_index), self.embed_size * self.embed_size)
        
        self.update_us_tr = []
        for hop in range(self.p_hop):
            self.update_us_tr.append(nn.Linear(self.embed_size * 2, self.embed_size).to(self.device))
        self.update_us_tr = nn.ModuleList(self.update_us_tr)

        self.cal_state_prop = nn.Linear(self.embed_size * 3, 1)


    def reset(self, uids=None):
        
        self.lstm_state_cache = []

        self.uids = [uid for uid in uids for _ in range(self.sub_batch_size)]
        self.memories_h = {}
        self.memories_r = {}
        self.memories_t = {}

        for i in range(max(1,self.p_hop)):

            self.memories_h[i] = th.cat([th.cat([self.kg_emb.lookup_emb(u_set[0], type_index = torch.LongTensor([u_set[1]]).to(self.device))
                                 for u_set in self.user_triplet_set[user][i][0]], 0).unsqueeze(0) for user in self.uids], 0)
            
            self.memories_r[i] = th.cat([self.relation_emb(torch.LongTensor([self.rela_2_index[relation] 
                                    for relation in self.user_triplet_set[user][i][1]]).to(self.device)).unsqueeze(0) for user in self.uids], 0)
            
            self.memories_r[i] = self.memories_r[i].view(-1, self.n_memory, self.embed_size, self.embed_size)

            self.memories_t[i] = th.cat([th.cat([self.kg_emb.lookup_emb(u_set[0], type_index = torch.LongTensor([u_set[1]]).to(self.device))
                                 for u_set in self.user_triplet_set[user][i][2]], 0).unsqueeze(0) for user in self.uids], 0)

        
        self.prev_state_h, self.prev_state_c = self.state_lstm.set_up_hidden_state(len(self.uids))
        #print('OOOOOO', self.prev_state_h, isinstance(self.prev_state_h,nn.Parameter))


    def forward(self, inputs):
        state, res_user_emb, next_enti_emb, next_action_emb,  act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]

        state_tr = state.unsqueeze(1).repeat(1, next_action_emb.shape[1], 1)
        probs_st = state_tr * next_action_emb 

        res_user_emb = res_user_emb.unsqueeze(1).repeat(1, next_enti_emb.shape[1], 1)
        probs_user = res_user_emb * next_enti_emb

        probs_st = probs_st.sum(-1)

        probs_user = probs_user.sum(-1)

        scalar = self.scalar.unsqueeze(1).repeat(probs_user.shape[0],1)

        probs = probs_st + self.scalar * probs_user
        probs = probs.masked_fill(~act_mask, value=torch.tensor(-1e10))
        act_probs = F.softmax(probs, dim=-1)

        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.4)

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def select_action(self, batch_state, batch_next_action_emb, batch_act_mask, device):
        
        act_mask = torch.BoolTensor(batch_act_mask).to(device)  # Tensor of [bs, act_dim]
        
        state_output, res_user_emb = batch_state[0], batch_state[1]
        next_enti_emb, next_action_emb = batch_next_action_emb[0], batch_next_action_emb[1]
        probs, value = self((state_output, res_user_emb, next_enti_emb, next_action_emb, act_mask))  # act_probs: [bs, act_dim], state_value: [bs, 1]

        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())

        return acts.cpu().numpy().tolist()

    def rn_query_st(self, state, relation_embed_dual, rn_step):
        #print('RN QUERY ST')
        user_embeddings = self.memories_h[0][:,0]
        #print(user_embeddings.shape)
        # state = th.cat([state.squeeze(), user_embeddings], -1)
        state = th.cat([state.squeeze(1)], -1)
        relation_embed_dual = th.cat([relation_embed_dual.squeeze(1)], -1)
        #print(state.shape)
        #print(relation_embed_dual.shape)
        o_list = []
        for hop in range(self.p_hop):
            
            h_expanded = torch.unsqueeze(self.memories_t[hop], dim=3)
            #print(f'\t hop={hop} ', h_expanded.shape)
            #print(f'\t hop={hop} ', self.memories_r[hop].shape)
            Rh = torch.squeeze(torch.matmul(self.memories_r[hop], h_expanded))
            # [batch_size, dim, 1]
            #print(f'\t hop={hop} Rh ', Rh.shape)
            #print(f'\t hop={hop} take tail shape', self.memories_t[0].shape)
            v =  state.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1)
            #print(f'\t hop={hop} Rh ', v.shape)
            r_v = relation_embed_dual.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1, 1)
            #print(f'\t hop={hop} r_v ', r_v.shape)
            r_vh = torch.squeeze(torch.matmul(r_v, h_expanded))
            #print(f'\t hop={hop}  r_vh', r_vh.shape)
            t_u = user_embeddings.unsqueeze(1).repeat(1, self.memories_t[0].shape[1], 1)
            #print(f'\t hop={hop} Rh ', t_u.shape)
            q_Rh = self.rh_query[rn_step](Rh)
            q_v = self.v_query[rn_step](v)
            t_u = self.t_u_query[rn_step](t_u)
            o_r = self.o_r_query[rn_step](r_vh)

            t_state = torch.tanh(q_Rh + q_v + t_u + o_r)
            # print('t_state = ', t_state.shape)
            #print('rn_cal_prop: ', self.rn_cal_state_prop(t_state).shape)
            probs = torch.squeeze(self.rn_cal_state_prop(t_state),-1)
            #print('probs: ', probs.shape)
            probs_normalized = F.softmax(probs, dim=1)

            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (self.memories_t[hop] * probs_expanded).sum(dim=1).unsqueeze(1)

            o_list.append(o)

        o_list = torch.cat(o_list, 1)

        user_o = o_list.sum(1)

        return user_o

    def update_query_embedding(self, selc_entitiy):
        # update before query
        selc_entitiy = selc_entitiy.repeat(1, self.memories_t[0].shape[1], 1)
        
        for hop in range(self.p_hop):
            tmp_memories_t = th.cat([self.memories_t[hop], selc_entitiy], -1)
            self.memories_t[hop] = self.update_us_tr[hop](tmp_memories_t)

    def update_path_info_memories(self, up_date_hop):

        new_memories_h = {}
        new_memories_r = {}
        new_memories_t = {}

        for i in range(max(1,self.p_hop)):
            new_memories_h[i] = []
            new_memories_r[i] = []
            new_memories_t[i] = []

            for row in up_date_hop:
                new_memories_h[i].append(self.memories_h[i][row,:,:].unsqueeze(0))
                new_memories_r[i].append(self.memories_r[i][row,:,:,:].unsqueeze(0))
                new_memories_t[i].append(self.memories_t[i][row,:,:].unsqueeze(0))

            self.memories_h[i] = th.cat(new_memories_h[i], 0).to(self.device)
            self.memories_r[i] = th.cat(new_memories_r[i], 0).to(self.device)
            self.memories_t[i] = th.cat(new_memories_t[i], 0).to(self.device)
    def update_path_info(self, up_date_hop):

        new_uids = []

        for row in up_date_hop:
            new_uids.append(self.uids[row])

        self.uids = new_uids

        new_prev_state_h = []
        new_prev_state_c = []

        for row in up_date_hop:
            new_prev_state_h.append(self.prev_state_h[:,row,:].unsqueeze(1))
            new_prev_state_c.append(self.prev_state_c[:,row,:].unsqueeze(1))

        self.prev_state_h = nn.Parameter(th.cat(new_prev_state_h, 1)).to(self.device)
        self.prev_state_c = nn.Parameter(th.cat(new_prev_state_c, 1)).to(self.device)
    def generate_st_emb(self, batch_path, up_date_hop = None):
        if up_date_hop != None:
            self.update_path_info(up_date_hop)
            self.update_path_info_memories(up_date_hop)

        tmp_state = [self._get_state_update(index, path) for index, path in enumerate(batch_path)]

        all_state = th.cat([ts[3].unsqueeze(0) for ts in tmp_state], 0)
        #print(all_state.shape)
        if len(batch_path[0]) != 1:
            selc_entitiy = th.cat([ts[0].unsqueeze(0) for ts in tmp_state], 0)
            #print(selc_entitiy.shape)
            self.update_query_embedding(selc_entitiy)
        #print('Prev state: ', all_state.shape, self.prev_state_h.shape,  self.prev_state_c.shape)
        #print('AAAAAAA', self.prev_state_h)
        #_,x1,_ = self.state_lstm(all_state, 
        #            self.prev_state_h,  self.prev_state_c)
        #print(x1, isinstance(x1,nn.Parameter))
        state_output, self.prev_state_h, self.prev_state_c = self.state_lstm(all_state, 
                    self.prev_state_h,  self.prev_state_c)
        #print('New state: ', state_output.shape, self.prev_state_h.shape,  self.prev_state_c.shape)
        curr_node_embed = th.cat([ts[0].unsqueeze(0) for ts in tmp_state], 0)
        relation_embed = th.cat([ts[1].unsqueeze(0) for ts in tmp_state], 0)
        relation_embed_dual = th.cat([ts[2].unsqueeze(0) for ts in tmp_state], 0)
        #print('Rel: ', curr_node_embed.shape, relation_embed.shape, relation_embed_dual.shape)

        #print('Reasoning step')
        state_tmp = relation_embed
        #print(state_tmp.shape)
        for rn_step in range(self.reasoning_step):
            query_state = self.rn_query_st(state_tmp, relation_embed_dual, rn_step)
            #print(rn_step, ' ', query_state.shape)
            if rn_step < self.reasoning_step - 1: 
                state_tmp_ = th.cat([query_state, state_tmp], -1)
                #print('s1.1: ', state_tmp_.shape)
                state_tmp = self.update_rn_state[rn_step](state_tmp_)
                #print('s1.2: ', state_tmp.shape)
        # input()
        res_user_emb = query_state
        #print('Res user emb: ', query_state.shape)
        state_output = state_output.squeeze(1)#state_output.squeeze() #results in crash if batch size is 1,
        # furthermore the squeeze is useless, as for normal batches the tensors before squeeze do not have collapsable dimensions
        res_user_emb = res_user_emb.squeeze(1)# res_user_emb.squeeze() #results in crash if batch size is 1
        # furthermore the squeeze is useless, as for normal batches the tensors before squeeze do not have collapsable dimensions
        #print('Squeeze: ', state_output.shape, res_user_emb.shape)
        return [state_output, res_user_emb]

    def generate_act_emb(self, batch_path, batch_curr_actions):
        all_action_set = [self._get_actions(index, actions_sets[0], actions_sets[1]) 
                    for index, actions_sets in enumerate(zip(batch_path, batch_curr_actions))]
        enti_emb = th.cat([action_set[0].unsqueeze(0) for action_set in all_action_set], 0)
        next_action_state = th.cat([action_set[1].unsqueeze(0) for action_set in all_action_set], 0)

        return [enti_emb, next_action_state]
    def _get_actions(self, index, curr_path, curr_actions):

        last_relation, curr_node_type, curr_node_id = curr_path[-1]
        entities_embs = []
        relation_embs = []

        for action_set in curr_actions:
            if action_set[0] == SELF_LOOP: next_node_type = curr_node_type
            else: next_node_type = self._get_next_node_type(curr_node_type, action_set[0], action_set[1])
            enti_emb = self.kg_emb.lookup_emb(next_node_type,
                            type_index = torch.LongTensor([action_set[1]]).to(self.device))
            entities_embs.append(enti_emb)
            rela_emb = self.kg_emb.lookup_rela_emb(action_set[0])
            relation_embs.append(rela_emb)

        pad_emb = self.kg_emb.lookup_rela_emb(PADDING)
        for _ in range(self.act_dim - len(entities_embs)):
            entities_embs.append(pad_emb)
            relation_embs.append(pad_emb)

        enti_emb = th.cat(entities_embs, 0)
        rela_emb = th.cat(relation_embs, 0)

        next_action_state = th.cat([enti_emb, rela_emb], -1)
        
        return [enti_emb, next_action_state]

    def action_encoder(self, relation_emb, entitiy_emb):
        action_embedding = th.cat([relation_emb, entitiy_emb], -1)
        return action_embedding
    def _get_state_update(self, index, path):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""
        if len(path) == 1:
            user_embed = self.kg_emb.lookup_emb(USER, type_index = 
                        torch.LongTensor([path[0][-1]]).to(self.device))[0].unsqueeze(0)
            curr_node_embed = user_embed
            last_relation_embed = self.dummy_rela[path[0][-1], :, :]
            relation_embed_dual = self.dummy_rela_emb(torch.LongTensor([path[0][-1]]).to(self.device))
            relation_embed_dual = relation_embed_dual.view(self.embed_size, self.embed_size)

            st_emb = self.action_encoder(last_relation_embed, user_embed)
        else:
            last_relation, curr_node_type, curr_node_id = path[-1]
            # print('last_relation, curr_node_type, curr_node_id  = ', last_relation, curr_node_type, curr_node_id )
            curr_node_embed = self.kg_emb.lookup_emb(curr_node_type, 
                    type_index = torch.LongTensor([curr_node_id]).to(self.device))[0].unsqueeze(0)
            last_relation_embed = self.kg_emb.lookup_rela_emb(last_relation)[0].unsqueeze(0)

            relation_embed_dual = self.relation_emb(torch.LongTensor([self.rela_2_index[last_relation]]).to(self.device))
            relation_embed_dual = relation_embed_dual.view(self.embed_size, self.embed_size)

            st_emb = self.action_encoder(last_relation_embed, curr_node_embed)

        return [curr_node_embed, last_relation_embed.squeeze(), relation_embed_dual.squeeze(), st_emb]


    def update(self, optimizer, env_model, device, ent_weight, step):
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

        l2_reg = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param)
        l2_loss = self.l2_weight * l2_reg

        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss + l2_loss
        if self.training:
            optimizer.zero_grad()
            loss.backward()
            # if  step % 50 == 0:
            #     plot_grad_flow_v2(self.named_parameters(), self.args.log_dir, step)
            #     print('grad_cherck')

            optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()

