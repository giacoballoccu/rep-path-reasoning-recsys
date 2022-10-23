'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline Model, CKE, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
from models.knowledge_aware.load_data import Data


class CKE_loader(Data):
    def __init__(self, args, path, batch_style='list'):
        super().__init__(args, path, batch_style)

        self.exist_heads = list(self.kg_dict.keys())
        self.N_exist_heads = len(self.exist_heads)
    def __getitem__(self, idx): 
        
        def sample_pos_triples_for_h(h, num):
            # pos triples associated with head entity h.
            # format of kg_dict is {h: [t,r]}.
            pos_triples = self.kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break

                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]

                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_entities, size=1)[0]

                if (t, r) not in self.kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts
        h = self.exist_heads[idx]
        pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
        neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)

        if len(pos_rs) == 1:
            pos_rs = pos_rs[0]  
        if len(pos_ts) == 1:
            pos_ts = pos_ts[0]  
        if len(neg_ts) == 1:
            neg_ts = neg_ts[0]              
        
        if self.batch_style_id == 0:
            return h, pos_rs, pos_ts, neg_ts
        else:
            return {'heads': h, 'relations': pos_rs, 'pos_tails':pos_ts, 'neg_tails':neg_ts}   





    def as_train_feed_dict(self, model, batch_data):#users, pos_items, neg_items, heads, relations, pos_tails, neg_tails ):
        if self.batch_style_id == 0:
            users, pos_items, neg_items, heads, relations, pos_tails, neg_tails = batch_data
            batch_data = {}
            batch_data['users'] = users
            batch_data['pos_items'] = pos_items
            batch_data['neg_items'] = neg_items
            batch_data['heads'] = heads
            batch_data['relations'] = relations
            batch_data['pos_tails'] = pos_tails
            batch_data['neg_tails'] = neg_tails            


        feed_dict ={
            model.u: batch_data['users'],
            model.pos_i: batch_data['pos_items'],
            model.neg_i: batch_data['neg_items'],

            model.h: batch_data['heads'],
            model.r: batch_data['relations'],
            model.pos_t: batch_data['pos_tails'],
            model.neg_t: batch_data['neg_tails']
        }
        

        return feed_dict

        
    def as_test_feed_dict(self, model, user_batch, item_batch, drop_flag=False):
        feed_dict = {
            model.u: user_batch,
            model.pos_i: item_batch
        }

        return feed_dict  




    def __len__(self):
        # number of existing heads after the preprocessing described in the paper, 
        # determines the length of the training dataset, for which a positive an negative are extracted
        return self.N_exist_heads
    





