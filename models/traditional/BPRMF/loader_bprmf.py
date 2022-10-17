'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline Model, BPRMF, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from models.traditional.load_data import Data


class BPRMF_loader(Data):
    def __init__(self, args, path, batch_style='list'):
        super().__init__(args, path, batch_style)


    def as_test_feed_dict(self, model, user_batch, item_batch, drop_flag=False):

        feed_dict = {
            model.users: user_batch,
            model.pos_items: item_batch
        }

        return feed_dict  
    def as_train_feed_dict(self, model,batch_data):# users, pos_items, neg_items):
        if self.batch_style_id == 0:
            users, pos_items, neg_items = batch_data
            batch_data = {}
            batch_data['users'] = users
            batch_data['pos_items'] = pos_items
            batch_data['neg_items'] = neg_items       
        #batch_data = {}
        #batch_data['users'] = users
        #batch_data['pos_items'] = pos_items
        #batch_data['neg_items'] = neg_items
        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items']
        }

        return feed_dict   