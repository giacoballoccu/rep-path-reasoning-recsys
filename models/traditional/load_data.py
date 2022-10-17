
import collections
import os

import numpy as np
import random as rd
import torch
import torch.utils.data
from torch.utils.data import Dataset
import math
class Data(Dataset):

    def __init__(self, args, path, batch_style='list'):
        super(Data).__init__()

        self.batch_styles = {'list': 0, 'map': 1}
        assert batch_style in list(
            self.batch_styles.keys()), f'Error: got {batch_style} but valid batch styles are {list(self.batch_styles.keys())}'
        self.path = path
        self.args = args
        self.batch_style = batch_style
        self.batch_style_id = self.batch_styles[self.batch_style]

        self.batch_size = args.batch_size

        train_file = os.path.join(path, 'preprocessed/kgat/train.txt')
        valid_file = os.path.join(path, 'preprocessed/kgat/valid.txt')
        test_file = os.path.join(path, 'preprocessed/kgat/test.txt')

        kg_file = os.path.join(path, 'preprocessed/kgat/kg_final.txt')

        # ----------get number of users and items & then load rating data from train_file & test_file------------.
        self.n_train, self.n_valid, self.n_test = 0, 0, 0

        self.n_users, self.n_items = 0, 0

        self.train_data, self.train_user_dict = self._load_ratings(train_file)
        self.valid_data, self.valid_user_dict = self._load_ratings(valid_file)
        self.test_data, self.test_user_dict = self._load_ratings(test_file)
        self.exist_users = list(self.train_user_dict.keys())
        self.N_exist_users = len(self.exist_users)

        self._statistic_ratings()

        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)

        # ----------print the basic info about the dataset-------------.
        self.batch_size_kg = self.n_triples // (self.n_train // self.batch_size)
        self._print_data_info()

    # reading train & test interaction data.
    def _load_ratings(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict

    def _statistic_ratings(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(max(self.train_data[:, 1]), max(self.valid_data[:, 1])),
                           max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test = len(self.test_data)

    # reading train & test interaction data.
    def _load_kg(self, file_name):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        # self.n_relations = len(set(kg_np[:, 1]))
        # self.n_entities = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_triples = len(kg_np)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _print_data_info(self):
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))
        print('[batch_size, batch_size_kg]=[%d, %d]' % (self.batch_size, self.batch_size_kg))

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            test_iids = self.test_user_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

    def __len__(self):
        # number of existing users after the preprocessing described in the paper,
        # determines the length of the training dataset, for which a positive an negative are extracted
        return len(self.exist_users)

    ##_generate_train_cf_batch
    def __getitem__(self, idx):
        """
        if self.batch_size <= self.n_users:
            user = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        """

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items, size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        """
        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        """
        u = self.exist_users[idx]
        pos_item = sample_pos_items_for_u(u, 1)
        neg_item = sample_neg_items_for_u(u, 1)
        if len(pos_item) == 1:
            pos_item = pos_item[0]
        if len(neg_item) == 1:
            neg_item = neg_item[0]

        if self.batch_style_id == 0:
            return u, pos_item, neg_item
        else:
            return {'users': u, 'pos_items': pos_item,
                    'neg_items': neg_item}  # u, pos_item, neg_item #users, pos_items, neg_items

    def as_test_feed_dict(self, model, user_batch, item_batch, drop_flag=True):

        feed_dict = {
            model.users: user_batch,
            model.pos_items: item_batch,
            model.mess_dropout: [0.] * len(eval(self.args.layer_size)),
            model.node_dropout: [0.] * len(eval(self.args.layer_size)),

        }

        return feed_dict

    def as_train_feed_dict(self, model, batch_data):
        if self.batch_style_id == 0:
            users, pos_items, neg_items = batch_data
            batch_data = {}
            batch_data['users'] = users
            batch_data['pos_items'] = pos_items
            batch_data['neg_items'] = neg_items

        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items'],

            model.mess_dropout: eval(self.args.mess_dropout),
            model.node_dropout: eval(self.args.node_dropout),
        }

        return feed_dict
