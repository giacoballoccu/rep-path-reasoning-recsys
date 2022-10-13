from __future__ import absolute_import, division, print_function

import os
import numpy as np
import gzip
import pickle
from easydict import EasyDict as edict
import random
from UCPR.utils import *
import pandas as pd
import random



class Dataset(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, args, set_name='train', word_sampling_rate=1e-4):
        self.dataset_name = args.dataset
        self.data_dir = DATASET_DIR[self.dataset_name]
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt.gz'
        entity_filename_edict, relation_filename_edict = self.infer_kg_structure()
        #Other relation names doesn't contain the main interaction
        self.entity_names, self.other_relation_names = list(entity_filename_edict.keys()), list(relation_filename_edict.keys())
        #self.relation_names =  self.other_relation_names + [INTERACTION[self.dataset_name] ]
        print(self.relation_names)
        self.load_entities(entity_filename_edict)
        self.load_product_relations(relation_filename_edict)
        self.load_reviews()

    #This will not work if your entities are composed by multiple words, e.g. if you name an entity related_product
    #this script will consider as a relation, please use a single word for relations
    def infer_kg_structure(self):
        file_list = os.listdir(self.data_dir)
        entity_filenames = [filename for filename in file_list if len(filename.split("_")) == 1]
        entity_filename_edict = edict()
        entity_names = []
        for entity_file in entity_filenames:
            if os.path.isdir(os.path.join(self.data_dir, entity_file)): continue
            name = entity_file.split(".")[0]
            if name in ["train", "valid", "test"]: continue
            entity_names.append(name)
            entity_filename_edict[name] = entity_file

        relation_filenames = [filename for filename in file_list if len(filename.split("_")) > 1]
        relation_filename_edict = edict()
        relation_names = []
        for relation_file in relation_filenames:
            name = relation_file.split(".")[0]
            relation_names.append(name)
            relation_filename_edict[name] = relation_file

        self.relation2entity = {}
        for rel_name in relation_names:
            entity_name = rel_name.split("_")[-1]
            self.relation2entity[rel_name] = entity_name

        self.relation2entity[INTERACTION[self.dataset_name]] = MAIN_PRODUCT_INTERACTION[self.dataset_name][0] 
        
        return entity_filename_edict, relation_filename_edict

    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            return [line.decode('utf-8').strip() for line in f]

    def load_entities(self, entity_filename_edict):
        """Load 10 global entities from data files:
        'user','movie','actor','director','producer','production_company','category','editor','writter','cinematographer'
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """
        for name in entity_filename_edict:
            vocab = [x.split("\t")[0] for x in self._load_file(entity_filename_edict[name])][1:] #Remove header
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab) + 1))
            print('Load', name, 'of size', len(vocab))

    def load_reviews(self):
        """Load user-product reviews from train/test data files.
        Create member variable `review` associated with following attributes:
        - `data`: list of tuples (user_idx, product_idx, [word_idx...]).
        - `size`: number of reviews.
        - `product_distrib`: product vocab frequency among all eviews.
        - `product_uniform_distrib`: product vocab frequency (all 1's)
        - `word_distrib`: word vocab frequency among all reviews.
        - `review_count`: number of words (including duplicates).
        - `review_distrib`: always 1.
        """
        review_data = []  # (user_idx, product_idx, rating out of 5, timestamp)
        product_distrib = np.zeros(self.product.vocab_size)
        invalid_users = 0
        invalid_pid = 0
        for line in self._load_file(self.review_file):
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            rating = int(arr[2])
            timestamp = int(arr[3])
            review_data.append((user_idx, product_idx, rating, timestamp))
            product_distrib[product_idx] += 1
        print(f"Invalid users: {invalid_users}, invalid items: {invalid_pid}")
        self.review = edict(
            data=review_data,
            size=len(review_data),
            product_distrib=product_distrib,
            product_uniform_distrib=np.ones(self.product.vocab_size),
            review_count=len(review_data),
            review_distrib=np.ones(len(review_data))  # set to 1 now
        )

        print('Load review of size', self.review.size)

    def load_product_relations(self, relation_filename_edict):
        """Load 8 product -> ? relations:
        - 'directed_by': movie -> director
        - 'produced_by_company': movie->production_company,
        - 'produced_by_producer': movie->producer,
        - 'starring': movie->actor,
        - 'belong_to': movie->category,
        - 'edited_by': movie->editor,
        - 'written_by': movie->writter,
        - 'cinematography': movie->cinematographer,

        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """
        product_relations = edict()
        for rel_name, rel_filename in relation_filename_edict.items():
            entity_name = self.relation2entity[rel_name]
            product_relations[rel_name] = (rel_filename, getattr(self, entity_name))
        #E.g:
        #    product_relations = edict(
        #        belong_to=("belong_to_genre.txt.gz", self.genre),
        #        featured_by=("featured_by_artist.txt.gz", self.artist),
        #        mixed_by=('mixed_by_engineer.txt.gz', self.engineer),
        #        produced_by=('produced_by_producer.txt.gz', self.producer),
        #    )


        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            relation = edict(
                data=[],
                et_vocab=product_relations[name][1].vocab,  # copy of brand, catgory ... 's vocab
                et_distrib=np.zeros(product_relations[name][1].vocab_size)  # [1] means self.brand ..
            )
            size = 0
            for line in self._load_file(product_relations[name][0]):  # [0] means brand_p_b.txt.gz ..
                knowledge = []
                line = line.split('\t')
                for x in line:  # some lines may be empty
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                        size += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load', name, 'of size', size)


class DataLoader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.review_size = self.dataset.review.size
        self.product_relations = get_knowledge_derived_relations(dataset.dataset_name)
        self.finished_review_num = 0
        self.reset()

    def reset(self):
        # Shuffle reviews order
        self.review_seq = np.random.permutation(self.review_size)
        self.cur_review_i = 0
        self.cur_word_i = 0
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x n_relations], where each row contains
        (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        batch = []
        review_idx = self.review_seq[self.cur_review_i]
        user_idx, product_idx, rating, _ = self.dataset.review.data[review_idx]
        product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in
                             self.product_relations}  # DEFINES THE ORDER OF BATCH_IDX

        while len(batch) < self.batch_size:
            data = [user_idx, product_idx]
            for pr in self.product_relations:
                if len(product_knowledge[pr]) <= 0:
                    data.append(-1)
                else:
                    data.append(random.choice(product_knowledge[pr]))
            batch.append(data)

            self.cur_review_i += 1
            self.finished_review_num += 1
            if self.cur_review_i >= self.review_size:
                self._has_next = False
                break
            review_idx = self.review_seq[self.cur_review_i]
            user_idx, product_idx, rating, _ = self.dataset.review.data[review_idx]
            product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations}
        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next