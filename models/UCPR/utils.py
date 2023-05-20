from __future__ import absolute_import, division, print_function
from easydict import EasyDict as edict

import os
import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import csv
# import scipy.sparse as sp
import torch
from collections import defaultdict
import shutil
# Dataset names.
# from sklearn.feature_extraction.text import TfidfTransformer

ML1M = 'ml1m'
LFM1M = 'lfm1m'
CELL = 'cellphones'

ROOT_DIR = os.environ('TREX_DATA_ROOT') if 'TREX_DATA_ROOT' in os.environ else '../..'

# STILL NOT SUPPORTED = beauty, cell, cloth
BEAUTY_CORE ='beauty'
CELL_CORE = CELL
CLOTH_CORE = 'cloth'
# retro compatiblity 
MOVIE_CORE = 'ml1m'
AZ_BOOK_CORE = 'book'
MODEL = 'ucpr'
TRANSE='transe'
# Dataset directories.
DATASET_DIR = {
    ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed/{MODEL}',
    LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed/{MODEL}',
    CELL: f'{ROOT_DIR}/data/{CELL}/preprocessed/{MODEL}'
}

# Model result directories.
TMP_DIR = {
    ML1M: f'{DATASET_DIR[ML1M]}/tmp',
    LFM1M: f'{DATASET_DIR[LFM1M]}/tmp',
    CELL: f'{DATASET_DIR[CELL]}/tmp',
}

VALID_METRICS_FILE_NAME = 'valid_metrics.json'


TRANSE_OPT_METRIC = 'valid_loss'
#OPTIM_HPARAMS_METRIC = 'avg_valid_reward'
OPTIM_HPARAMS_METRIC = 'valid_reward'
OPTIM_HPARAMS_LAST_K = 100 # last 100 episodes
#OPTIM_HPARAMS_METRIC = 'ndcg'
LOG_DIR = f'{ROOT_DIR}/results'


LOG_DATASET_DIR = {
    ML1M: f'{LOG_DIR}/{ML1M}/{MODEL}',
    LFM1M: f'{LOG_DIR}/{LFM1M}/{MODEL}',
    CELL: f'{LOG_DIR}/{CELL}/{MODEL}',
}

# for compatibility, CFG_DIR, BEST_CFG_DIR have been modified s,t, they are independent from the dataset
CFG_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/hparams_cfg',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/hparams_cfg',
    CELL: f'{LOG_DATASET_DIR[CELL]}/hparams_cfg',
}
BEST_CFG_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/best_hparams_cfg',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/best_hparams_cfg',
    CELL: f'{LOG_DATASET_DIR[CELL]}/best_hparams_cfg',
}



TEST_METRICS_FILE_NAME = 'test_metrics.json'
RECOM_METRICS_FILE_NAME = 'recommender_metrics.json'

RECOM_METRICS_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{RECOM_METRICS_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{RECOM_METRICS_FILE_NAME}',
    CELL: f'{CFG_DIR[CELL]}/{RECOM_METRICS_FILE_NAME}',
}

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

TRANSE_TEST_METRICS_FILE_NAME = 'test_metrics_transe.json'
TRANSE_CFG_FILE_NAME = 'config_transe.json'
TRANSE_TEST_METRICS_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{TRANSE_TEST_METRICS_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{TRANSE_TEST_METRICS_FILE_NAME}',
    CELL: f'{CFG_DIR[CELL]}/{TRANSE_TEST_METRICS_FILE_NAME}',
}
BEST_TRANSE_TEST_METRICS_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{TRANSE_TEST_METRICS_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{TRANSE_TEST_METRICS_FILE_NAME}',
    CELL: f'{BEST_CFG_DIR[CELL]}/{TRANSE_TEST_METRICS_FILE_NAME}',
}
TRANSE_CFG_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{TRANSE_CFG_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{TRANSE_CFG_FILE_NAME}',
    CELL: f'{CFG_DIR[CELL]}/{TRANSE_CFG_FILE_NAME}',
}
BEST_TRANSE_CFG_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{TRANSE_CFG_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{TRANSE_CFG_FILE_NAME}',
    CELL: f'{BEST_CFG_DIR[CELL]}/{TRANSE_CFG_FILE_NAME}',
}










TRANSE_HPARAMS_FILE = f'transe_{MODEL}_hparams_file.json'
HPARAMS_FILE = f'{MODEL}_hparams_file.json'




SAVE_MODEL_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/save',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/save',
    CELL: f'{LOG_DATASET_DIR[CELL]}/save',
}


EVALUATION = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/eva_pre',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/eva_pre',
    CELL: f'{LOG_DATASET_DIR[CELL]}/eva_pre',
}

EVALUATION_2 = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/eval',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/eval',
    CELL: f'{LOG_DATASET_DIR[CELL]}/eval',
}

CASE_ST = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/case_st',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/case_st',
    CELL: f'{LOG_DATASET_DIR[CELL]}/case_st',
}

TEST = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/test',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/test',
    CELL: f'{LOG_DATASET_DIR[CELL]}/test',
}



# Label files.
LABELS = {
    ML1M: (TMP_DIR[ML1M] + '/train_label.pkl', TMP_DIR[ML1M] + '/valid_label.pkl', TMP_DIR[ML1M] + '/test_label.pkl'),
    LFM1M: (TMP_DIR[LFM1M] + '/train_label.pkl', TMP_DIR[LFM1M] + '/valid_label.pkl', TMP_DIR[LFM1M] + '/test_label.pkl'),
    CELL: (TMP_DIR[CELL] + '/train_label.pkl', TMP_DIR[CELL] + '/valid_label.pkl', TMP_DIR[CELL] + '/test_label.pkl')
}

# UCPR SPECIFIC RELATIONS
PADDING = 'padding'
SELF_LOOP = 'self_loop'



# ENTITIES/RELATIONS SHARED BY ALL DATASETS
USER = 'user'
PRODUCT = 'product'
INTERACTION = {
    ML1M: "watched",
    LFM1M: "listened",
    CELL: "purchase",
}
SELF_LOOP = 'self_loop'
PRODUCED_BY_PRODUCER = 'produced_by_producer'
PRODUCER = 'producer'

# ML1M ENTITIES
CINEMATOGRAPHER = 'cinematographer'
PRODCOMPANY = 'prodcompany'
COMPOSER = 'composer'
CATEGORY = 'category'
ACTOR = 'actor'
COUNTRY = 'country'
WIKIPAGE = 'wikipage'
EDITOR = 'editor'
WRITTER = 'writter'
DIRECTOR = 'director'

# LASTFM ENTITIES
ARTIST = 'artist'
ENGINEER = 'engineer'
GENRE = 'genre'

# CELL ENTITIES
BRAND = 'brand'
RPRODUCT = 'rproduct'

# ML1M RELATIONS
DIRECTED_BY_DIRECTOR = 'directed_by_director'
PRODUCED_BY_COMPANY = 'produced_by_prodcompany'
STARRED_BY_ACTOR = 'starred_by_actor'
RELATED_TO_WIKIPAGE = 'related_to_wikipage'
EDITED_BY_EDITOR = 'edited_by_editor'
WROTE_BY_WRITTER = 'wrote_by_writter'
CINEMATOGRAPHY_BY_CINEMATOGRAPHER = 'cinematography_by_cinematographer'
COMPOSED_BY_COMPOSER = 'composed_by_composer'
PRODUCED_IN_COUNTRY = 'produced_in_country'
BELONG_TO_CATEGORY = 'belong_to_category'

# LASTFM RELATIONS
MIXED_BY_ENGINEER = 'mixed_by_engineer'
FEATURED_BY_ARTIST = 'featured_by_artist'
BELONG_TO_GENRE = 'belong_to_genre'

# CELL RELATIONS
PURCHASE = 'purchase'
ALSO_BOUGHT_RP = 'also_bought_related_product'
ALSO_VIEWED_RP = 'also_viewed_related_product'
ALSO_BOUGHT_P = 'also_bought_product'
ALSO_VIEWED_P = 'also_viewed_product'





KG_RELATION = {
    ML1M: {
        USER: {
            INTERACTION[ML1M]: PRODUCT,
        },
        ACTOR: {
            STARRED_BY_ACTOR: PRODUCT,
        },
        DIRECTOR: {
            DIRECTED_BY_DIRECTOR: PRODUCT,
        },
        PRODUCT: {
            INTERACTION[ML1M]: USER,
            PRODUCED_BY_COMPANY: PRODCOMPANY,
            PRODUCED_BY_PRODUCER: PRODUCER,
            EDITED_BY_EDITOR: EDITOR,
            WROTE_BY_WRITTER: WRITTER,
            CINEMATOGRAPHY_BY_CINEMATOGRAPHER: CINEMATOGRAPHER,
            BELONG_TO_CATEGORY: CATEGORY,
            DIRECTED_BY_DIRECTOR: DIRECTOR,
            STARRED_BY_ACTOR: ACTOR,
            COMPOSED_BY_COMPOSER: COMPOSER,
            PRODUCED_IN_COUNTRY: COUNTRY,
            RELATED_TO_WIKIPAGE: WIKIPAGE,
        },
        PRODCOMPANY: {
            PRODUCED_BY_COMPANY: PRODUCT,
        },
        COMPOSER: {
            COMPOSED_BY_COMPOSER: PRODUCT,
        },
        PRODUCER: {
            PRODUCED_BY_PRODUCER: PRODUCT,
        },
        WRITTER: {
            WROTE_BY_WRITTER: PRODUCT,
        },
        EDITOR: {
            EDITED_BY_EDITOR: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO_CATEGORY: PRODUCT,
        },
        CINEMATOGRAPHER: {
            CINEMATOGRAPHY_BY_CINEMATOGRAPHER: PRODUCT,
        },
        COUNTRY: {
            PRODUCED_IN_COUNTRY: PRODUCT,
        },
        WIKIPAGE: {
            RELATED_TO_WIKIPAGE: PRODUCT,
        }
    },
    LFM1M: {
        USER: {
            INTERACTION[LFM1M]: PRODUCT,
        },
        ARTIST: {
            FEATURED_BY_ARTIST: PRODUCT,
        },
        ENGINEER: {
            MIXED_BY_ENGINEER: PRODUCT,
        },
        PRODUCT: {
            INTERACTION[LFM1M]: USER,
            PRODUCED_BY_PRODUCER: PRODUCER,
            FEATURED_BY_ARTIST: ARTIST,
            MIXED_BY_ENGINEER: ENGINEER,
            BELONG_TO_GENRE: GENRE,
        },
        PRODUCER: {
            PRODUCED_BY_PRODUCER: PRODUCT,
        },
        GENRE: {
            BELONG_TO_GENRE: PRODUCT,
        },
    },
    CELL: {
        USER: {
            PURCHASE: PRODUCT,
        },
        PRODUCT: {
            PURCHASE: USER,
            PRODUCED_BY_COMPANY: BRAND,
            BELONG_TO_CATEGORY: CATEGORY,
            ALSO_BOUGHT_RP: RPRODUCT,
            ALSO_VIEWED_RP: RPRODUCT,
            ALSO_BOUGHT_P: PRODUCT,
            ALSO_VIEWED_P: PRODUCT,
        },
        BRAND: {
            PRODUCED_BY_COMPANY: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO_CATEGORY: PRODUCT,
        },
        RPRODUCT: {
            ALSO_BOUGHT_RP: PRODUCT,
            ALSO_VIEWED_RP: PRODUCT,
        }
    },
}

# 0 is reserved to the main relation, 1 to mention
PATH_PATTERN = {
    ML1M: {
        0: ((None, USER), (INTERACTION[ML1M], PRODUCT), (INTERACTION[ML1M], USER), (INTERACTION[ML1M], PRODUCT)),
        2: ((None, USER), (INTERACTION[ML1M], PRODUCT), (CINEMATOGRAPHY_BY_CINEMATOGRAPHER, CINEMATOGRAPHER), (CINEMATOGRAPHY_BY_CINEMATOGRAPHER, PRODUCT)),
        3: ((None, USER), (INTERACTION[ML1M], PRODUCT), (PRODUCED_BY_COMPANY, PRODCOMPANY), (PRODUCED_BY_COMPANY, PRODUCT)),
        4: ((None, USER), (INTERACTION[ML1M], PRODUCT), (COMPOSED_BY_COMPOSER, COMPOSER), (COMPOSED_BY_COMPOSER, PRODUCT)),
        5: ((None, USER), (INTERACTION[ML1M], PRODUCT), (BELONG_TO_CATEGORY, CATEGORY), (BELONG_TO_CATEGORY, PRODUCT)),
        7: ((None, USER), (INTERACTION[ML1M], PRODUCT), (STARRED_BY_ACTOR, ACTOR), (STARRED_BY_ACTOR, PRODUCT)),
        8: ((None, USER), (INTERACTION[ML1M], PRODUCT), (EDITED_BY_EDITOR, EDITOR), (EDITED_BY_EDITOR, PRODUCT)),
        9: ((None, USER), (INTERACTION[ML1M], PRODUCT), (PRODUCED_BY_PRODUCER, PRODUCER), (PRODUCED_BY_PRODUCER, PRODUCT)),
        10: ((None, USER), (INTERACTION[ML1M], PRODUCT), (WROTE_BY_WRITTER, WRITTER), (WROTE_BY_WRITTER, PRODUCT)),
        11: ((None, USER), (INTERACTION[ML1M], PRODUCT), (DIRECTED_BY_DIRECTOR, DIRECTOR), (DIRECTED_BY_DIRECTOR, PRODUCT)),
        12: ((None, USER), (INTERACTION[ML1M], PRODUCT), (PRODUCED_IN_COUNTRY, COUNTRY), (PRODUCED_IN_COUNTRY, PRODUCT)),
        13: ((None, USER), (INTERACTION[ML1M], PRODUCT), (RELATED_TO_WIKIPAGE, WIKIPAGE), (RELATED_TO_WIKIPAGE, PRODUCT)),
    },
    LFM1M: {
        0: ((None, USER), (INTERACTION[LFM1M], PRODUCT), (INTERACTION[LFM1M], USER), (INTERACTION[LFM1M], PRODUCT)),
        2: ((None, USER), (INTERACTION[LFM1M], PRODUCT), (BELONG_TO_GENRE, GENRE), (BELONG_TO_GENRE, PRODUCT)),
        4: ((None, USER), (INTERACTION[LFM1M], PRODUCT), (FEATURED_BY_ARTIST, ARTIST), (FEATURED_BY_ARTIST, PRODUCT)),
        5: ((None, USER), (INTERACTION[LFM1M], PRODUCT), (MIXED_BY_ENGINEER, ENGINEER), (MIXED_BY_ENGINEER, PRODUCT)),
        6: ((None, USER), (INTERACTION[LFM1M], PRODUCT), (PRODUCED_BY_PRODUCER, PRODUCER), (PRODUCED_BY_PRODUCER, PRODUCT)),
    },
    CELL: {
        0: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
        2: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO_CATEGORY, CATEGORY), (BELONG_TO_CATEGORY, PRODUCT)),
        3: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY_COMPANY, BRAND), (PRODUCED_BY_COMPANY, PRODUCT)),
        4: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT_P, PRODUCT)),
        5: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED_P, PRODUCT)),
        6: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT_RP, RPRODUCT), (ALSO_BOUGHT_RP, PRODUCT)),
        10: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED_RP, RPRODUCT), (ALSO_VIEWED_RP, PRODUCT)),
    }
}

MAIN_PRODUCT_INTERACTION = {
    ML1M: (PRODUCT, INTERACTION[ML1M]),
    LFM1M: (PRODUCT, INTERACTION[LFM1M]),
    CELL: (PRODUCT, PURCHASE)
}


def get_entities(dataset_name):
    return list(KG_RELATION[dataset_name].keys())


def get_knowledge_derived_relations(dataset_name):
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    ans = list(KG_RELATION[dataset_name][main_entity].keys())
    ans.remove(main_relation)
    return ans



def get_dataset_relations(dataset_name, entity_head):    
    return list(KG_RELATION[dataset_name][entity_head].keys())




def get_entity_tail(dataset_name, relation):
    entity_head, _ = MAIN_PRODUCT_INTERACTION[dataset_name]
    return KG_RELATION[dataset_name][entity_head][relation]



def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(dataset, dataset_obj):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_labels(dataset, labels, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'valid':
        label_file = LABELS[dataset][1]
    elif mode == 'test':
        label_file = LABELS[dataset][2]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)
    f.close()

def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'valid':
        label_file = LABELS[dataset][1]
    elif mode == 'test':
        label_file = LABELS[dataset][2]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products


def save_embed(dataset, embed):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    pickle.dump(embed, open(embed_file, 'wb'))


def load_embed(dataset, embed_model=TRANSE):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    print('Load embedding:', embed_file, ' for ', dataset)
    if not os.path.exists(embed_file):
        default_emb_path = os.path.join(ROOT_DIR, 'pretrained', dataset, MODEL, embed_model, 'transe_embed.pkl')
        shutil.copyfile(default_emb_path, embed_file)

    embed = pickle.load(open(embed_file, 'rb'))
    print('Loaded: ', embed.keys())
    return embed


# Receive paths in form (score, prob, [path]) return the last relationship
def get_path_pattern(path):
    return path[-1][-1][0]


def get_pid_to_kgid_mapping(dataset_name):
    if dataset_name == "ml1m":
        file = open(DATASET_DIR[dataset_name] + "/entities/mappings/movie.txt", "r")
    elif dataset_name == "lfm1m":
        file = open(DATASET_DIR[dataset_name] + "/entities/mappings/song.txt", "r")
    else:
        print("Dataset mapping not found!")
        exit(-1)
    reader = csv.reader(file, delimiter=' ')
    dataset_pid2kg_pid = {}
    next(reader, None)
    for row in reader:
        if dataset_name == "ml1m" or dataset_name == "lfm1m":
            dataset_pid2kg_pid[int(row[0])] = int(row[1])
    file.close()
    return dataset_pid2kg_pid


def get_validation_pids(dataset_name):
    if not os.path.isfile(os.path.join(DATASET_DIR[dataset_name], 'valid.txt')):
        return []
    validation_pids = defaultdict(set)
    with open(os.path.join(DATASET_DIR[dataset_name], 'valid.txt')) as valid_file:
        reader = csv.reader(valid_file, delimiter=" ")
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            validation_pids[uid].add(pid)
    valid_file.close()
    return validation_pids


def get_uid_to_kgid_mapping(dataset_name):
    dataset_uid2kg_uid = {}
    with open(DATASET_DIR[dataset_name] + "/entities/mappings/user.txt", 'r') as file:
        reader = csv.reader(file, delimiter=" ")
        next(reader, None)
        for row in reader:
            if dataset_name == "ml1m" or dataset_name == "lfm1m":
                uid_review = int(row[0])
            uid_kg = int(row[1])
            dataset_uid2kg_uid[uid_review] = uid_kg
    return dataset_uid2kg_uid


def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    # CHANGED
    kg = pickle.load(open(kg_file, 'rb'))
    return kg


def shuffle(arr):
    for i in range(len(arr) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i + 1)

        # Swap arr[i] with the element at random index
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def makedirs(dataset_name):
    os.makedirs(BEST_CFG_DIR[dataset_name], exist_ok=True)
    os.makedirs(CFG_DIR[dataset_name], exist_ok=True)
