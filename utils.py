import csv
import os
import pickle

import pandas as pd
import json
import gzip

"""
File reading and os
"""
# Datasets
ML1M = "ml1m"
LFM1M = "lfm1m"

# For future
CELL = "cell"
BEAUTY = "beauty"
CLOTH = "cloth"

DATASETS = [ML1M, LFM1M]
DATASETS_WITH_WORDS = [CELL, BEAUTY, CLOTH]
AMAZON_DATASETS = [CELL, BEAUTY, CLOTH]

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
USER = 'user'

# Type of entities interacted with user feedback
PRODUCT = 'product'
WORD = 'word'

MAIN_INTERACTION = {
    ML1M: "watched",
    LFM1M: "listened",
}
# Sensible attributes
GENDER = "gender"
AGE = "age"
OVERALL = "overall"


"""
Non-Personalized methods
"""
MOSTPOP = "most_pop"
"""
Knowledge-Aware methods (don't produce explanations paths)
"""
CKE = 'cke'
CFKG = 'cfkg'
KGAT = 'kgat'
RIPPLE = 'ripple'
BPRMF = "bprmf"
NFM = "nfm"
KNOWLEDGE_AWARE_METHODS = [CKE, CFKG, KGAT, RIPPLE, NFM, BPRMF]
FM = "fm"
"""
Path reasoning methods
"""
PGPR = 'pgpr'
CAFE = 'cafe'
UCPR = 'ucpr'
MLR = 'mlr'
PATH_REASONING_METHODS = [PGPR, UCPR, CAFE, MLR]



TRANSE = 'transe'
EMBEDDING_METHODS = [TRANSE]

def ensure_dataset_name(dataset_name):
    if dataset_name not in DATASETS:
        print("Dataset not recognised, check for typos")
        exit(-1)
    return


def get_raw_data_dir(dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/"


def get_raw_kg_dir(dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/kg/"


def get_data_dir(dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/preprocessed/"


def get_tmp_dir(dataset_name, model_name):
    return os.path.join(get_data_dir(dataset_name), model_name, "tmp")


def get_result_dir(dataset_name, model_name=None):
    ensure_dataset_name(dataset_name)
    if model_name == None:
        return f"results/{dataset_name}/"
    return f"results/{dataset_name}/{model_name}/"


def get_model_data_dir(model_name, dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/preprocessed/{model_name}/"


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


"""
Mappings
"""

DATASET_SENSIBLE_ATTRIBUTE_MATRIX = {
    ML1M: {GENDER: 1, AGE: 1},
    LFM1M: {GENDER: 1, AGE: 1},
}

def get_dataset_id2model_kg_id(dataset_name, model_name, what="user"):
    model_data_dir = get_model_data_dir(model_name, dataset_name)
    if model_name in KNOWLEDGE_AWARE_METHODS:
        model_data_dir = get_model_data_dir(model_name, dataset_name)
    file = open(os.path.join(model_data_dir, f"mappings/{what}_mapping.txt"), "r")
    csv_reader = csv.reader(file, delimiter='\t')
    dataset_pid2model_kg_pid = {}
    next(csv_reader, None)
    for row in csv_reader:
        dataset_pid2model_kg_pid[row[0]] = int(row[1])
    file.close()
    return dataset_pid2model_kg_pid

def get_uid_to_sensible_attribute(dataset_name, model_name, attribute):
    if attribute not in DATASET_SENSIBLE_ATTRIBUTE_MATRIX[dataset_name] or \
            DATASET_SENSIBLE_ATTRIBUTE_MATRIX[dataset_name][attribute] == 0:
        print("Wrong / not available sensible attribute specified")
        exit(-1)

    # Create mapping
    uid2attribute = {}
    ensure_dataset_name(dataset_name)
    data_dir = get_data_dir(dataset_name)

    dataset_pid2model_kg_pid = get_dataset_id2model_kg_id(dataset_name, model_name, what="user")
    with open(data_dir + "users.txt", 'r') as users_file:
        reader = csv.reader(users_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            uid = dataset_pid2model_kg_pid[row[0]]
            age = row[1]
            gender = row[2]
            uid2attribute[uid] = gender if attribute == GENDER else age
    users_file.close()

    return uid2attribute


def get_item_genre(dataset_name, model_name):
    data_dir = get_data_dir(dataset_name)
    dataset_id2model_kg_id = get_dataset_id2model_kg_id(dataset_name, model_name, "product")
    dataset_id2model_kg_id = dict(zip([int(x) for x in dataset_id2model_kg_id.keys()], dataset_id2model_kg_id.values())) #TODO FIX SPAGHETTI
    item_genre_df = pd.read_csv(os.path.join(data_dir, "products.txt"), sep="\t")
    item_genre_df.pid = item_genre_df.pid.map(dataset_id2model_kg_id)
    return dict(zip(item_genre_df.pid, item_genre_df.genre))

def get_mostpop_topk(dataset_name, model_name, k):
    result_dir = get_result_dir(dataset_name)
    with open(os.path.join(result_dir, "most_pop", "item_topks.pkl"), 'rb') as f:
        most_pop_topks = pickle.load(f)
    f.close()
    dataset_uid2model_kg_id = get_dataset_id2model_kg_id(dataset_name, model_name, "user")
    dataset_pid2model_kg_id = get_dataset_id2model_kg_id(dataset_name, model_name, "product")
    most_pop_topks = {dataset_uid2model_kg_id[uid]: [dataset_pid2model_kg_id[pid] for pid in topk[:k]]
                      for uid, topk in most_pop_topks.items()}
    return most_pop_topks

def get_item_count(dataset_name):
    data_dir = get_data_dir(dataset_name)
    df_items = pd.read_csv(data_dir + "products.txt", sep="\t")
    return df_items.pid.unique().shape[0]

def get_item_pop(dataset_name, model_name):
    data_dir = get_data_dir(dataset_name)
    dataset_id2model_kg_id = get_dataset_id2model_kg_id(dataset_name, model_name,"product")
    dataset_id2model_kg_id = dict(zip([int(x) for x in dataset_id2model_kg_id.keys()], dataset_id2model_kg_id.values())) #TODO FIX SPAGHETTI
    df_items = pd.read_csv(data_dir + "products.txt", sep="\t")
    df_items.pid = df_items.pid.map(dataset_id2model_kg_id)
    return dict(zip(df_items.pid, df_items.pop_item))

def get_item_provider_pop(dataset_name, model_name):
    data_dir = get_data_dir(dataset_name)
    dataset_id2model_kg_id = get_dataset_id2model_kg_id(dataset_name, model_name,"product")
    dataset_id2model_kg_id = dict(zip([int(x) for x in dataset_id2model_kg_id.keys()], dataset_id2model_kg_id.values())) #TODO FIX SPAGHETTI
    df_items = pd.read_csv(data_dir + "products.txt", sep="\t")
    df_items.pid = df_items.pid.map(dataset_id2model_kg_id)
    return dict(zip(df_items.pid, df_items.pop_provider))


def load_labels(dataset_name, model_name, split=TRAIN): #TODO MAKE IT AGNOSITC WITH MODEL.CALLS
    if split != TRAIN and split != VALID and split != TEST:
        raise Exception('mode should be one of {train, valid, test}.')
    if model_name == PGPR or model_name == UCPR:
        tmp_dir = get_tmp_dir(dataset_name, model_name)
        label_path = os.path.join(tmp_dir, f"{split}_label.pkl")
        user_products = pickle.load(open(label_path, 'rb'))
    elif model_name == CAFE:
        user_products = {}
        model_data_dir = get_model_data_dir(model_name, dataset_name)
        label_path = os.path.join(model_data_dir, f"{split}.txt.gz")
        with gzip.open(label_path, 'rt') as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                user_products[int(row[0])] = [int(pid) for pid in row[1:]]
    elif model_name in [KGAT, CKE, CFKG]:
        user_products = {}
        model_data_dir = get_model_data_dir(model_name, dataset_name)
        label_path = os.path.join(model_data_dir, f"{split}.txt")
        with open(label_path, 'rt') as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                user_products[int(row[0])] = [int(pid) for pid in row[1:]]
    return user_products


def load_kg(dataset_name, model_name):
    if model_name == PGPR:
        from models.PGPR.knowledge_graph import KnowledgeGraph
        kg_path = os.path.join(get_data_dir(dataset_name), model_name, "tmp", "kg.pkl")
    if model_name == CAFE:
        from models.CAFE.knowledge_graph import KnowledgeGraph
        kg_path = os.path.join(get_data_dir(dataset_name), model_name, "tmp", "kg.pkl")
    if model_name == UCPR:
        from models.UCPR.preprocess.knowledge_graph import KnowledgeGraph
        kg_path = os.path.join(get_data_dir(dataset_name), model_name, "tmp", "kg.pkl")
    kg = pickle.load(open(kg_path, 'rb'))
    return kg


def load_embed(dataset_name, model_name): #TODO MAKE IT AGNOSITC WITH MODEL.CALLS
    tmp_dir = get_tmp_dir(dataset_name, model_name)
    if model_name == PGPR or model_name == UCPR:
        embed_file = os.path.join(tmp_dir, f"transe_embed.pkl")
    elif model_name == CAFE:
        embed_file = os.path.join(tmp_dir, f"embed.pkl")
    embeds = pickle.load(open(embed_file, 'rb'))
    return embeds

def get_kb_id2dataset_id(dataset_name):
    input_folder = get_data_dir(dataset_name)
    eid2kg_df = pd.read_csv(os.path.join(input_folder, "i2kg_map.txt"), sep="\t")
    eid2kg_id = dict(zip(eid2kg_df.eid, eid2kg_df.pid))
    return eid2kg_id
