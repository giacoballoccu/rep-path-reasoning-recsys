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
LFM1M = "lastfm"

# For future
CELL = "cell"
BEAUTY = "beauty"
CLOTH = "cloth"

DATASETS = [ML1M, LFM1M]
DATASETS_WITH_WORDS = [CELL, BEAUTY, CLOTH]

TRAIN = 'train'
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
Knowledge-Aware methods (don't produce explanations paths)
"""
CKE = 'cke'
CFKG = 'cfkg'
KGAT = 'kgat'
RIPPLE = 'ripple'
KNOWLEDGE_AWARE_METHODS = [CKE, CFKG, KGAT, RIPPLE]
"""
Path reasoning methods
"""
PGPR = 'pgpr'
CAFE = 'cafe'
UCPR = 'ucpr'
PATH_REASONING_METHODS = [PGPR, UCPR, CAFE]


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


def get_result_dir(dataset_name, model_name):
    ensure_dataset_name(dataset_name)
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


def get_uid_to_sensible_attribute(dataset_name, attribute):
    if attribute not in DATASET_SENSIBLE_ATTRIBUTE_MATRIX[dataset_name] or \
            DATASET_SENSIBLE_ATTRIBUTE_MATRIX[dataset_name][attribute] == 0:
        print("Wrong / not available sensible attribute specified")
        exit(-1)

    # Convertion to standardize human readable format
    if attribute == GENDER:
        attribute2name = {"M": "Male", "F": "Female"}
    elif attribute == AGE:
        attribute2name = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}

    # Create mapping
    uid2attribute = {}
    ensure_dataset_name(dataset_name)
    data_dir = get_data_dir(dataset_name)
    with open(data_dir + "users.txt", 'r') as users_file:
        reader = csv.reader(users_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            uid = row[0]
            gender = row[1]
            age = int(row[2])
            uid2attribute[uid] = attribute2name[(gender if attribute == GENDER else age)]
    users_file.close()

    return uid2attribute


def get_item_genre(dataset_name):
    data_dir = get_data_dir(dataset_name)
    item_genre_df = pd.read_csv(os.path.join(data_dir, "products.txt"), sep="\t", header=True)
    return dict(zip(item_genre_df.pid, item_genre_df.genre))


def get_item_count(dataset_name):
    data_dir = get_data_dir(dataset_name)
    df_items = pd.read_csv(data_dir + "products.txt", sep="\t")
    return df_items.pid.unique().shape[0]


def load_labels(dataset_name, model_name, split=TRAIN):
    if split != TRAIN or split != TEST:
        raise Exception('mode should be one of {train, test}.')
    get_tmp_dir = get_data_dir(dataset_name, model_name)
    label_path = os.path.join(get_tmp_dir, f"{split}.pkl")
    user_products = pickle.load(open(label_path, 'rb'))
    return user_products


def load_kg(dataset_name, model_name):
    kg_path = os.path.join(get_data_dir(dataset_name), model_name, "tmp", "kg.pkl")
    kg = pickle.load(open(kg_path, 'rb'))
    return kg


def load_embed(dataset_name, model_name):
    get_tmp_dir = get_data_dir(dataset_name, model_name)
    embed_file = os.path.join(get_tmp_dir, f"transe_embed.pkl")
    embeds = pickle.load(open(embed_file, 'rb'))
    return embeds
