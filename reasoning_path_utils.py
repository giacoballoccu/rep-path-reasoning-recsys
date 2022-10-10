from collections import defaultdict

import numpy as np
from utils import *
from easydict import EasyDict as edict

LIR = "lir"
SEP = "sep"
PTD = "ptd"
LID = "lid"
SED = "sed"
PTC = "ptc"
PPT = "ppt"
PATH_QUALITY_METRICS = [LIR, SEP, PTD, LID, SED, PTC, PPT]


def entity2plain_text(dataset_name, model_name):
    entity2name = entity2plain_text(dataset_name, model_name)
    return entity2name


# (self_loop user 0) (watched movie 2408) (watched user 1953) (watched movie 277) #hop3
# (self_loop user 0) (mention word 2408) (described_as product 1953) (self_loop product 1953) #hop2
def get_linked_interaction_triple(path):
    linked_interaction_id, linked_interaction_rel, linked_interaction_type = path[1][-1], path[1][0], path[1][1]
    return linked_interaction_id, linked_interaction_rel


def get_shared_entity_tuple(path):
    path_type = path[-1][0]
    if path_type == 'self_loop':  # Handle size 3
        shared_entity_id, shared_entity_type = path[-2][-1], path[-2][1]
        return shared_entity_id, shared_entity_type
    shared_entity_id, shared_entity_type = path[-2][-1], path[-2][1]
    return shared_entity_id, shared_entity_type


def get_path_type(path):
    path_type = path[-1][0]
    if path_type == 'self_loop':  # Handle size 3
        path_type = path[-2][0]
    return path_type


def get_path_pattern(path):
    return [path_tuple[0] for path_tuple in path[1:]]


def get_path_types_in_kg(dataset_name):
    df_kg = pd.DataFrame(os.join(get_data_dir(dataset_name), "kg_final.txt"), sep="\t", header=True)
    return list(df_kg.kg.relations.unique())


def get_no_path_types_in_kg(dataset_name):
    df_kg = pd.DataFrame(os.join(get_data_dir(dataset_name), "kg_final.txt"), sep="\t", header=True)
    return len(df_kg.kg.relations.unique())


def load_LIR_matrix(self):
    lir_matrix_filepath = os.path.join(self.precomputed_folderpath, self.dataset_name, "LIR_matrix.pkl")
    lir_words_matrix_filepath = os.path.join(self.precomputed_folderpath, self.dataset_name, "LIR_matrix_words.pkl")
    if os.path.isfile(lir_matrix_filepath):
        print("Loading pre-computed LIR-matrix")
        with open(lir_matrix_filepath, 'rb') as f:
            self.LIR_matrix = pickle.load(f)
        f.close()

        if self.dataset_name != "ml1m" and self.dataset_name != "lastfm":
            with open(lir_words_matrix_filepath, 'rb') as f:
                self.LIR_matrix_words = pickle.load(f)
            f.close()
    else:
        print("Generating LIR-matrix")
        self.generate_LIR_matrix()
        with open(os.path.join(self.precomputed_folderpath, self.dataset_name, 'LIR_matrix.pkl'), 'wb') as f:
            pickle.dump(self.LIR_matrix, f)
        f.close()
        with open(os.path.join(self.precomputed_folderpath, self.dataset_name, 'LIR_matrix_words.pkl'), 'wb') as f:
            pickle.dump(self.LIR_matrix_words, f)
        f.close()


def load_SEP_matrix(self):
    sep_matrix_filepath = os.path.join(self.precomputed_folderpath, self.dataset_name, "SEP_matrix.pkl")
    if os.path.isfile(sep_matrix_filepath):
        print("Loading pre-computed SEP-matrix")
        with open(sep_matrix_filepath, 'rb') as f:
            self.SEP_matrix = pickle.load(f)
        f.close()
    else:
        print("Generating SEP-matrix")
        self.generate_SEP_matrix()
        with open(os.path.join(self.precomputed_folderpath, self.dataset_name, 'SEP_matrix.pkl'), 'wb') as f:
            pickle.dump(self.SEP_matrix, f)
        f.close()


def get_dataset_pid2model_kg_pid(dataset_name):
    data_dir = get_data_dir(dataset_name)
    file = open(os.path.join(data_dir, "mappings/product_mappings.txt", "r"))  # TODO CHECK if we want that filename
    csv_reader = csv.reader(file, delimiter='\t')
    dataset_pid2model_kg_pid = {}
    next(csv_reader, None)
    for row in csv_reader:
        dataset_pid2model_kg_pid[row[1]] = int(row[0])
    file.close()
    return dataset_pid2model_kg_pid


def get_dataset_uid2model_kg_uid(dataset_name):
    data_dir = get_data_dir(dataset_name)
    file = open(os.path.join(data_dir, "mappings/user_mappings.txt", "r"))
    reader = csv.reader(file, delimiter="\t")
    dataset_uid2model_kg_pid = {}
    next(reader, None)
    for row in reader:
        uid_review = row[1]
        uid_kg = int(row[0])
        dataset_uid2model_kg_pid[uid_review] = uid_kg
    return dataset_uid2model_kg_pid


def get_interaction2timestamp_map(dataset_name, model_name):
    data_dir = get_model_data_dir(dataset_name, model_name)
    # Load if already computated
    metadata_filepath = os.path.join(data_dir, "time_metadata.pkl")
    if os.path.isfile(metadata_filepath):
        with open(metadata_filepath, 'rb') as f:
            user2pid_time_tuple = pickle.load(f)
        f.close()
        return user2pid_time_tuple
    # Compute and save if not yet
    else:
        user2pid_time_tuple = defaultdict(list)
        dataset2kg = get_dataset_pid2model_kg_pid(dataset_name)
        file = open(os.path.join(data_dir, "train.txt"), 'r')
        csv_reader = csv.reader(file, delimiter=' ')
        uid_mapping = get_dataset_uid2model_kg_uid(dataset_name)
        for row in csv_reader:
            uid = uid_mapping[row[0]]
            pid = row[1]
            pid_model_kg = dataset2kg[pid]
            timestamp = int(row[3])
            user2pid_time_tuple[uid].append((PRODUCT, pid_model_kg, timestamp))
            if dataset_name in DATASETS_WITH_WORDS:
                words = row[3:]
                for word in words:
                    user2pid_time_tuple[uid].append((WORD, word, timestamp))
        with open(os.path.join(metadata_filepath), 'wb') as f:
            pickle.dump((user2pid_time_tuple), f)
        f.close()
        return user2pid_time_tuple


def generate_LIR_matrix(dataset_name, model_name):
    def normalized_ema(values):
        if max(values) == min(values):
            values = np.array([i for i in range(len(values))])
        else:
            values = np.array([i for i in values])
        values = pd.Series(values)
        ema_vals = values.ewm(span=len(values)).mean().tolist()
        min_res = min(ema_vals)
        max_res = max(ema_vals)
        return [(x - min_res) / (max_res - min_res) for x in ema_vals]

    uid_timestamp = get_interaction2timestamp_map(dataset_name, model_name)
    LIR_matrix = {uid: {} for uid in uid_timestamp.keys()}
    for uid in uid_timestamp.keys():
        linked_interaction_types = [PRODUCT]
        if dataset_name in DATASETS_WITH_WORDS:
            linked_interaction_types = [PRODUCT, WORD]
        for linked_entity_type in linked_interaction_types:
            interactions = [type_id_time for type_id_time in uid_timestamp[uid] if
                            type_id_time[0] == linked_entity_type]
            interactions.sort(key=lambda x: x[2])
            if len(uid_timestamp[uid]) <= 1:  # Skips users with only one review in train (can happen with lastfm)
                continue
            ema_timestamps = normalized_ema(interactions)
            pid_lir = {}
            for i in range(len(interactions)):
                pid = interactions[i][1]
                lir = ema_timestamps[i]
                pid_lir[pid] = lir
            LIR_matrix[uid][linked_entity_type] = pid_lir
    return LIR_matrix


def generate_SEP_matrix(dataset_name, model_name):
    def normalized_ema(values):
        if max(values) == min(values):
            values = np.array([i for i in range(len(values))])
        else:
            values = np.array([i for i in values])
        values = pd.Series(values)
        ema_vals = values.ewm(span=len(values)).mean().tolist()
        min_res = min(ema_vals)
        max_res = max(ema_vals)
        return [(x - min_res) / (max_res - min_res) for x in ema_vals]

    # Precompute entity distribution
    SEP_matrix = {}
    degrees = load_kg(dataset_name, model_name).degrees
    for type, eid_degree in degrees.items():
        eid_degree_tuples = list(zip(eid_degree.keys(), eid_degree.values()))
        eid_degree_tuples.sort(key=lambda x: x[1])
        ema_es = normalized_ema([x[1] for x in eid_degree_tuples])
        pid_weigth = {}
        for idx in range(len(ema_es)):
            pid = eid_degree_tuples[idx][0]
            pid_weigth[pid] = ema_es[idx]

        SEP_matrix[type] = pid_weigth
