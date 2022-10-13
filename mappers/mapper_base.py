from collections import defaultdict

from utils import *

class MapperBase(object):
    def __init__(self,args):
        self.args = args
        self.dataset_name = args.data
        self.model_name = args.model
        self.valid_size = args.valid_size
        self.train_size = args.train_size
        self.test_size = 1 - args.train_size - args.valid_size
        print(f"Creating data/{self.dataset_name}/preprocessed/{self.model_name}/ filesystem")

    """
    def time_based_train(self):
        train_size = self.train_size
        dataset_name, model_name = self.dataset_name, self.model_name
        input_folder = get_data_dir(dataset_name)
        output_folder = get_model_data_dir(model_name, dataset_name)

        ensure_dir(output_folder)

        uid2pids_timestamp_tuple = defaultdict(list)
        with open(os.path.join(input_folder, 'ratings.txt'), 'r') as ratings_file:  # uid	pid	rating	timestamp
            reader = csv.reader(ratings_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                uid, pid, rating, timestamp = row
                uid2pids_timestamp_tuple[uid].append([pid, int(timestamp)])
        ratings_file.close()

        for uid in uid2pids_timestamp_tuple.keys():
            uid2pids_timestamp_tuple[uid].sort(key=lambda x: x[1])

        train = {}
        for uid, pid_time_tuples in uid2pids_timestamp_tuple.items():
            n_interactions = len(pid_time_tuples)
            train_end = int(n_interactions * train_size)
            train[uid] = pid_time_tuples[:train_end]

        return train
    """
    def get_splits(self):
        input_dir = get_data_dir(self.dataset_name)
        self.train, self.valid, self.test = {}, {}, {}
        for set in ["train", "valid", "test"]:
            with open(os.path.join(input_dir, f"{set}.txt"), 'r') as set_file:
                curr_set = getattr(self, set)
                reader = csv.reader(set_file, delimiter="\t")
                for row in reader:
                    uid, pid, interaction, time = row
                    if uid not in curr_set:
                        curr_set[uid] = []
                    curr_set[uid].append((pid, time))
            set_file.close()

    """
    def time_based_train_test_split(self, ratings_uid2new_id, ratings_pid2new_id):
        train_size, valid_size, test_size = self.train_size, self.valid_size, self.test_size
        dataset_name, model_name = self.dataset_name, self.model_name
        input_folder = get_data_dir(dataset_name)
        output_folder = get_model_data_dir(model_name, dataset_name)

        ensure_dir(output_folder)

        uid2pids_timestamp_tuple = defaultdict(list)
        with open(os.path.join(input_folder, 'ratings.txt'), 'r') as ratings_file:  # uid	pid	rating	timestamp
            reader = csv.reader(ratings_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                uid, pid, rating, timestamp = row
                #uid, pid = ratings_uid2new_id[uid], ratings_pid2new_id[pid]
                uid = ratings_uid2new_id[uid]
                pid = ratings_pid2new_id[pid]
                uid2pids_timestamp_tuple[uid].append([pid, int(timestamp)])
        ratings_file.close()

        for uid in uid2pids_timestamp_tuple.keys():
            uid2pids_timestamp_tuple[uid].sort(key=lambda x: x[1])

        train, valid, test = {}, {}, {}
        for uid, pid_time_tuples in uid2pids_timestamp_tuple.items():
            n_interactions = len(pid_time_tuples)
            train_end = int(n_interactions * train_size)
            valid_end = train_end + int(n_interactions * valid_size)
            train[uid], valid[uid], test[uid] = pid_time_tuples[:train_end], pid_time_tuples[train_end:valid_end], pid_time_tuples[valid_end:]

        return train, valid, test
    """
    def write_uid_pid_mappings(self):
        ratings_uid2new_id_df = pd.DataFrame(list(zip(self.ratings_uid2new_id.keys(), self.ratings_uid2new_id.values())),
                                             columns=["rating_id", "new_id"])
        ratings_uid2new_id_df.to_csv(os.path.join(self.mapping_folder, "user_mapping.txt"), sep="\t", index=False)

        ratings_pid2new_id_df = pd.DataFrame(list(zip(self.ratings_pid2new_id.keys(), self.ratings_pid2new_id.values())),
                                             columns=["rating_id", "new_id"])
        ratings_pid2new_id_df.to_csv(os.path.join(self.mapping_folder, "product_mapping.txt"), sep="\t", index=False)