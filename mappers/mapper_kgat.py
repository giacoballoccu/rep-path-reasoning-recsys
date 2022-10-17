import pandas as pd

from mappers.mapper_base import MapperBase
from utils import *
class MapperKGAT(MapperBase):
    ''''''
    def __init__(self, args):
        super().__init__(args)
        self.sep = " "
        self.map_to_KGAT()
        self.get_splits()
        self.write_split_KGAT()
        self.mapping_folder = os.path.join(get_model_data_dir(args.model, args.data), "mappings")
        ensure_dir(self.mapping_folder)
        self.write_uid_pid_mappings()

    def map_to_KGAT(self):
        dataset_name = self.dataset_name
        input_folder = get_data_dir(dataset_name)
        output_folder = get_model_data_dir(self.model_name, dataset_name)

        ensure_dir(output_folder)
        #item_list.txt
        i2kg_df = pd.read_csv(os.path.join(input_folder, "i2kg_map.txt"), sep="\t")
        i2kg_df = i2kg_df[["entity", "pid"]]
        i2kg_df.insert(1, "remap_id", list(range(i2kg_df.shape[0])))
        i2kg_df.rename({"pid": "org_id"}, axis=1, inplace=True)
        i2kg_df = i2kg_df[["org_id", "remap_id", "entity"]]

        i2kg_df.to_csv(os.path.join(output_folder, "item_list.txt"), sep=self.sep, index=False)
        self.ratings_pid2new_id = dict(zip(i2kg_df.org_id, i2kg_df.remap_id))

        #entity_list.txt
        entity_df = pd.read_csv(os.path.join(input_folder, "e_map.txt"), sep="\t", names=["eid", "name", "entity"])
        n_entities = entity_df.shape[0]
        non_product_entities = entity_df[~entity_df.entity.isin(i2kg_df.entity)]
        assert non_product_entities.shape[0] + i2kg_df.shape[0] == n_entities
        entity_df = entity_df[["entity", "eid"]]
        entity_df.rename({"entity": "org_id", "eid": "remap_id"}, axis=1, inplace=True)
        entity_df.to_csv(os.path.join(output_folder, "entity_list.txt"), sep=self.sep, index=False)

        #user_list.txt
        users_df = pd.read_csv(os.path.join(input_folder, "users.txt"), sep="\t")
        users_df.insert(1, "remap_id", list(range(users_df.shape[0])))
        users_df.rename({"uid": "org_id"}, axis=1, inplace=True)
        users_df.to_csv(os.path.join(output_folder, "user_list.txt"), sep=self.sep, index=False)
        self.ratings_uid2new_id = dict(zip(users_df.org_id, users_df.remap_id))

        #relation_list.txt
        #id	kb_relation	name
        r_map_df = pd.read_csv(os.path.join(input_folder, "r_map.txt"), sep="\t")
        r_map_df.rename({"id": "remap_id", "kb_relation": "org_id"}, axis=1, inplace=True)
        r_map_df = r_map_df[["org_id", "remap_id"]]
        r_map_df.to_csv(os.path.join(output_folder, "relation_list.txt"), sep=self.sep, index=False)

        #kg_final.txt
        kg_final_df = pd.read_csv(os.path.join(input_folder, "kg_final.txt"), sep="\t")
        kg_final_df = kg_final_df[["entity_head", "relation", "entity_tail"]]
        kg_final_df.to_csv(os.path.join(output_folder, "kg_final.txt"), sep=self.sep, header=False, index=False)

    def write_split_KGAT(self):
        dataset_name = self.dataset_name
        output_folder = get_model_data_dir(self.model_name, dataset_name)
        for set in ["train", "valid", "test"]:
            with open(os.path.join(output_folder, f"{set}.txt"), 'w') as set_file:
                writer = csv.writer(set_file, delimiter=self.sep)
                set_values = getattr(self, set)
                for uid in set_values.keys():
                    pid_time_tuples = set_values[uid]
                    uid = self.ratings_uid2new_id[int(uid)]
                    pids = [self.ratings_pid2new_id[int(pid)] for pid, time in pid_time_tuples]
                    writer.writerow([uid] + pids)
            set_file.close()
