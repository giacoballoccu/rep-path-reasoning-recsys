from collections import defaultdict

from mappers.mapper_base import MapperBase
from utils import *

class MapperUCPR(MapperBase):
    ''''''
    def __init__(self, args):
        super().__init__(args)
        self.map_to_UCPR()
        self.get_splits()
        self.write_split_UCPR()
        self.mapping_folder = os.path.join(get_model_data_dir(args.model, args.data), "mappings")
        ensure_dir(self.mapping_folder)
        self.write_uid_pid_mappings()

    def map_to_UCPR(self):
        dataset_name = self.dataset_name
        #if dataset_name in AMAZON_DATASETS:
        #    map_to_UCPR_amazon(dataset_name)
        #    return
        input_folder = get_data_dir(dataset_name)
        output_folder = get_model_data_dir(self.model_name, dataset_name)

        ensure_dir(output_folder)

        relations_df = pd.read_csv(os.path.join(input_folder, "r_map.txt"), sep="\t")
        rid2entity_name = dict(zip(relations_df.id, relations_df.name.apply(lambda x: x.split("_")[-1])))
        rid2relation_name = dict(zip(relations_df.id, relations_df.name))

        # Save users entities
        user_df = pd.read_csv(os.path.join(input_folder, "users.txt"), sep="\t")
        user_df.insert(0, "new_id", list(range(user_df.shape[0])))
        user_df = user_df[["new_id", "uid"]]
        user_df.to_csv(os.path.join(output_folder, "user.txt.gz"),
                       index=False,
                       sep="\t",
                       compression="gzip")

        self.ratings_uid2new_id = dict(zip([str(uid) for uid in list(user_df.uid)], user_df.new_id))

        # Save products entities
        eid2new_id = {}
        pid2kg_df = pd.read_csv(os.path.join(input_folder, "i2kg_map.txt"), sep="\t")
        pid2kg_df.insert(0, "new_id", [str(new_id) for new_id in range(pid2kg_df.shape[0])])
        products_df = pid2kg_df[["new_id", "pid"]]
        products_df.to_csv(os.path.join(output_folder, "product.txt.gz"),
                           index=False,
                           sep="\t",
                           compression="gzip")
        self.ratings_pid2new_id = dict(zip([str(pid) for pid in list(products_df.pid)], products_df.new_id))
        eid2new_id[PRODUCT] = dict(zip(pid2kg_df.eid, pid2kg_df.new_id))

        # Get external entities by relation and save them
        kg_df = pd.read_csv(os.path.join(input_folder, "kg_final.txt"), sep="\t")
        for rid, entity_name in rid2entity_name.items():
            unique_entities_by_type = list(kg_df[kg_df.relation == rid].entity_tail.unique())
            entity_by_type_df = pd.DataFrame(unique_entities_by_type, columns=["eid"])
            entity_by_type_df.insert(0, "new_id", [str(new_id) for new_id in range(entity_by_type_df.shape[0])])
            entity_by_type_df.to_csv(os.path.join(output_folder, f"{entity_name}.txt.gz"),
                                     index=False,
                                     sep="\t",
                                     compression="gzip")
            eid2new_id[entity_name] = dict(zip(entity_by_type_df.eid, entity_by_type_df.new_id))

        # Group relations
        for rid, relation_name in rid2relation_name.items():
            entity_name = rid2entity_name[rid]
            triplets_grouped_by_rel = kg_df[kg_df.relation == rid]
            triplets_grouped_by_rel.entity_head = triplets_grouped_by_rel.entity_head.map(eid2new_id[PRODUCT])
            triplets_grouped_by_rel.entity_tail = triplets_grouped_by_rel.entity_tail.map(eid2new_id[entity_name])
            triplets_grouped_by_pid = defaultdict(list)
            for tuple in list(zip(triplets_grouped_by_rel.entity_head, triplets_grouped_by_rel.entity_tail)):
                pid, entity_tail = tuple
                triplets_grouped_by_pid[pid].append(entity_tail)
            # Save relations
            with gzip.open(os.path.join(output_folder, f"{relation_name}.txt.gz"), 'wt') as curr_rel_file:
                writer = csv.writer(curr_rel_file, delimiter="\t")
                for new_pid in range(len(self.ratings_pid2new_id.keys())):
                    writer.writerow(triplets_grouped_by_pid[str(new_pid)])
            curr_rel_file.close()


    def write_split_UCPR(self):
        dataset_name = self.dataset_name
        #if dataset_name in AMAZON_DATASETS:
        #    map_to_UCPR_amazon(dataset_name)
        #    return
        output_folder = get_model_data_dir(self.model_name, dataset_name)
        for set in ["train", "valid", "test"]:
            with gzip.open(os.path.join(output_folder, f"{set}.txt.gz"), 'wt') as set_file:
                writer = csv.writer(set_file, delimiter="\t")
                set_values = getattr(self, set)
                for uid, pid_time_tuples in set_values.items():
                    uid = self.ratings_uid2new_id[uid]
                    for pid, time in pid_time_tuples:
                        pid = self.ratings_pid2new_id[pid]
                        writer.writerow([uid, pid, 1, time])
            set_file.close()


    def write_uid_pid_mappings(self):
        ratings_uid2new_id_df = pd.DataFrame(list(zip(self.ratings_uid2new_id.keys(), self.ratings_uid2new_id.values())),
                                             columns=["rating_id", "new_id"])
        ratings_uid2new_id_df.to_csv(os.path.join(self.mapping_folder, "user_mapping.txt"), sep="\t", index=False)

        ratings_pid2new_id_df = pd.DataFrame(list(zip(self.ratings_pid2new_id.keys(), self.ratings_pid2new_id.values())),
                                             columns=["rating_id", "new_id"])
        ratings_pid2new_id_df.to_csv(os.path.join(self.mapping_folder, "product_mapping.txt"), sep="\t", index=False)