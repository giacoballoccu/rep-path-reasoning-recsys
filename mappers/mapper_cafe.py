from mappers.mapper_base import MapperBase
from utils import *

class MapperCAFE(MapperBase):
    ''''''
    def __init__(self, args):
        super().__init__(args)
        ratings_uid2new_id, ratings_pid2new_id = self.map_to_CAFE()
        self.train, self.valid, self.test = self.time_based_train_test_split(ratings_uid2new_id, ratings_pid2new_id)
        self.write_split_CAFE()

    def write_split_CAFE(self):
        dataset_name = self.dataset_name
        #if dataset_name in AMAZON_DATASETS:
        #    map_to_PGPR_amazon(dataset_name)
        #    return
        output_folder = get_model_data_dir(self.model_name, dataset_name)
        for set in ["train", "valid", "test"]:
            with gzip.open(os.path.join(output_folder, f"{set}.txt.gz"), 'wt') as set_file:
                writer = csv.writer(set_file, delimiter="\t")
                if set == "train":
                    set_values = self.train
                if set == "valid":
                    set_values = self.valid
                if set == "test":
                    set_values = self.test
                for uid in range(len(set_values.keys())):
                    uid = str(uid)
                    pid_time_tuples = set_values[uid]
                    pids = [pid for pid, time in pid_time_tuples]
                    writer.writerow([uid] + pids)
            set_file.close()

    def map_to_CAFE(self):
        dataset_name = self.dataset_name
        #if dataset_name in AMAZON_DATASETS:
        #    map_to_PGPR_amazon(dataset_name)
        #    return
        input_folder = get_data_dir(dataset_name)
        output_folder = get_model_data_dir(self.model_name, dataset_name)

        ensure_dir(output_folder)

        relations_df = pd.read_csv(os.path.join(input_folder, "r_map.txt"), sep="\t")
        rid2entity_name = dict(zip(relations_df.id, relations_df.name.apply(lambda x: x.split("_")[-1])))
        rid2relation_name = dict(zip(relations_df.id, relations_df.name))

        """
        KG Relations
        """
        # Write new relations
        new_rid2relation_name = {0: MAIN_INTERACTION[dataset_name]}
        offset = len(new_rid2relation_name)
        old_rid2new_rid = {}
        for rid, relation_name in rid2relation_name.items():
            new_rid2relation_name[rid + offset] = relation_name
            old_rid2new_rid[rid] = rid + offset
        # Add reverse relations
        n_relations = len(new_rid2relation_name)
        rid2reverse_rid = {}
        for rid, relation_name in list(new_rid2relation_name.items()):
            new_rid2relation_name[n_relations +rid] = f"rev_{relation_name}"
            rid2reverse_rid[rid] = n_relations +rid

        # Save kg_relations file
        relations_df = pd.DataFrame(list(zip(new_rid2relation_name.keys(), new_rid2relation_name.values())))
        relations_df.to_csv(os.path.join(output_folder, "kg_relations.txt.gz"),
                            index=False, header=False, sep="\t", compression="gzip")

        """
        KG Rules
        """

        with gzip.open(output_folder + 'kg_rules.txt.gz', 'wt') as kg_rules_file:
            main_relation = 0
            # rules must be defined by hand
            writer = csv.writer(kg_rules_file, delimiter="\t")
            for rid in rid2reverse_rid.keys():
                forward, reverse = rid, rid2reverse_rid[rid]
                if rid == main_relation:
                    writer.writerow([main_relation, reverse, forward])
                else:
                    writer.writerow([main_relation, forward, reverse])
        kg_rules_file.close()

        """
        KG entities
        """
        # Collecting user entities
        all_entities_df = pd.DataFrame([], columns=["local_id", "name"]).astype(str)
        # Save users entities
        user_df = pd.read_csv(os.path.join(input_folder, "users.txt"), sep="\t")
        user_df.insert(0, "local_id", list(range(user_df.shape[0])))
        user_df = user_df[["local_id", "uid"]]
        uid2local_id = dict(zip([str(uid) for uid in list(user_df.uid)], [str(pid) for pid in list(user_df.local_id)]))
        user_df.local_id = user_df.local_id.apply(lambda x: f"user_{x}")
        user_df.uid = user_df.uid.apply(lambda x: f"user_{x}")
        user_df.rename({"uid": "name"}, axis=1, inplace=True)
        all_entities_df = all_entities_df.append(user_df)

        # Collecting product entities
        pid2kg_df = pd.read_csv(os.path.join(input_folder, "i2kg_map.txt"), sep="\t")
        pid2kg_df.insert(0, "local_id", [str(local_id) for local_id in range(pid2kg_df.shape[0])])
        products_df = pid2kg_df[["local_id", "entity"]]
        pid2local_id = dict(zip([str(pid) for pid in list(pid2kg_df.pid)], [str(pid) for pid in list(pid2kg_df.local_id)]))
        products_df.local_id = products_df.local_id.apply(lambda x: f"product_{x}")
        products_df.rename({"entity": "name"}, axis=1, inplace=True)
        all_entities_df = all_entities_df.append(products_df)

        # eid_df = pd.read_csv(os.path.join(input_folder, "e_map.txt"), sep="\t")
        # eid2entity = dict(zip([str(eid) for eid in list(eid_df.eid)], eid_df.entity))

        # Collecting external entities
        kg_df = pd.read_csv(os.path.join(input_folder, "kg_final.txt"), sep="\t")
        for rid, entity_name in rid2entity_name.items():
            unique_entities_by_type = list(kg_df[kg_df.relation == rid].entity_tail.unique())
            entity_by_type_df = pd.DataFrame(unique_entities_by_type, columns=["entity"])
            entity_by_type_df.insert(0, "local_id", [str(local_id) for local_id in range(entity_by_type_df.shape[0])])
            entity_by_type_df.local_id = entity_by_type_df.local_id.apply(lambda x: f"{entity_name}_{x}")
            entity_by_type_df.rename({"entity": "name"}, axis=1, inplace=True)
            all_entities_df = all_entities_df.append(entity_by_type_df)

        # Write kg_entities.txt.gz
        all_entities_df.reset_index(inplace=True)
        all_entities_df["global_id"] = all_entities_df.index
        all_entities_df = all_entities_df[["global_id", "local_id", "name"]]
        all_entities_df.to_csv(os.path.join(output_folder, "kg_entities.txt.gz"),
                               index=False,
                               sep="\t",
                               compression="gzip")

        # Extract ratings id (uid, pid) to global id
        user_entities_df = all_entities_df[all_entities_df.local_id.str.contains("user")]
        user_entities_df.name = user_entities_df.name.apply(lambda x: x.split("_")[-1])
        rating_uid2global_id = dict(zip(user_entities_df.name, user_entities_df.global_id))
        # Should be pid: global_id
        item2kg_df = pd.read_csv(os.path.join(input_folder, "i2kg_map.txt"), sep="\t")
        item2kg_df = item2kg_df[["entity", "pid"]]
        item2kg_df = pd.merge(item2kg_df, all_entities_df[all_entities_df.local_id.str.contains("product_")], left_on="entity", right_on="name")
        rating_pid2global_id = dict(zip([str(pid) for pid in list(item2kg_df.pid)], item2kg_df.global_id))

        """
        Collect triplets
        """
        triplets_df = pd.DataFrame([], columns=["entity_head", "relation", "entity_tail"])

        # Collect user interaction triplets
        train = self.time_based_train()
        interaction_triplets = []
        main_interaction, rev_main_interaction = 0, rid2reverse_rid[0]
        for uid, pid_time_tuples in train.items():
            uid = rating_uid2global_id[uid]
            for pid, time in pid_time_tuples:
                pid = rating_pid2global_id[pid]
                interaction_triplets.append([uid, main_interaction, pid])
                interaction_triplets.append([pid, rev_main_interaction, uid])

        # Insert user interaction triplets to triplets_df
        triplets_df = triplets_df.append \
            (pd.DataFrame(interaction_triplets, columns=["entity_head", "relation", "entity_tail"]))

        # Extract kg id (dbpedia, freebase) to global id
        kg_id2global_id = {}

        # Products
        product_entities_df = all_entities_df[all_entities_df.local_id.str.contains("product")]
        kg_id2global_id[PRODUCT] = dict(zip(product_entities_df.name, product_entities_df.global_id))

        # Other entities
        for rid, entity_name in rid2entity_name.items():
            unique_entities_by_type = list(kg_df[kg_df.relation == rid].entity_tail.unique())
            entity_by_type_df = pd.DataFrame(unique_entities_by_type, columns=["name"])
            entity_by_type_df = pd.merge(entity_by_type_df, all_entities_df, on="name")
            kg_id2global_id[entity_name] = dict(zip(entity_by_type_df.name, entity_by_type_df.global_id))

        # Insert other entities to kg_triplets df
        for rid, entity_name in rid2entity_name.items():
            triplets_by_type = kg_df[kg_df.relation == rid]
            # Map triplets
            triplets_by_type.entity_head = triplets_by_type.entity_head.map(kg_id2global_id[PRODUCT])
            triplets_by_type.entity_tail = triplets_by_type.entity_tail.map(kg_id2global_id[entity_name])
            triplets_by_type.relation = triplets_by_type.relation.map(old_rid2new_rid)
            # Create reverse triplet
            rev_triplets_by_type = triplets_by_type.copy()
            rev_triplets_by_type.rename({"entity_head": "entity_tail", "entity_tail": "entity_head"}, axis=1, inplace=True)
            rev_triplets_by_type = rev_triplets_by_type[["entity_head", "relation", "entity_tail"]]
            rev_triplets_by_type.relation = rev_triplets_by_type.relation.map(rid2reverse_rid)
            triplets_df = triplets_df.append(triplets_by_type)
            triplets_df = triplets_df.append(rev_triplets_by_type)

        triplets_df.to_csv(os.path.join(output_folder, "kg_triples.txt.gz"),
                           index=False,
                           sep="\t",
                           compression="gzip")
        return uid2local_id, pid2local_id