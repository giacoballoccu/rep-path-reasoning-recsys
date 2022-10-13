from mappers.mapper_base import MapperBase
from utils import *

class MapperCAFE(MapperBase):
    ''''''
    def __init__(self, args):
        super().__init__(args)
        self.map_to_CAFE()
        self.get_splits()
        self.write_split_CAFE()
        self.mapping_folder = os.path.join(get_model_data_dir(args.model, args.data), "mappings")
        ensure_dir(self.mapping_folder)
        self.write_uid_pid_mappings()

    def write_split_CAFE(self):
        dataset_name = self.dataset_name
        #if dataset_name in AMAZON_DATASETS:
        #    map_to_PGPR_amazon(dataset_name)
        #    return
        output_folder = get_model_data_dir(self.model_name, dataset_name)
        for set in ["train", "valid", "test"]:
            with gzip.open(os.path.join(output_folder, f"{set}.txt.gz"), 'wt') as set_file:
                writer = csv.writer(set_file, delimiter="\t")
                set_values = getattr(self, set)
                for uid in set_values.keys():
                    pid_time_tuples = set_values[uid]
                    uid = self.ratings_uid2new_id[uid]
                    pids = [self.ratings_pid2new_id[pid] for pid, time in pid_time_tuples]
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
            new_rid2relation_name[n_relations + rid] = f"rev_{relation_name}"
            rid2reverse_rid[rid] = n_relations + rid

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
        # Extract old eid to read kg_final.txt
        e_map_df = pd.read_csv(os.path.join(input_folder, "e_map.txt"), sep="\t",
                               names=["old_eid", "name", "entity"]).iloc[1:, :]
        e_map_df = e_map_df[["old_eid", "entity"]]
        entity2old_eid = dict(zip(e_map_df.entity, e_map_df.old_eid))
        old_eid2entity = dict(zip([int(x) for x in e_map_df.old_eid], e_map_df.entity))

        # Collecting user entities
        all_entities_df = pd.DataFrame([], columns=["local_id", "name"])
        # Save users entities
        user_df = pd.read_csv(os.path.join(input_folder, "users.txt"), sep="\t")
        user_df.insert(0, "local_id", list(range(user_df.shape[0])))
        user_df = user_df[["local_id", "uid"]]
        self.ratings_uid2new_id = dict(zip([str(uid) for uid in list(user_df.uid)], [str(pid) for pid in list(user_df.local_id)])) #local_id
        user_df.local_id = user_df.local_id.apply(lambda x: f"user_{x}")
        user_df.uid = user_df.uid.apply(lambda x: f"user_{x}")
        user_df.rename({"uid": "name"}, axis=1, inplace=True)
        all_entities_df = all_entities_df.append(user_df)


        # Collecting product entities
        pid2kg_df = pd.read_csv(os.path.join(input_folder, "i2kg_map.txt"), sep="\t")
        pid2kg_df.insert(0, "local_id", [str(local_id) for local_id in range(pid2kg_df.shape[0])])
        products_df = pid2kg_df[["local_id", "entity"]]
        self.ratings_pid2new_id = dict(zip([str(pid) for pid in list(pid2kg_df.pid)], [str(pid) for pid in list(pid2kg_df.local_id)])) #local_id
        products_df.local_id = products_df.local_id.apply(lambda x: f"product_{x}")
        products_df["old_eid"] = products_df.entity.map(entity2old_eid)
        assert products_df.old_eid.isnull().sum() == 0
        products_df.rename({"entity": "name"}, axis=1, inplace=True)
        all_entities_df = all_entities_df.append(products_df)

        # eid_df = pd.read_csv(os.path.join(input_folder, "e_map.txt"), sep="\t")
        # eid2entity = dict(zip([str(eid) for eid in list(eid_df.eid)], eid_df.entity))

        # Collecting external entities
        kg_df = pd.read_csv(os.path.join(input_folder, "kg_final.txt"), sep="\t")
        assert kg_df.entity_tail.isnull().sum() == 0
        assert kg_df.entity_head.isnull().sum() == 0
        for rid, entity_name in rid2entity_name.items():
            unique_entities_by_type = list(kg_df[kg_df.relation == rid].entity_tail.unique())
            entity_by_type_df = pd.DataFrame(unique_entities_by_type, columns=["old_eid"])
            entity_by_type_df.insert(0, "local_id", [str(local_id) for local_id in range(entity_by_type_df.shape[0])])
            entity_by_type_df.local_id = entity_by_type_df.local_id.apply(lambda x: f"{entity_name}_{x}")
            entity_by_type_df["name"] = entity_by_type_df.old_eid.map(old_eid2entity)
            all_entities_df = all_entities_df.append(entity_by_type_df)

        # Write kg_entities.txt.gz
        all_entities_df.reset_index(inplace=True)
        all_entities_df["global_id"] = all_entities_df.index
        external_entities_df = all_entities_df.copy()
        external_entities_df = external_entities_df[~external_entities_df.local_id.str.contains("user")]
        external_entities_df.old_eid = external_entities_df.old_eid.astype("int64")
        all_entities_df = all_entities_df[["global_id", "local_id", "name"]]
        all_entities_df.to_csv(os.path.join(output_folder, "kg_entities.txt.gz"),
                               index=False,
                               sep="\t",
                               compression="gzip")

        # Extract ratings id (uid, pid) to global id
        user_entities_df = all_entities_df[all_entities_df.local_id.str.contains("user")]
        user_entities_df.name = user_entities_df.name.apply(lambda x: x.split("_")[-1])
        rating_uid2global_id = dict(zip(user_entities_df.name, user_entities_df.global_id))

        item2kg_df = pd.read_csv(os.path.join(input_folder, "i2kg_map.txt"), sep="\t")
        item2kg_df = item2kg_df[["entity", "pid"]]
        item2kg_df = pd.merge(item2kg_df, all_entities_df[all_entities_df.local_id.str.contains("product_")], left_on="entity", right_on="name")
        rating_pid2global_id = dict(zip([str(pid) for pid in list(item2kg_df.pid)], item2kg_df.global_id))

        """
        Collect triplets
        """
        triplets_df = pd.DataFrame([], columns=["entity_head", "relation", "entity_tail"])

        # Collect user interaction triplets
        self.get_splits()
        interaction_triplets = []
        main_interaction, rev_main_interaction = 0, rid2reverse_rid[0]
        for uid, pid_time_tuples in self.train.items():
            uid = rating_uid2global_id[uid]
            for pid, time in pid_time_tuples:
                pid = rating_pid2global_id[pid]
                interaction_triplets.append([uid, main_interaction, pid])
                interaction_triplets.append([pid, rev_main_interaction, uid])

        # Insert user interaction triplets to triplets_df
        triplets_df = triplets_df.append \
            (pd.DataFrame(interaction_triplets, columns=["entity_head", "relation", "entity_tail"], dtype="int64"))

        old_eid2global_id = {}
        # Products
        product_entities_df = external_entities_df[external_entities_df.local_id.str.contains("product")]
        old_eid2global_id[PRODUCT] = dict(zip(product_entities_df.old_eid, product_entities_df.global_id))

        # Other entities
        for rid, entity_name in rid2entity_name.items():
            unique_entities_by_type = list(kg_df[kg_df.relation == rid].entity_tail.unique())
            entity_by_type_df = pd.DataFrame(unique_entities_by_type, columns=["old_eid"])
            before_merge = entity_by_type_df.shape[0]
            entity_by_type_df = pd.merge(entity_by_type_df, external_entities_df, on="old_eid")
            if entity_by_type_df.old_eid.isnull().sum() > 0:
                entity_by_type_df.dropna(axis=0, inplace=True)
            after_merge = entity_by_type_df.shape[0]
            assert before_merge == after_merge
            old_eid2global_id[entity_name] = dict(zip(entity_by_type_df.old_eid, entity_by_type_df.global_id))

        # Insert other entities to kg_triplets df
        for rid, entity_name in rid2entity_name.items():
            triplets_by_type = kg_df[kg_df.relation == rid]
            # Map triplets
            triplets_by_type.entity_head = triplets_by_type.entity_head.map(old_eid2global_id[PRODUCT])
            print(triplets_by_type.entity_head.isnull().sum())
            triplets_by_type.entity_tail = triplets_by_type.entity_tail.map(old_eid2global_id[entity_name])
            print(triplets_by_type.entity_tail.isnull().sum())
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
