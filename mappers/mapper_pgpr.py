from collections import defaultdict

from mappers.mapper_base import MapperBase
from utils import *

class MapperPGPR(MapperBase):
    ''''''
    def __init__(self, args):
        super().__init__(args)
        ratings_uid2new_id, ratings_pid2new_id = self.map_to_PGPR()
        self.train, self.valid, self.test = self.time_based_train_test_split(ratings_uid2new_id, ratings_pid2new_id)
        self.write_split_PGPR()

    def map_to_PGPR(self):
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

        # Save users entities
        user_df = pd.read_csv(os.path.join(input_folder, "users.txt"), sep="\t")
        user_df.insert(0, "new_id", list(range(user_df.shape[0])))
        user_df = user_df[["new_id", "uid"]]
        user_df.to_csv(os.path.join(output_folder, "user.txt.gz"),
                       index=False,
                       sep="\t",
                       compression="gzip")

        uid2new_id = dict(zip([str(uid) for uid in list(user_df.uid)], user_df.new_id))

        # Save products entities
        eid2new_id = {}
        pid2kg_df = pd.read_csv(os.path.join(input_folder, "i2kg_map.txt"), sep="\t")
        pid2kg_df.insert(0, "new_id", [str(new_id) for new_id in range(pid2kg_df.shape[0])])
        products_df = pid2kg_df[["new_id", "pid"]]
        products_df.to_csv(os.path.join(output_folder, "product.txt.gz"),
                           index=False,
                           sep="\t",
                           compression="gzip")
        pid2new_id = dict(zip([str(pid) for pid in list(products_df.pid)], products_df.new_id))
        eid2new_id[PRODUCT] = dict(zip(pid2kg_df.entity, pid2kg_df.new_id))

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
                for new_pid in range(len(pid2new_id.keys())):
                    writer.writerow(triplets_grouped_by_pid[str(new_pid)])
            curr_rel_file.close()
        return uid2new_id, pid2new_id

    def write_split_PGPR(self):
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
                for uid, pid_time_tuples in set_values.items():
                    for pid, time in pid_time_tuples:
                        writer.writerow([uid, pid, 1, time])
            set_file.close()

    """Old
    def map_to_PGPR_amazon(self):
        input_folder = f'data/{dataset_name}/preprocessed/'
        input_folder_kg = f'data/{dataset_name}/preprocessed/'
        output_folder = f'data/{dataset_name}/preprocessed/pgpr/'

        ensure_dir(output_folder)

        relation_id2entity = {}
        relation_id2relation_name = {}
        with open(input_folder_kg + 'r_map.txt', 'r') as relation_file:
            reader = csv.reader(relation_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                relation_id, relation_url = int(row[0]), row[1]
                relation_id2entity[relation_id] = relation_to_entity[dataset_name][relation_url]
                if relation_id in [1, 2, 4, 5]:
                    relation_id2relation_name[relation_id] = relation_url + f'_p_{relation_id2entity[relation_id][:2]}'
                else:
                    relation_id2relation_name[relation_id] = relation_id2entity[
                                                                 relation_id] + f'_p_{relation_id2entity[relation_id][:2]}'
        relation_file.close()

        entity2dataset_id = {}
        with open(input_folder_kg + 'i2kg_map.txt', 'r') as item_to_kg_file:
            reader = csv.reader(item_to_kg_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                dataset_id, entity_id = row[0], row[-1]
                entity2dataset_id[entity_id] = dataset_id
        item_to_kg_file.close()

        dataset_id2new_id = {}
        with open(input_folder + "products.txt", 'r') as item_to_kg_file:
            reader = csv.reader(item_to_kg_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                new_id, dataset_id = row[0], row[1]
                dataset_id2new_id[dataset_id] = new_id
        item_to_kg_file.close()

        triplets_groupby_entity = defaultdict(set)
        relation_pid_to_entity = {relation_file_name: defaultdict(list) for relation_file_name in
                                  relation_id2relation_name.values()}
        with open(input_folder_kg + 'kg_final.txt', 'r') as kg_file:
            reader = csv.reader(kg_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                entity_head, entity_tail, relation = row[0], row[1], row[2]
                if relation == "1" or relation == "4":
                    triplets_groupby_entity['related_product'].add(entity_tail)
                    relation_pid_to_entity[relation_id2relation_name[int(relation)]][entity_head].append(entity_tail)
                elif relation == "1" or relation == "4":
                    triplets_groupby_entity['product'].add(entity_tail)
                    relation_pid_to_entity[relation_id2relation_name[int(relation)]][entity_head].append(entity_tail)
                else:
                    triplets_groupby_entity[relation_id2entity[int(relation)]].add(entity_tail)
                    relation_pid_to_entity[relation_id2relation_name[int(relation)]][entity_head].append(entity_tail)
        kg_file.close()

        entity_to_entity_url = {}
        with open(input_folder_kg + 'e_map.txt', 'r') as entities_file:
            reader = csv.reader(entities_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                entity_id, entity_url = row[0], row[1]
                entity_to_entity_url[entity_id] = entity_url
        entities_file.close()

        entity_id2new_id = defaultdict(dict)
        for entity_name, entity_list in triplets_groupby_entity.items():
            if entity_name == "product":
                for new_id, entity in enumerate(set(entity_list)):
                    entity_id2new_id[entity_name][entity] = new_id
                continue
            with gzip.open(output_folder + f'{entity_name}.txt.gz', 'wt') as entity_file:
                writer = csv.writer(entity_file, delimiter="\t")
                writer.writerow(['new_id', 'name'])
                for new_id, entity in enumerate(set(entity_list)):
                    writer.writerow([new_id, entity])
                    entity_id2new_id[entity_name][entity] = new_id
            entity_file.close()

        for relation_name, items_list in relation_pid_to_entity.items():
            entity_name = relation_name2entity_name[dataset_name][relation_name]
            # print(relation_name, entity_name)
            with gzip.open(output_folder + f'{relation_name}.txt.gz', 'wt') as relation_file:
                writer = csv.writer(relation_file, delimiter="\t")
                # writer.writerow(['new_id', 'name'])
                for i in range(len(dataset_id2new_id.keys()) + 1):
                    entity_list = items_list[str(i)]
                    entity_list_mapped = [entity_id2new_id[entity_name][entity_id] for entity_id in entity_list]
                    writer.writerow(entity_list_mapped)
            relation_file.close()

        with gzip.open(output_folder + 'product.txt.gz', 'wt') as product_fileo:
            writer = csv.writer(product_fileo, delimiter="\t")
            with open(input_folder + 'products.txt', 'r') as product_file:
                reader = csv.reader(product_file, delimiter="\t")
                for row in reader:
                    writer.writerow(row)
            product_file.close()
        product_fileo.close()

        with gzip.open(output_folder + 'user.txt.gz', 'wt') as users_fileo:
            writer = csv.writer(users_fileo, delimiter="\t")
            with open(input_folder + 'users.txt', 'r') as users_file:
                reader = csv.reader(users_file, delimiter="\t")
                for row in reader:
                    writer.writerow(row)
            users_file.close()
        users_fileo.close()
    """