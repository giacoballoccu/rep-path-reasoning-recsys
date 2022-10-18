import argparse
import shutil
from collections import Counter, defaultdict

import pandas as pd
from utils import *


def preprocess_ml1m(args):
    dataset_name = args.data
    preprocess_users(dataset_name)
    preprocess_kg_ml1m(args)

def preprocess_lfm1m(args):
    perform_k_core(args)
    if args.preprocess_kg:
        preprocess_kg_lfm1m(args)

def preprocess_kg_lfm1m(args):
    raw_kg_dir = get_raw_kg_dir(args.data)
    output_dir = get_data_dir(args.data)
    kg_files = os.listdir(os.path.join(raw_kg_dir))
    for kg_file in kg_files:
        src = os.path.join(raw_kg_dir, kg_file)
        dst = os.path.join(output_dir, kg_file)
        shutil.copyfile(src, dst)

    kg_triplets_df = pd.read_csv(os.path.join(output_dir, "kg_final.txt"), sep="\t")
    entity_df = pd.read_csv(os.path.join(output_dir, "e_map.txt"), sep="\t", names=["eid", "name", "entity"]).iloc[1: , :]
    entity2eid = dict(zip(entity_df.entity, entity_df.eid))
    kg_triplets_df.entity_head = kg_triplets_df.entity_head.map(entity2eid)
    kg_triplets_df.entity_tail = kg_triplets_df.entity_tail.map(entity2eid)
    kg_triplets_df.dropna(inplace=True)
    kg_triplets_df = kg_triplets_df.astype("int64")
    kg_triplets_df.to_csv(os.path.join(output_dir, "kg_final.txt"), sep="\t", index=False)

    remove_entites_with_different_relations(output_dir)

def remove_entites_with_different_relations(output_dir):
    kg_triplets_df = pd.read_csv(os.path.join(output_dir, "kg_final.txt"), sep="\t")
    n_triplets = kg_triplets_df.shape[0]
    kg_triplets_list = [list(a) for a in zip(kg_triplets_df.entity_head,
                                             kg_triplets_df.relation, kg_triplets_df.entity_tail)]
    kg_triplets_list.sort(key=lambda x: x[1])
    entity_tail_rel = {}
    valid_triplets = []
    for triplet in list(kg_triplets_list):
        entity_h, r, entity_t = triplet
        if entity_t not in entity_tail_rel:
            entity_tail_rel[entity_t] = r
        else:
            if entity_tail_rel[entity_t] != r:
                continue
        valid_triplets.append(triplet)
    kg_triplets_df = pd.DataFrame(valid_triplets, columns=["entity_head", "relation", "entity_tail"])

    #Propagate removal and reset eid
    i2kg_df = pd.read_csv(os.path.join(output_dir, "i2kg_map.txt"), sep="\t")
    valid_products = i2kg_df.eid.unique()
    valid_tails = kg_triplets_df.entity_tail.unique()
    entity_df = pd.read_csv(os.path.join(output_dir, "e_map.txt"), sep="\t")
    entity_df = entity_df[(entity_df.eid.isin(valid_tails)) | (entity_df.eid.isin(valid_products))]
    entity_df.rename({"eid": "old_eid"}, axis=1, inplace=True)
    entity_df.insert(0, "eid", list(range(entity_df.shape[0])))
    old_eid2new_eid = dict(zip(entity_df.old_eid, entity_df.eid))
    kg_triplets_df.entity_tail = kg_triplets_df.entity_tail.map(old_eid2new_eid)
    entity_df.drop("old_eid", axis=1, inplace=True)
    entity_df.to_csv(os.path.join(output_dir, "e_map.txt"), sep="\t", index=False)
    kg_triplets_df.to_csv(os.path.join(output_dir, "kg_final.txt"), sep="\t", index=False)
    print(f"Removed {n_triplets - kg_triplets_df.shape[0]} triplets")


def preprocess_kg_ml1m(args):
    dataset_name = args.data
    dataset_raw_folder = get_raw_kg_dir(dataset_name)
    output_dir = get_data_dir(dataset_name)
    if dataset_name == ML1M:
        relation2plain_name = { "http://dbpedia.org/ontology/cinematography": "cinematography_by_cinematographer",
                                "http://dbpedia.org/property/productionCompanies": "produced_by_prodcompany",
                                "http://dbpedia.org/property/composer": "composed_by_composer",
                                "http://purl.org/dc/terms/subject": "belong_to_category",
                                "http://dbpedia.org/ontology/starring": "starred_by_actor",
                                "http://dbpedia.org/ontology/country": "produced_in_country",
                                "http://dbpedia.org/ontology/wikiPageWikiLink": "related_to_wikipage",
                                "http://dbpedia.org/ontology/editing": "edited_by_editor",
                                "http://dbpedia.org/property/producers": "produced_by_producer",
                                "http://dbpedia.org/property/allWriting": "wrote_by_writter",
                                "http://dbpedia.org/ontology/director": "directed_by_director"
                                }
    #Add relation name and stardardize r_map.txt
    relations_df = pd.read_csv(os.path.join(dataset_raw_folder, "r_map.txt"), sep="\t")
    relations_df.insert(0, "id", list(range(relations_df.shape[0])))
    old_rid2new_rid = dict(zip(relations_df.relation_id, relations_df.id))
    relations_df.rename({"relation_url": "kb_relation"}, axis=1, inplace=True)
    relations_df["name"] = relations_df.kb_relation.map(relation2plain_name)
    relations_df = relations_df[["id", "kb_relation", "name"]]
    relations_df.to_csv(os.path.join(output_dir, "r_map.txt"), sep="\t", index=False)

    #Standardize e_map.txt
    products_df = pd.read_csv(os.path.join(output_dir, "products.txt"), sep="\t")
    i2kg_df = pd.read_csv(os.path.join(dataset_raw_folder, "i2kg_map.txt"), sep="\t")
    i2kg_df = i2kg_df[i2kg_df.dataset_id.isin(products_df.pid)]
    items_entity_ids = i2kg_df.entity_id.unique()
    e_map_df = pd.read_csv(os.path.join(dataset_raw_folder, "e_map.txt"), sep="\t")
    e_map_df["name"] = e_map_df.entity_url.apply(lambda x: x.split("/")[-1])
    entity_items = e_map_df[e_map_df.entity_id.isin(items_entity_ids)]
    other_entities = e_map_df[~e_map_df.entity_id.isin(items_entity_ids)]
    all_entities = entity_items.append(other_entities)
    all_entities.insert(0, "eid", list(range(all_entities.shape[0])))
    old_eid2new_eid = dict(zip(all_entities.entity_id, all_entities.eid))
    all_entities = all_entities[["eid", "name", "entity_url"]]
    all_entities.rename({"entity_url": "entity"}, axis=1, inplace=True)
    all_entities.to_csv(os.path.join(output_dir, "e_map.txt"), sep="\t", index=False)

    #Standardize i2kg.txt
    i2kg_df = i2kg_df[["dataset_id", "entity_url"]]
    i2kg_df.rename({"dataset_id": "pid", "entity_url": "entity"}, axis=1, inplace=True)
    entity_items = all_entities[all_entities.name.isin(entity_items.name)]
    entity_items = pd.merge(entity_items, i2kg_df, on="entity")
    entity_items.drop_duplicates(subset="name", inplace=True)
    entity_items = entity_items[["eid", "pid", "name", "entity"]]
    entity_items.to_csv(os.path.join(output_dir, "i2kg_map.txt"), sep="\t", index=False)

    #Standardize kg_final.txt
    kg_triplets_df = pd.read_csv(os.path.join(dataset_raw_folder, "kg_final.txt"), sep="\t")
    kg_triplets_df.entity_head = kg_triplets_df.entity_head.map(old_eid2new_eid)
    kg_triplets_df.entity_tail = kg_triplets_df.entity_tail.map(old_eid2new_eid)
    kg_triplets_df.relation = kg_triplets_df.relation.map(old_rid2new_rid)
    kg_triplets_df.to_csv(os.path.join(output_dir, "kg_final.txt"), sep="\t", index=False)

    remove_entites_with_different_relations(output_dir)

def perform_k_core(args):
    k_user, k_item = args.k_user, args.k_item
    dataset_name = args.data
    input_dir = get_raw_data_dir(dataset_name)
    output_dir = get_data_dir(dataset_name)
    raw_ratings_filename = os.path.join(input_dir, "ratings.txt")
    raw_products_filename = os.path.join(input_dir, "products.txt")
    raw_users_filename = os.path.join(input_dir, "users.txt")

    ratings_df = pd.read_csv(raw_ratings_filename, sep="\t")
    counts_col_user = ratings_df.groupby("uid")["uid"].transform(len)
    counts_col_songs = ratings_df.groupby("pid")["pid"].transform(len)
    mask_user = counts_col_user >= k_user
    mask_products = counts_col_songs >= k_item
    print(f"Number of ratings before: {ratings_df.shape[0]}")
    ratings_df = ratings_df[mask_user & mask_products]
    print(f"Number of ratings after: {ratings_df.shape[0]}")
    ratings_df.to_csv(os.path.join(output_dir, "ratings.txt"), sep="\t", index=False)

    #Propagate removals on products
    products_df = pd.read_csv(raw_products_filename, sep="\t")
    print(f"Number of products before: {products_df.shape[0]}")
    products_df = products_df[products_df.pid.isin(ratings_df.pid)]
    print(f"Number of products after: {products_df.shape[0]}")
    products_df.to_csv(os.path.join(output_dir, "products.txt"), sep="\t", index=False)

    #Propagate removals on users
    users_df = pd.read_csv(raw_users_filename, sep="\t")
    print(f"Number of users before: {users_df.shape[0]}")
    users_df = users_df[users_df.uid.isin(ratings_df.uid)]
    print(f"Number of users after: {users_df.shape[0]}")
    users_df.to_csv(os.path.join(output_dir, "users.txt"), sep="\t", index=False)

def time_based_train_test_split(dataset_name, train_size, valid_size):
    dataset_name = dataset_name
    input_folder = get_data_dir(dataset_name)
    output_folder = input_folder

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

    train, valid, test = {}, {}, {}
    for uid, pid_time_tuples in uid2pids_timestamp_tuple.items():
        n_interactions = len(pid_time_tuples)
        train_end = int(n_interactions * train_size)
        valid_end = train_end + int(n_interactions * valid_size)+1
        train[uid], valid[uid], test[uid] = pid_time_tuples[:train_end], pid_time_tuples[train_end:valid_end], pid_time_tuples[valid_end:]

    for set_filename in [(train, "train.txt"), (valid, "valid.txt"), (test, "test.txt")]:
        set_values, filename = set_filename
        with open(os.path.join(output_folder, filename), 'w') as set_file:
            writer = csv.writer(set_file, delimiter="\t")
            for uid, pid_time_tuples in set_values.items():
                for pid, time in pid_time_tuples:
                    writer.writerow([uid, pid, 1, time])
        set_file.close()

def preprocess_products(dataset_name):
    raw_data_dir = get_raw_data_dir(dataset_name)
    data_dir = get_data_dir(dataset_name)
    products_df = pd.read_csv(os.path.join(raw_data_dir, "products.txt"), sep="\t")
    if dataset_name == ML1M:
        #Add provider
        provider_df = pd.read_csv(os.path.join(raw_data_dir, "directions.dat"), sep="::")
        provider_df = provider_df.astype("object")
        provider_df = provider_df[provider_df.movieId.isin(products_df.pid)]
        provider_df.drop_duplicates(subset="movieId", inplace=True)
        products_df = pd.merge(products_df, provider_df, how="outer", left_on="pid", right_on="movieId")
        if products_df.dirId.isnull().values.any():
            products_df.dirId.fillna(-1, inplace=True)
        products_df.rename({"movie_name": "name", "dirId": "provider_id"}, axis=1, inplace=True)
    elif dataset_name == LFM1M:
        products_df.rename({"artist_id": "provider_id"}, axis=1, inplace=True)

    #Add item popularity
    interactions_df = pd.read_csv(os.path.join(data_dir, "train.txt"), sep="\t", names=["uid", "pid", "interaction", "timestamp"])
    product2interaction_number = Counter(interactions_df.pid)
    most_interacted = max(product2interaction_number.values())
    less_interacted = 0 if len(list(product2interaction_number.keys())) != products_df.pid.unique().shape[0] \
        else min(product2interaction_number.values())
    for pid in list(products_df.pid.unique()):
        occ = product2interaction_number[pid] if pid in product2interaction_number else 0
        product2interaction_number[pid] = (occ - less_interacted) / (most_interacted - less_interacted)

    products_df.insert(3, "pop_item", product2interaction_number.values(), allow_duplicates=True)

    #Add provider popularity
    item2provider = dict(zip(products_df.pid, products_df.provider_id))
    interaction_provider_df = interactions_df.copy()
    interaction_provider_df.provider_id = interaction_provider_df.pid.map(item2provider)
    provider2interaction_number = Counter(interaction_provider_df.provider_id)
    provider2interaction_number[-1] = 0
    most_interacted, less_interacted = max(provider2interaction_number.values()), min(provider2interaction_number.values())
    for pid in provider2interaction_number.keys():
        occ = provider2interaction_number[pid]
        provider2interaction_number[pid] = (occ - less_interacted) / (most_interacted - less_interacted)
    products_df["pop_provider"] = products_df.provider_id.map(provider2interaction_number)
    products_df = products_df[["pid", "name", "provider_id", "genre", "pop_item", "pop_provider"]]
    products_df.to_csv(os.path.join(data_dir, "products.txt"), sep="\t", index=False)

def preprocess_users(dataset_name):
    raw_data_dir = get_raw_data_dir(dataset_name)
    data_dir = get_data_dir(dataset_name)
    users_df = pd.read_csv(os.path.join(raw_data_dir, "users.txt"), sep="\t")
    users_df.age = users_df.age.apply(lambda x: categorigal_to_categorigal_age(x))
    users_df.to_csv(f"{data_dir}/users.txt", sep="\t", index=False)

def categorigal_to_categorigal_age(age_group_number):
    if age_group_number == 1:   return "Under 18"
    if age_group_number == 18:  return "18-24"
    if age_group_number == 25:  return "25-34"
    if age_group_number == 35:  return "35-44"
    if age_group_number == 45:  return "45-49"
    if age_group_number == 50:  return "50-55"
    if age_group_number == 56:  return "56+"


def test_dataset_integrity(dataset_name):
    i2kg_filepath = f"data/{dataset_name}/preprocessed/i2kg_map.txt"
    i2kg_df = pd.read_csv(i2kg_filepath, sep="\t", dtype="object")
    if i2kg_df.shape[0] != i2kg_df.pid.unique().shape[0]:
        print("i2kg contains duplicates")
        drop_i2kg_map_duplicates(i2kg_filepath, i2kg_df)

    products_filepath = f"data/{dataset_name}/preprocessed/products.txt"
    products_df = pd.read_csv(products_filepath, sep="\t")
    if i2kg_df.shape[0] != products_df.shape[0]:
        print("i2kg/product len missmatch")
        drop_products_not_in_i2kg(products_filepath, products_df, i2kg_df)

    interactions_filepath = f"data/{dataset_name}/preprocessed/ratings.txt"
    interactions_df = pd.read_csv(interactions_filepath, sep="\t", dtype="object")
    if interactions_df[~interactions_df.pid.isin(i2kg_df.pid)].shape[0] > 0:
        print("Ratings contain interactions that involve a removed product or user")
        remove_interactions_with_removed_product(interactions_filepath, interactions_df, i2kg_df)

    e_map_filepath = f"data/{dataset_name}/preprocessed/e_map.txt"
    entity_df = pd.read_csv(e_map_filepath, sep="\t", dtype="object", names=["eid", "name", "entity"])
    entity_item = entity_df[entity_df.entity.isin(i2kg_df.entity)]
    if entity_item.shape[0] != i2kg_df.shape[0]:
        print("missing items in e_map")
        #drop_triplets_with_not_existing_tails(e_map_filepath, )

    print(entity_df.shape[0], entity_df.entity.unique().shape[0])
    if entity_df.shape[0] != entity_df.entity.unique().shape[0]:
        print("e_map contains duplicates")
        entity_df.drop_duplicates(subset="entity", inplace=True)
        if entity_df.entity.isnull().sum() != 0:
            entity_df.dropna(axis=0, inplace=True)
        entity_df.to_csv(e_map_filepath, sep="\t", index=False)
    other_entities = entity_df[~entity_df.entity.isin(i2kg_df.entity)]
    assert other_entities.shape[0] + entity_item.shape[0] == entity_df.shape[0]

    kg_final_filename = f"data/{dataset_name}/preprocessed/kg_final.txt"
    triplets_df = pd.read_csv(kg_final_filename, sep="\t", dtype="object",)
    print(other_entities.entity.unique().shape[0]-1, triplets_df.entity_tail.unique().shape[0])
    triplets_with_removed_head = triplets_df[~triplets_df.entity_head.isin(i2kg_df.eid)]
    if triplets_with_removed_head.shape[0] > 0:
        triplets_df = triplets_df[triplets_df.entity_head.isin(i2kg_df.eid)]
        triplets_df.to_csv(kg_final_filename, sep="\t", index=False)

    cleaned_triplets = triplets_df[~triplets_df.entity_tail.isin(entity_item.eid)]
    print(triplets_df.shape[0], cleaned_triplets.shape[0])
    if triplets_df.shape[0] != cleaned_triplets.shape[0]:
        print("Triplets contain corrupted triplets with a item as tail")
        cleaned_triplets.to_csv(kg_final_filename, sep="\t", index=False)

    cleaned_triplets = triplets_df[triplets_df.entity_tail.isin(other_entities.eid)]
    if triplets_df.shape[0] != cleaned_triplets.shape[0]:
        drop_triplets_with_corrupted_tail(kg_final_filename, cleaned_triplets)
        print(f"Removed {triplets_df.shape[0] - cleaned_triplets.shape[0]} corrupted triplets")

def drop_triplets_with_corrupted_tail(kg_final_filename, cleaned_triplets):
    cleaned_triplets.to_csv(kg_final_filename, sep="\t", index=False)

def drop_i2kg_map_duplicates(i2kg_filepath, i2kg_df):
    i2kg_df.drop_duplicates(subset="pid", inplace=True)
    i2kg_df.to_csv(i2kg_filepath, sep="\t", index=False)

def drop_products_not_in_i2kg(products_filepath, products_df, i2kg_df):
    products_df = products_df[products_df.pid.isin(i2kg_df.pid)]
    products_df.to_csv(products_filepath, sep="\t", index=False)

def remove_interactions_with_removed_product(interactions_filepath, interactions_df, i2kg_df):
    interactions_df = interactions_df[~interactions_df.pid.isin(i2kg_df.pid)]
    interactions_df.to_csv(interactions_filepath, sep="\t", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=LFM1M, help='One of {ML1M, LFM1M}')
    parser.add_argument('--k_user', type=int, default=10, help='threshold for discarding users with less than k interactions')
    parser.add_argument('--k_item', type=int, default=5, help='threshold for discarding items with less than k occurences')
    parser.add_argument('--preprocess_kg', type=bool, default=False, help='whether to preprocess kg or not')
    parser.add_argument('--train_size', type=float, default=0.6, help='size of the train set expressed in 0.x')
    parser.add_argument('--valid_size', type=float, default=0.2, help='size of the valid set expressed in 0.x')
    args = parser.parse_args()

    if args.data == ML1M:
        preprocess_ml1m(args)
        preprocess_kg_ml1m(args)
    if args.data == LFM1M:
        preprocess_kg_lfm1m(args)
    time_based_train_test_split(args.data, args.train_size, args.valid_size)
    preprocess_products(args.data)
    test_dataset_integrity(args.data)

if __name__ == '__main__':
    main()
