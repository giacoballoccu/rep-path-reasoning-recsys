import argparse

import pandas as pd

def test_dataset_integrity(dataset_name):
    i2kg_filepath = f"{dataset_name}/preprocessed/i2kg_map.txt"
    i2kg_df = pd.read_csv(i2kg_filepath, sep="\t", dtype="object")
    if i2kg_df.shape[0] != i2kg_df.pid.unique().shape[0]:
        print("i2kg contains duplicates")
        drop_i2kg_map_duplicates(i2kg_filepath, i2kg_df)

    products_filepath = f"{dataset_name}/preprocessed/products.txt"
    products_df = pd.read_csv(products_filepath, sep="\t")
    if i2kg_df.shape[0] != products_df.shape[0]:
        print("i2kg/product len missmatch")
        drop_products_not_in_i2kg(products_filepath, products_df, i2kg_df)

    interactions_filepath = f"{dataset_name}/preprocessed/ratings.txt"
    interactions_df = pd.read_csv(interactions_filepath, sep="\t", dtype="object")
    if interactions_df[~interactions_df.pid.isin(i2kg_df.pid)].shape[0] > 0:
        print("Ratings contain interactions that involve a removed product or user")
        remove_interactions_with_removed_product(interactions_filepath, interactions_df, i2kg_df)

    e_map_filepath = f"{dataset_name}/preprocessed/e_map.txt"
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

    kg_final_filename = f"{dataset_name}/preprocessed/kg_final.txt"
    triplets_df = pd.read_csv(kg_final_filename, sep="\t", dtype="object",)

    triplets_with_removed_head = triplets_df[~triplets_df.entity_head.isin(i2kg_df.eid)]
    print(triplets_with_removed_head.entity_head.unique().shape[0])
    print(triplets_with_removed_head.shape[0], triplets_df.shape[0])
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
    parser.add_argument('--data', type=str, default="lfm1m", help='One of {ML1M, LFM1M}')
    args = parser.parse_args()

    test_dataset_integrity(args.data)


if __name__ == '__main__':
    main()