import argparse

import pandas as pd

def test_dataset_integrity(dataset_name):
    i2kg_df = pd.read_csv(f"{dataset_name}/preprocessed/i2kg_map.txt", sep="\t")
    if i2kg_df.shape[0] != i2kg_df.pid.unique().shape[0]:
        print("i2kg contains duplicates")

    products_df = pd.read_csv(f"{dataset_name}/preprocessed/products.txt", sep="\t")
    if i2kg_df.shape[0] != products_df.shape[0]:
        print("i2kg/product len missmatch")

    interactions_df = pd.read_csv(f"{dataset_name}/preprocessed/ratings.txt", sep="\t")
    if interactions_df[~interactions_df.pid.isin(i2kg_df.pid)].shape[0] > 0:
        print("Ratings contain interactions that involve a removed product or user")

    entity_df = pd.read_csv(f"{dataset_name}/preprocessed/e_map.txt", sep="\t", names=["eid", "name", "entity"]).iloc[1:, :]
    entity_item = entity_df[entity_df.entity.isin(i2kg_df.entity)]
    if entity_item.shape[0] != i2kg_df.shape[0]:
        print("missing items in e_map")

    triplets_df = pd.read_csv(f"{dataset_name}/preprocessed/kg_final.txt", sep="\t")
    triplets_df_check = triplets_df[triplets_df.entity_tail.isin(entity_item)]
    if triplets_df_check.shape[0] > 0:
        print("Triplets contain corrupted triplets with a item as tail")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="ml1m", help='One of {ML1M, LFM1M}')
    args = parser.parse_args()

    test_dataset_integrity(args.data)


if __name__ == '__main__':
    main()