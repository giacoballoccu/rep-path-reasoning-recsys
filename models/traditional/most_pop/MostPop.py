

"""Most Popular. Item are recommended based on their popularity (not personalized).
Parameters
----------
name: string, default: 'most_pop'
    The name of the recommender model.
"""
import argparse
from collections import defaultdict
from utils import *
from tqdm import tqdm

def load_labels(dataset_name, split=TRAIN):
    if split != TRAIN and split != VALID and split != TEST:
        raise Exception('mode should be one of {train, valid, test}.')
    data_dir = get_data_dir(dataset_name)
    label_path = os.path.join(data_dir, f"{split}.txt")
    user_products = defaultdict(list)
    with open("../../../" + label_path, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            uid, pid, interaction, time = row
            user_products[uid].append(pid)
    return user_products

def score(args, train_labels, valid_labels, test_labels):
    #Calculate pop for items
    item_idxs = {}
    for uid, pids in train_labels.items():
        for pid in pids:
            if pid not in item_idxs:
                item_idxs[pid] = 0
            item_idxs[pid] += 1
    for uid, pids in valid_labels.items():
        for pid in pids:
            if pid not in item_idxs:
                item_idxs[pid] = 0
            item_idxs[pid] += 1

    #Sort item by popularity
    most_pop_item_sorted = list(zip(item_idxs.keys(), item_idxs.values()))
    most_pop_item_sorted.sort(key=lambda x: x[1], reverse=True)
    
    #Produce topks
    pbar = tqdm(total=len(train_labels.keys()))
    user_topks = {}
    for uid in test_labels.keys():
        user_topks[uid] = []
        i = 0
        while len(user_topks[uid]) < args.k or i < len(most_pop_item_sorted):
            pid, pop = most_pop_item_sorted[i]
            if pid in train_labels[uid]:
                i += 1
                continue
            user_topks[uid].append(pid)
            i+=1
        pbar.update(1)

    #Save topks
    result_dir = "../../../" + get_result_dir(args.data, args.name)
    ensure_dir(result_dir)
    with open(os.path.join(result_dir, "item_topks.pkl"), 'wb') as f:
        pickle.dump(user_topks, f)
    f.close()
    return user_topks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ML1M, help='One of {ML1M}')
    parser.add_argument('--name', type=str, default='most_pop', help='directory name.')
    parser.add_argument('--k', type=int, default=100, help='')
    parser.add_argument('--evaluate', type=bool, default=True, help='')
    args = parser.parse_args()
    
    train_labels = load_labels(args.data, 'train')
    valid_labels = load_labels(args.data, 'valid')
    test_labels = load_labels(args.data, 'test')

    result_dir = "../../../" + get_result_dir(args.data, args.name)

    most_pop_preds_filename = os.path.join(result_dir, "item_topks.pkl")
    if os.path.isfile(most_pop_preds_filename):
        print("Loading pre-computed MostPop Recommendations to calculate Serendipity")
        with open(most_pop_preds_filename, 'rb') as f:
            user_topks = pickle.load(f)
        f.close()
    else:
        print("Executing MostPop to calculate Serendipity")
        user_topks = score(args, train_labels, valid_labels, test_labels)
    return user_topks

if __name__ == '__main__':
    main()