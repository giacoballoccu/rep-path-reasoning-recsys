import argparse
from collections import defaultdict

from tqdm import tqdm

from reasoning_path_utils import *
from utils import *
import pickle
import os
from rec_quality_metrics import *
from reasoning_path import *


def load_pred_paths(args):
    result_folder = get_result_dir(args.data, args.model)
    pred_paths_path = os.path.join(result_folder, 'pred_paths.pkl')
    with open(pred_paths_path, 'rb') as pred_paths_file:
        pred_paths = pickle.load(pred_paths_file)
    pred_paths_file.close()
    return pred_paths


def statistical_test(distrib1, distrib2):
    pass  # TODO


def topk_from_paths(args, train_labels, test_labels):
    k = args.k
    dataset_name = args.data
    embeds = load_embed(dataset_name, args.model)
    user_embeds = embeds[USER]
    main_relation = MAIN_INTERACTION[dataset_name]
    interaction_embeds = embeds[main_relation][0]
    item_embeds = embeds[PRODUCT]
    scores = np.dot(user_embeds + interaction_embeds, item_embeds.T)

    # {uid: {pid: [(path_score, path_prob, path), ..., ], ..., }, ..., }
    pred_paths = load_pred_paths(args)

    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = defaultdict(list)
    for uid in pred_paths:
        train_pids = set(train_labels[uid])
        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            # Get the path with highest probability
            #(-7.2741203, -5.727427, [('self_loop', 'user', 0), ('watched', 'product', 2215), ('rev_watched', 'user', 1711), ('watched', 'product', 447)])
            #
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0])

    # 3) Compute top 10 recommended products for each user.
    sort_by = 'score'
    pred_labels = {}
    pred_paths_topk = {}
    for uid in best_pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)
        topk_pids = [p[-1][2] for _, _, p in sorted_path[:k]]  # from largest to smallest
        topk_paths = [(path_tuple[-1][-1][2],) + path_tuple for path_tuple in sorted_path[:k]] # paths for the top10

        # add up to 10 pids if not enough, this product will not be explainable by a path
        if args.add_products and len(topk_pids) < k:
            train_pids = set(train_labels[uid])
            cand_pids = np.argsort(scores[uid])
            for cand_pid in cand_pids[::-1]:
                if cand_pid in train_pids or cand_pid in topk_pids:
                    continue
                topk_pids.append(cand_pid)
                topk_paths.append((cand_pid, scores[uid][cand_pid], None, None))  # Placeholder for no explanation
                if len(topk_pids) >= k:
                    break
        # end of add
        pred_labels[uid] = topk_pids[::-1]  # change order to from smallest to largest!
        pred_paths_topk[uid] = topk_paths[::-1]

    result_dir = get_result_dir(dataset_name, args.model)
    path_topk_filepath = os.path.join(result_dir, "path_topk.pkl")
    if os.path.exists(path_topk_filepath):
        with open(path_topk_filepath, "rb") as path_topk_file:
            print(f"Loading reasoning path objects in  {path_topk_filepath}")
            pred_paths_topk = pickle.load(path_topk_file)
        path_topk_file.close()
    else:
        pred_paths_topk = pathfy(args, pred_paths_topk)
        with open(path_topk_filepath, "wb") as path_topk_file:
            print(f"Saving reasoning path objects from {path_topk_filepath}")
            pickle.dump(pred_paths_topk, path_topk_file)
        path_topk_file.close()
    evaluate(args, pred_labels, test_labels, pred_paths_topk)


def evaluate_rec_quality(args, topk_items, test_labels):
    dataset_name = args.data
    k = args.k
    results_dir = get_result_dir(args.data, args.model)

    uid2gender = get_uid_to_sensible_attribute(dataset_name, args.model, GENDER)
    uid2age = get_uid_to_sensible_attribute(dataset_name, args.model, AGE)

    groups = [*set(uid2gender.values()), *set(uid2age.values()), OVERALL]
    rec_quality_metrics = {metric: {group: [] for group in groups} for metric in REC_QUALITY_METRICS_TOPK}
    avg_rec_quality_metrics = {metric: {group: defaultdict(int) for group in groups} for metric in
                               REC_QUALITY_METRICS_TOPK}

    # Coverage
    n_items_in_catalog = get_item_count(dataset_name)
    recommended_items_by_group = {group: set() for group in groups}

    #Novelty
    pid2popularity = get_item_pop(dataset_name, args.model)

    #Provider Exposure Fairness
    pid2provider_popularity = get_item_provider_pop(dataset_name, args.model)

    #Diversity
    pid2genre = get_item_genre(dataset_name, args.model)

    #Serendipity
    mostpop_topk = get_mostpop_topk(dataset_name, args.model, args.k)

    # Storage for results if saving to csv is specified (used for plotitng)
    distributions_rows = []
    avgs_rows = []
    pbar = tqdm(desc="Evaluating rec quality", total=len(topk_items.keys()))
    # Evaluate recommendation quality for users' topk
    for uid, topk in topk_items.items():
        gender = uid2gender[uid]
        age = uid2age[uid]

        hits = []
        for pid in topk:
            hits.append(1 if pid in test_labels[uid] else 0)

        # If the model has predicted less than 10 items pad with zeros
        while len(hits) < k:
            hits.append(0)

        for metric in REC_QUALITY_METRICS_TOPK:
            if metric == NDCG:
                metric_value = ndcg_at_k(hits, k)
            if metric == MMR:
                metric_value = mmr_at_k(hits, k)
            if metric == SERENDIPITY:
                metric_value = serendipity_at_k(topk, mostpop_topk[uid], k)
            if metric == DIVERSITY:
                metric_value = diversity_at_k(topk, pid2genre)
            if metric == NOVELTY:
                metric_value = novelty_at_k(topk, pid2popularity)
            if metric == PFAIRNESS:
                metric_value = exposure_pfairness(topk, pid2provider_popularity)
            rec_quality_metrics[metric][gender].append(metric_value)
            rec_quality_metrics[metric][age].append(metric_value)
            rec_quality_metrics[metric][OVERALL].append(metric_value)
            if args.save_distrib:
                distributions_rows.append([dataset_name, args.model, gender, age, metric, metric_value])
        # For coverage
        recommended_items_by_group[gender] |= set(topk)
        recommended_items_by_group[age] |= set(topk)
        recommended_items_by_group[OVERALL] |= set(topk)
        pbar.update(1)




    # Save as csv if specified
    if args.save_distrib:
        distributions_df = pd.DataFrame(distributions_rows,
                                        columns=["dataset", "model", "gender", "age", "metric", "value"])
        distributions_df.to_csv(results_dir + "rec_quality_group_distrib.csv", sep="\t", index=False)

    # Compute average values for metrics
    for metric, group_values in rec_quality_metrics.items():
        for group, values in group_values.items():
            avg_value = np.mean(values)
            avg_rec_quality_metrics[metric][group] = avg_value
            if args.save_avg:
                avgs_rows.append([dataset_name, args.model, group, metric, avg_value])

    # Compute global metrics
    c_fairness = {}
    for metric in REC_QUALITY_METRICS_GLOBAL:
        if metric == CFAIRNESS:
            c_fairness = consumer_fairness(rec_quality_metrics, avg_rec_quality_metrics)
        if metric == COVERAGE:
            avg_rec_quality_metrics[metric] = coverage(recommended_items_by_group, n_items_in_catalog)

    # Print results
    print_rec_quality_metrics(avg_rec_quality_metrics, c_fairness)

    # Save as csv if specified
    if args.save_avg:
        avgs_df = pd.DataFrame(avgs_rows,
                               columns=["dataset", "model", "group", "metric", "value"])
        avgs_df.to_csv(results_dir + "rec_quality_group_avg_values.csv", sep="\t", index=False)

    return rec_quality_metrics, avg_rec_quality_metrics


def evaluate_path_quality(args, topk_paths):
    dataset_name = args.data
    results_dir = get_result_dir(args.data, args.model)

    uid2gender = get_uid_to_sensible_attribute(dataset_name, args.model, GENDER)
    uid2age = get_uid_to_sensible_attribute(dataset_name, args.model, AGE)

    groups = [*set(uid2gender.values()), *set(uid2age.values()), OVERALL]
    path_quality_metrics = {metric: {group: [] for group in groups} for metric in PATH_QUALITY_METRICS}
    avg_path_quality_metrics = {metric: {group: defaultdict(int) for group in groups} for metric in
                                PATH_QUALITY_METRICS}

    # Storage for results if saving to csv is specified (used for plotitng)
    distributions_rows = []
    avgs_rows = []
    pbar = tqdm(desc="Evaluating path quality", total=len(topk_paths.keys()))
    # Evaluate recommendation quality for users' topk
    for uid, topk_reasoning_paths in topk_paths.items():
        gender = uid2gender[uid]
        age = uid2age[uid]
        for metric in PATH_QUALITY_METRICS:
            if metric == LIR:
                metric_value = topk_reasoning_paths.topk_lir()
            elif metric == SEP:
                metric_value = topk_reasoning_paths.topk_sep()
            elif metric == PTD:
                metric_value = topk_reasoning_paths.topk_ptd()
            elif metric == LID:
                metric_value = topk_reasoning_paths.topk_lid()
            elif metric == SED:
                metric_value = topk_reasoning_paths.topk_sed()
            elif metric == PTC:
                metric_value = topk_reasoning_paths.topk_ptc()
            elif metric == PPT:
                metric_value = topk_reasoning_paths.topk_ppt()
            elif metric == FIDELITY:
                metric_value = topk_reasoning_paths.topk_fidelity()
            path_quality_metrics[metric][gender].append(metric_value)
            path_quality_metrics[metric][age].append(metric_value)
            path_quality_metrics[metric][OVERALL].append(metric_value)
            if args.save_distrib:
                distributions_rows.append([dataset_name, args.model, gender, age, metric, metric_value])
        pbar.update(1)
    # Save as csv if specified
    if args.save_distrib:
        distributions_df = pd.DataFrame(distributions_rows,
                                        columns=["dataset", "model", "gender", "age", "metric", "value"])
        distributions_df.to_csv(results_dir + "path_quality_group_distrib.csv", sep="\t", index=False)

    # Compute average values for metrics
    for metric, group_values in path_quality_metrics.items():
        for group, values in group_values.items():
            avg_value = np.mean(values)
            avg_path_quality_metrics[metric][group] = avg_value
            if args.save_avg:
                avgs_rows.append([dataset_name, args.model, group, metric, avg_value])

    c_fairness = {}
    for metric in REC_QUALITY_METRICS_GLOBAL:
        if metric == CFAIRNESS:
            c_fairness = consumer_fairness(path_quality_metrics, avg_path_quality_metrics)

    # Print results
    print_path_quality_metrics(avg_path_quality_metrics, c_fairness)

    # Save as csv if specified
    if args.save_avg:
        avgs_df = pd.DataFrame(avgs_rows,
                               columns=["dataset", "model", "group", "metric", "value"])
        avgs_df.to_csv(results_dir + "path_quality_group_avg_values.csv", sep="\t", index=False)

    return path_quality_metrics, avg_path_quality_metrics

def evaluate(args, topk_items, test_labels, topk_paths=None):
    # NDCG, MMR, SERENDIPITY, COVERAGE, DIVERSITY, NOVELTY
    if args.evaluate_rec_quality:
        evaluate_rec_quality(args, topk_items, test_labels)
    """
     Evaluate path quality
     """
    if topk_paths != None:
        # LIR, SEP, PTD, LID, SED, PTC, PPTD, %EXP AMONG ITEMS
        if args.evaluate_path_quality:
            evaluate_path_quality(args, topk_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=PGPR, help='which model to evaluate')
    parser.add_argument('--data', type=str, default=LFM1M, help='which dataset evaluate')
    parser.add_argument('--create_topk', type=bool, default=True,
                        help='whether to create the topk from predicted paths or not')
    parser.add_argument('--k', type=int, default=10, help='size of the topk')
    parser.add_argument('--add_products', default=True, type=bool,
                        help='whether to fill the top-k (if there are less predicted path) with items that have no explanation path')
    parser.add_argument('--evaluate_rec_quality', default=True, type=bool,
                        help='whether to evaluate rec quality of predicted topk')
    parser.add_argument('--evaluate_path_quality', default=True, type=bool,
                        help='whether to evaluate associated reasoning path quality of predicted topk paths')
    parser.add_argument('--save_avg', default=True, type=bool,
                        help='whether to save the average value for every metric and group')
    parser.add_argument('--save_distrib', default=True, type=bool,
                        help='whether to save the distributions for every metric and group')
    parser.add_argument('--show_case_study', default=False, type=bool, help='whether to visualize a random user topk')
    args = parser.parse_args()

    results_dir = get_result_dir(args.data, args.model)

    train_labels = load_labels(args.data, args.model, 'train')  # TODO: STANDARDIZE KGAT E CO TO CREATE A TMP FOLDER WITH THIS STUFF
    valid_labels = load_labels(args.data, args.model, 'valid')
    test_labels = load_labels(args.data, args.model,  'test')

    if args.model in PATH_REASONING_METHODS:
        topk_from_paths(args, train_labels, test_labels)
    exit()
    evaluate(args, None, None)  # USED FOR TESTING

    # Check if args inputted are correct
    if args.model not in KNOWLEDGE_AWARE_METHODS and args.model not in PATH_REASONING_METHODS:
        raise Exception("Model selected not found among available ones")

    with open(results_dir + "topk_items.pkl", 'rb') as topk_items_file:
        topk_items = pickle.load(topk_items_file)
    topk_items_file.close()

    if args.model in PATH_REASONING_METHODS:
        with open(results_dir + "pred_paths.pkl", 'rb') as topk_paths_file:
            topk_paths = pickle.load(topk_paths_file)
        topk_paths_file.close()
    else:
        topk_paths = None

    evaluate(args, topk_items, topk_paths)


if __name__ == '__main__':
    main()
