import numpy as np
from utils import *
from easydict import EasyDict as edict

"""
Implemented metrics
"""
NDCG = "ndcg"
MMR = "mmr"
SERENDIPITY = "serendipity"
COVERAGE = "coverage"
DIVERSITY = "diversity"
NOVELTY = "novelty"
CFAIRNESS = "cfairness"
PFAIRNESS = "pfairness"

REC_QUALITY_METRICS_TOPK = [NDCG, MMR, SERENDIPITY, DIVERSITY,
                            NOVELTY, PFAIRNESS]  # CHECK EVERYTIME A NEW ONE IS IMPLEMENTED IF IS LOCAL O GLOBAL
REC_QUALITY_METRICS_GLOBAL = [COVERAGE, CFAIRNESS]
"""
Methods
"""


def MMR(hit_list, k):
    r = np.asfarray(hit_list)[:k]
    hit_idxs = np.nonzero(r)
    if len(hit_idxs[0]) > 0:
        return 1 / (hit_idxs[0][0] + 1)
    return 0.


def dcg_at_k(hit_list, k, method=1):
    r = np.asfarray(hit_list)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(hit_list, k, method=0):
    dcg_max = dcg_at_k(sorted(hit_list, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(hit_list, k, method) / dcg_max

"""

def print_rec_metrics(dataset_name, flags, metrics):
    attribute_list = get_attribute_list(dataset_name, flags)

    print("\n---Recommandation Quality---")
    print("Average for the entire user base:", end=" ")
    for metric, values in metrics.items():
        print("{}: {:.3f}".format(metric, np.array(values["Overall"]).mean()), end=" | ")
    print("")

    for attribute_category, values in attribute_list.items():
        print("\n-Statistic with user grouped by {} attribute".format(attribute_category))
        for attribute in values[1].values():
            print("{} group".format(attribute), end=" ")
            for metric_name, groups_values in metrics.items():
                print("{}: {:.3f}".format(metric_name, np.array(groups_values[attribute]).mean()), end=" | ")
            print("")
    print("\n")
"""#TODO

def print_rec_quality_metrics(avg_metrics):
    print("\n***---Recommandation Quality---***")
    print("Average for the entire user base:", end=" ")
    for metric, group_value in avg_metrics.items():
        print(f"{metric}: {np.mean(group_value[OVERALL]):.3f}", end=" | ")
    print("")

    for metric, groups_value in avg_metrics.items():
        print(f"\n--- {metric}---")
        for group, value in groups_value.items():
            print(f"{group}: {np.mean(group_value[OVERALL]):.3f}", end=" | ")
        print("")
    print("\n")


def generate_latex_row(model_name, avg_metrics):
    row = [model_name]
    for metric in avg_metrics.keys():
        row.append(avg_metrics[metric][OVERALL])
    return ' & '.join(row)


"""
Beyond Accuracy
"""

"""
    Catalog coverage https://dl.acm.org/doi/pdf/10.1145/2926720
"""


def coverage(recommended_items_by_group, n_items_in_catalog):
    group_metric_value = {}
    for group, item_set in recommended_items_by_group.items():
        group_metric_value[group] = len(item_set) / n_items_in_catalog
    return group_metric_value


def serendipity_at_k():
    pass


def diversity_at_k(topk_items, pid2genre):
    diversity_items_tok = [pid2genre[pid] for pid in topk_items]
    return np.mean(diversity_items_tok)


def novelty_at_k(topk_items, pid2popularity):
    novelty_items_topk = [1 - pid2popularity[pid] for pid in topk_items]
    return np.mean(novelty_items_topk)

def exposure_pfairness(topk_items, pid2provider_popularity):
    exposure_providers_topk = [pid2provider_popularity[pid] for pid in topk_items]
    return np.mean(exposure_providers_topk)

def consumer_fairness(metrics_distrib, avg_metrics):
    fairness_metrics = {}

    # Compute consumer fairness
    fairness_metrics[CFAIRNESS] = {}
    for metric, group_values in avg_metrics.items():
        if len(group_values) == 2:
            group1, group2 = list(group_values.keys())
            #statistically_significant = statistical_test(metrics_distrib[metric][group1], metrics_distrib[metric][group2]) TODO
            fairness_metrics[CFAIRNESS][metric] = (group1, group2, avg_metrics[metric][group1] -
                                                   avg_metrics[metric][group2],) #statistically_significant) TODO
        if len(group_values) > 2:
            pairwise_diffs = []
            for group1 in group_values.keys():
                for group2 in group_values.keys():
                    if group1 != group2:
                        pairwise_diffs.append(group_values[group1] - group_values[group2])
                fairness_metrics[CFAIRNESS][metric] = (np.NAN, np.NAN, np.mean(pairwise_diffs), np.NAN)

# REC_QUALITY_METRICS = [NDCG, MMR, SERENDIPITY, COVERAGE, DIVERSITY, NOVELTY]
# FAIRNESS_METRICS = [CFAIRNESS, PFAIRNESS]
