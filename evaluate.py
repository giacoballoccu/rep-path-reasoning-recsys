import argparse
from collections import defaultdict

from utils import *
import pickle
import os
from rec_quality_metrics import *

def evaluate(args, topk_items, test_labels, k=10, topk_paths=None):
     dataset_name = args.data
     results_dir = get_result_dir(args.data, args.model)

     uid2gender = get_uid_to_sensible_attribute(dataset_name, GENDER)
     uid2age = get_uid_to_sensible_attribute(dataset_name, AGE)

     groups = [*set(uid2gender.values()), *set(uid2age.values()), OVERALL]
     rec_quality_metrics = {metric: {group: [] for group in groups} for metric in REC_QUALITY_METRICS}
     avg_rec_quality_metrics = {metric: {group: defaultdict(int) for group in groups} for metric in REC_QUALITY_METRICS}

     #Storage for results if saving to csv is specified (used for plotitng)
     distributions_rows = []
     avgs_rows = []

     #Evaluate recommendation quality for users' topk
     for uid, topk in topk_items:
          gender = uid2gender[uid]
          age = uid2age[uid]

          hits = []
          for pid in topk:
               hits.append(1 if pid in test_labels else 0)

          #If the model has predicted less than 10 items pad with zeros
          while len(hits) < k:
               hits.append(0)

          for metric in REC_QUALITY_METRICS:
               if metric == NDCG:
                    metric_value = ndcg_at_k(hits, k)
               elif metric == MMR:
                    metric_value = MMR(hits, k)
               elif metric == SERENDIPITY:
                    pass
               elif metric == COVERAGE:
                    pass
               elif metric == DIVERSITY:
                    pass
               elif metric == NOVELTY:
                    pass
               rec_quality_metrics[metric][gender].append(metric_value)
               rec_quality_metrics[metric][age].append(metric_value)
               rec_quality_metrics[metric][OVERALL].append(metric_value)
               if args.save_distrib:
                    distributions_rows.append([dataset_name, args.model, gender, age, metric, metric_value])

     #Save as csv if specified
     if args.save_distribs:
          distributions_df = pd.DataFrame(distributions_rows, sep="\t", header=["dataset", "model", "gender", "age", "metric", "value"])
          distributions_df.to_csv(results_dir + "group_distributions.csv", sep="\t", index=False)

     #Compute average values for metrics
     for metric, group_values in rec_quality_metrics.items():
          group, values = group_values
          avg_value = np.mean(values)
          avg_rec_quality_metrics[metric][group] = avg_value
          if args.save_avgs:
               avgs_rows.append([dataset_name, args.model, group, metric, metric_value])

     #Print results
     print_rec_quality_metrics(avg_rec_quality_metrics)

     #Save as csv if specified
     if args.save_avgs:
          avgs_df = pd.DataFrame(avgs_rows, sep="\t",
                                          header=["dataset", "model", "group", "metric", "value"])
          avgs_df.to_csv(results_dir + "avg_values_quality.csv", sep="\t", index=False)

     """
     Fairness
     """
     fairness_metrics = {}
     # Compute consumer fairness
     avg_rec_quality_metrics[CFAIRNESS][]

     # Compute provider fairness

     """
     Evaluate path quality
     """
     if topk_paths != None:
          evalute_paths()



def main():
     parser = argparse.ArgumentParser()
     parser.add_argument('--model', type=str, default=PGPR, help='which model to evaluate')
     parser.add_argument('--data', type=str, default=ML1M, help='which dataset evaluate')
     parser.add_argument('--save_avgs', default=False, type=bool, help='whether to save the average value for every metric and group')
     parser.add_argument('--save_distribs', default=False, type=bool, help='whether to save the distributions for every metric and group')

     parser.add_argument('--show_case_study', default=False, type=bool, help='whether to visualize a random user topk')
     args = parser.parse_args()

     results_dir = get_result_dir(args.data, args.model)

     evaluate(args, None, None)
     #Check if args are correct
     if args.model not in KNOWLEDGE_AWARE_METHODS and args.model not in PATH_REASONING_METHODS:
          print("Check if methods is correctly selected among the available")

     with open(results_dir + "topk_items.pkl", 'rb') as topk_items_file:
          topk_items = pickle.load(topk_items_file)
     topk_items_file.close()

     if args.model in PATH_REASONING_METHODS:
          with open(results_dir + "topk_paths.pkl", 'rb') as topk_paths_file:
               topk_paths = pickle.load(topk_paths_file)
          topk_paths_file.close()
     else:
          topk_paths = None

     evaluate(args, topk_items, topk_paths)


if __name__ == '__main__':
    main()
