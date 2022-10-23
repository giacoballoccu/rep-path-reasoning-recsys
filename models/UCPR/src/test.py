from __future__ import absolute_import, division, print_function

import os
import argparse
import json
from math import log
from datetime import datetime
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import threading
from functools import reduce
import time
import pickle
import gc
import json
from easydict import EasyDict as edict
import itertools
from models.UCPR.utils import *
from models.UCPR.src.model.get_model.get_model import *
from models.UCPR.src.parser import parse_args
from models.UCPR.src.para_setting import parameter_path, parameter_path_th
import collections
def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def save_output(dataset_name, pred_paths):

    extracted_path_dir = LOG_DATASET_DIR[dataset_name]
    if not os.path.isdir(extracted_path_dir):
        os.makedirs(extracted_path_dir)
    print("Normalizing items scores...")
    # Get min and max score to performe normalization between 0 and 1
    score_list = []
    for uid, pid in pred_paths.items():
        for pid, path_list in pred_paths[uid].items():
            for path in path_list:
                score_list.append(float(path[0]))
    min_score = min(score_list)
    max_score = max(score_list)

    print("Saving pred_paths...")
    for uid in pred_paths.keys():
        curr_pred_paths = pred_paths[uid]
        for pid in curr_pred_paths.keys():
            curr_pred_paths_for_pid = curr_pred_paths[pid]
            for i, curr_path in enumerate(curr_pred_paths_for_pid):
                path_score = pred_paths[uid][pid][i][0]
                path_prob = pred_paths[uid][pid][i][1]
                path = pred_paths[uid][pid][i][2]
                new_path_score = (float(path_score) - min_score) / (max_score - min_score)
                pred_paths[uid][pid][i] = (new_path_score, path_prob, path)
    with open(extracted_path_dir + "/pred_paths.pkl", 'wb') as pred_paths_file:
        pickle.dump(pred_paths, pred_paths_file)
    pred_paths_file.close()

def evaluate(topk_matches, test_user_products, no_skip_user, dataset_name):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    metrics = edict(
        # ndcg_other=[],
        ndcg=[],
        hr=[],
        precision=[],
        recall=[],
    )
    ndcgs = []

    test_user_idxs = list(test_user_products.keys())
    x = defaultdict(int)
    rel_size = []
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 10:
            x['a'] +=1
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]

        if uid not in no_skip_user:
            x['b'] += 1
            continue
        if len(pred_list) == 0:
            x['c'] +=1
            continue
        rel_size.append(len(rel_set))
        k = 0
        hit_num = 0.0
        hit_list = []
        for pid in pred_list:
            k += 1
            if pid in rel_set:
                hit_num += 1
                hit_list.append(1)
            else:
                hit_list.append(0)
        #print(k, len(hit_list), collections.Counter(hit_list))
        ndcg = ndcg_at_k(hit_list, k)
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        metrics.ndcg.append(ndcg)
        metrics.hr.append(hit)
        metrics.recall.append(recall)
        metrics.precision.append(precision)
    avg_metrics = edict(
        ndcg=[],
        hr=[],
        precision=[],
        recall=[],
    )
    print("Average test set size: ", np.array(rel_size).mean())
    for metric, values in metrics.items():
        avg_metrics[metric] = np.mean(values)
        avg_metric_value = np.mean(values) * 100 if metric == "ndcg_other" else np.mean(values)
        n_users = len(values)
        print("Overall for noOfUser={}, {}={:.4f}".format(n_users, metric,
                                                          avg_metric_value))
        print("\n")
    makedirs(dataset_name)
    with open(RECOM_METRICS_FILE_PATH[dataset_name], 'w') as f:
        json.dump(metrics,f)

    return avg_metrics.precision, avg_metrics.recall, avg_metrics.ndcg, avg_metrics.hr,\
             invalid_users


def batch_beam_search(args, env, model, uids, device, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)
    state_pool = env.reset(args.epochs,uids)  # numpy of [bs, dim]
    
    model.reset(uids)

    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    index_ori_list = [_ for _ in range(len(uids))]
    idx_list = [i for i in range(len(uids))]
    # print('idx_list = ', idx_list)
    model.eval()
    for hop in range(3):

        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        
        state_tensor = model.generate_st_emb(path_pool, up_date_hop = idx_list) 

        batch_next_action_emb = model.generate_act_emb(path_pool, acts_pool)
        
        actmask_tensor = torch.BoolTensor(actmask_pool).to(device)
      
        try:
            next_enti_emb, next_action_emb = batch_next_action_emb[0], batch_next_action_emb[1]
            probs, _ = model((state_tensor[0],state_tensor[1], next_enti_emb, next_action_emb, actmask_tensor))
        except:
            probs, _ = model((state_tensor, batch_next_action_emb, actmask_tensor))  # Tensor of [bs, act_dim]
        
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        del actmask_tensor
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool, new_index_pool, new_idx = [], [], [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            index_ori = index_ori_list[row]

            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    if args.envir == 'p1':
                        next_node_type = KG_RELATION[args.dataset][path[-1][1]][relation]#KG_RELATION[path[-1][1]][relation]
                    else:
                        next_node_type = KG_RELATION[args.dataset][path[-1][1]][relation]#env.et_idx2ty[next_node_id]
                    # next_node_type = KG_RELATION[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
                new_index_pool.append(index_ori)
                new_idx.append(row)

        print(len(new_path_pool))
        print(len(new_idx))
        print()
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        index_ori_list = new_index_pool
        idx_list = new_idx

    gc.collect()

    return path_pool, probs_pool


def predict_paths(args, policy_file, path_file, train_labels, test_labels, pretest):
    print('Predicting paths...')
        
    env = KG_Env(args, Dataset(args), args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    print(policy_file)
    print('Loading pretrain')
    pretrain_sd = torch.load(policy_file)
    print('Loading model')
    model = Memory_Model(args, env.user_triplet_set, env.rela_2_index, 
                            env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    print('Model loaded')
    test_uids = list(test_labels.keys())
    test_uids = [uid for uid in test_uids if uid in train_labels and uid in env.user_list]

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []

    times = 0
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        # print(' bar state/text_uid = ', start_idx, '/', len(test_uids), end = '\r')
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        print(f'{start_idx}/{ len(test_uids)}')
        paths, probs = batch_beam_search(args, env, model, batch_uids, args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        times += 1

        if times % 50 == 0:
            str_batch_uids = [str(st) for st in batch_uids]
            fail_uids = ",".join(str_batch_uids)
            fail_batch = f"'batch_uids = ', {fail_uids}, {str(start_idx)}, {str(end_idx)}"
            args.logger.info(fail_batch)

        if pretest == 1 and  times >= 100: break

        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))




def get_validation_pids(dataset_name):
    if not os.path.isfile(os.path.join(DATASET_DIR[dataset_name], 'valid.txt')):
        return []
    validation_pids = defaultdict(set)
    with open(os.path.join(DATASET_DIR[dataset_name], 'valid.txt')) as valid_file:
        reader = csv.reader(valid_file, delimiter=" ")
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            validation_pids[uid].add(pid)
    valid_file.close()
    return validation_pids


def extract_paths(dataset_name, path_file, train_labels, valid_labels, test_labels):
    embeds = load_embed(dataset_name)
    
    main_product, main_interaction = MAIN_PRODUCT_INTERACTION[dataset_name]
    user_embeds = embeds[USER]
    purchase_embeds = embeds[main_interaction][0]
    product_embeds = embeds[main_product]#[0]
    print(user_embeds.shape)
    print(purchase_embeds.shape)
    print(product_embeds[0].shape)
    print(product_embeds.shape)

    scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)
    print(scores.shape)
    validation_pids = get_validation_pids(dataset_name)
    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {uid: {} for uid in test_labels}
    total_pre_user_num = {}

    no_skip_user = {}
    for path, probs in zip(results['paths'], results['probs']):
        uid = path[0][2]
        no_skip_user[uid] = 1
    print(results.keys())
    x = defaultdict(int)
    for idx, (path, probs) in enumerate(zip(results['paths'], results['probs'])):

        if path[-1][1] != main_product:
            #print('a')
            x['a'] += 1
            continue
        uid = path[0][2]
        if uid not in total_pre_user_num:
            total_pre_user_num[uid] = len(total_pre_user_num)
            x['b'] += 1
            #print('b')
        if uid not in pred_paths:
            #print('c')
            x['c'] += 1
            continue
        pid = path[-1][2]
        if uid in valid_labels and pid in valid_labels[uid]:
            #print('d')
            x['d'] += 1
            continue
        if pid in train_labels[uid]:
            #print('e')
            x['e'] += 1
            continue        
        if pid not in pred_paths[uid]:
            #print('f')
            x['f'] += 1
            pred_paths[uid][pid] = []

        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))
    print(x)
    #print(pred_paths)
    save_output(dataset_name, pred_paths)
    return pred_paths, scores


def evaluate_paths(topk,dataset_name, pred_paths, scores, train_labels, 
            test_labels, args, path_file, pretest=1):
   # train_labels, test_labels, args, path_file,  pretest=pretest):
    
    '''    
    embeds = load_embed(args.dataset)
    main_product, main_interaction = MAIN_PRODUCT_INTERACTION[args.dataset]
    user_embeds = embeds[USER]
    purchase_embeds = embeds[main_interaction][0]
    product_embeds = embeds[main_product]#[0]

    scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)
    validation_pids = get_validation_pids(args.dataset)
    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {uid: {} for uid in test_labels }#if uid in test_labels}#train_labels}

    total_pre_user_num = {}

    no_skip_user = {}

    for path, probs in zip(results['paths'], results['probs']):
        uid = path[0][2]
        no_skip_user[uid] = 1

    for path, probs in zip(results['paths'], results['probs']):
        if path[-1][1] != main_product:
            continue
        uid = path[0][2]
        if uid not in total_pre_user_num:
            total_pre_user_num[uid] = len(total_pre_user_num)
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        if uid in valid_labels and pid in valid_labels[uid]:
            continue
        if pid in train_labels[uid]:
            continue        
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []

        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))
    save_output(args.dataset, pred_paths)
    '''

    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}

    from collections import defaultdict
    #best_pred_paths_logging = {}
    for uid in pred_paths:
        train_pids = set(train_labels[uid])
        best_pred_paths[uid] = []
        #best_pred_paths_logging[uid] = []#defaultdict(list)
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0])
            #best_pred_paths_logging[uid].append(sorted_path[0])

    

    def prob_keyget(x):
        return (x[1], x[0])
    def score_keyget(x):
        return (x[0], x[1])
    sort_by = 'score'
    pred_labels = {}
    pred_paths_top10 = {}
    total_pro_num = 0
    for uid in best_pred_paths:
        if sort_by == 'score':
            keygetter = score_keyget
        elif sort_by == 'prob':
            keygetter = prob_keyget

        sorted_path = sorted(best_pred_paths[uid], key=keygetter, reverse=True)

        top10_pids = [p[-1][2] for _, _, p in sorted_path[:10]] 
        top10_paths = [p for _, _, p in sorted_path[:10]]

        if args.add_products and len(top10_pids) < 10:
            train_pids = set(train_labels[uid])
            cand_pids = np.argsort(scores[uid])
            for cand_pid in cand_pids[::-1]:
                if cand_pid in train_pids or cand_pid in top10_pids:
                    continue
                top10_pids.append(cand_pid)
                if len(top10_pids) >= 10:
                    break

        pred_labels[uid] = top10_pids[::-1]  # change order to from smallest to largest!
        pred_paths_top10[uid] = top10_paths[::-1]
        #print(uid, len(pred_labels[uid]), pred_labels[uid])
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {uid: {} for uid in test_labels}
    total_pre_user_num = {}

    no_skip_user = {}
    for path, probs in zip(results['paths'], results['probs']):
        uid = path[0][2]
        no_skip_user[uid] = 1

    avg_precision, avg_recall, avg_ndcg, avg_hit, invalid_users = evaluate(pred_labels, 
                test_labels, no_skip_user, dataset_name)
    print('precision: ', avg_precision)
    print('recall: ',  avg_recall) 
    print('ndcg: ', avg_ndcg)
    print('hit: ', avg_hit)


# In formula w of pi log(2 + (number of patterns of same pattern type among uv paths / total number of paths among uv paths))
def get_path_pattern_weigth(path_pattern_name, pred_uv_paths):
    n_same_path_pattern = 0
    total_paths = len(pred_uv_paths)
    for path in pred_uv_paths:
        if path_pattern_name == get_path_pattern(path):
            n_same_path_pattern += 1
    return log(2 + (n_same_path_pattern / total_paths))


def test(args, train_labels, valid_labels, test_labels, best_recall, pretest = 1):

    print('start predict')


    policy_file = args.policy_path  #args.save_model_dir + '/policy_model_epoch_{}.ckpt'.format(35)#args.eva_epochs)
    path_file = os.path.join(TMP_DIR[args.dataset], 'policy_paths_epoch{}_{}.pkl'.format(args.eva_epochs, args.topk_string)) #args.save_model_dir + '/' + 'pre' + str(pretest) + 'policy_paths_epoch{}_{}.pkl'.format(args.eva_epochs, args.topk_string)
    

    #if args.dataset in [BEAUTY_CORE, CELL_CORE, CLOTH_CORE]: 
    #sort_by_2 = 'prob'

    #eva_file_2 = args.log_dir + '/' + 'pre' + str(pretest) + sort_by_2 + '_eva'+ '_' + args.topk_string + '.txt'

    TOP_N_LOGGING = 100    
    
    if args.run_path or os.path.exists(path_file) == False:
        predict_paths(args, policy_file, path_file, train_labels, test_labels, pretest)#predict_paths(policy_file, path_file, args)
    if args.save_paths or args.run_eval():
        pred_paths, scores = extract_paths(args.dataset, path_file, train_labels, valid_labels, test_labels)
    if args.run_eval:
        evaluate_paths(TOP_N_LOGGING,args.dataset, pred_paths, scores,
                        train_labels, test_labels, args, path_file, pretest=pretest)



if __name__ == '__main__':
    args = parse_args()

    args.training = 0
    args.training = (args.training == 1)

    args.att_evaluation = False

    if args.envir == 'p1': 
        para_env = parameter_path_th
        KG_Env = KGEnvironment

    elif args.envir == 'p2':
        para_env = parameter_path
        KG_Env = KGEnvironment

    para_env(args)

    train_labels = load_labels(args.dataset, 'train')
    valid_labels = load_labels(args.dataset, 'valid')
    test_labels = load_labels(args.dataset, 'test')


    best_recall = 0


    args.eva_epochs = args.best_model_epoch
    test(args, train_labels, valid_labels, test_labels, best_recall, pretest = 0)


