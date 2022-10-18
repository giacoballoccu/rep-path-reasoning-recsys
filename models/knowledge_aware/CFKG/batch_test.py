'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import models.knowledge_aware.metrics as metrics
from parser import parse_args
import multiprocessing
import heapq
import numpy as np
import random
from itertools import cycle
from torch.utils.data import DataLoader, RandomSampler
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from loader_cfkg import CFKG_loader



train_cores = multiprocessing.cpu_count()
test_cores = multiprocessing.cpu_count()//2

args = parse_args()
Ks = eval(args.Ks)


data_generator = {}


MANUAL_SEED = 2019

torch.manual_seed(MANUAL_SEED)

def seed_worker(worker_id):
    torch.manual_seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)
    random.seed(MANUAL_SEED)

g = torch.Generator(device='cpu')
g.manual_seed(MANUAL_SEED)


ds = CFKG_loader(args=args, path=args.data_path + args.dataset)
data_generator['dataset'] = ds
data_generator['loader'] = DataLoader(ds,
                batch_size=ds.batch_size,
                sampler=RandomSampler(ds,
                    replacement=True,
                    generator=g) if args.with_replacement else None,
                shuffle=False if args.with_replacement else True,
                num_workers=train_cores,
                drop_last=True,
                persistent_workers=True
                    )
batch_test_flag = True


USR_NUM, ITEM_NUM = data_generator['dataset'].n_users, data_generator['dataset'].n_items
N_TRAIN, N_TEST = data_generator['dataset'].n_train, data_generator['dataset'].n_test
BATCH_SIZE = args.batch_size



def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks, save_topk=True):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    pids = []
    r = []
    for i in K_max_item_score:
        pids.append(i)
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc,pids

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks, save_topk=True):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    pids = []
    r = []
    for i in K_max_item_score:
        pids.append(i)
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc, pids


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator['dataset'].train_user_dict[u]
        valid_items = data_generator['dataset'].valid_user_dict[u]
    except Exception:
        training_items = []
        valid_items = []
    #user u's items in the test set
    user_pos_test = data_generator['dataset'].test_user_dict[u]

    all_items = set(range(ITEM_NUM))

    test_items = list((all_items - set(training_items)) - set(valid_items) )

    if args.test_flag == 'part':
        r, auc, pids = ranklist_by_heapq(user_pos_test, test_items, rating, Ks, save_topk=True)
    else:
        r, auc, pids = ranklist_by_sorted(user_pos_test, test_items, rating, Ks, save_topk=True)

    # # .......checking.......
    # try:
    #     assert len(user_pos_test) != 0
    # except Exception:
    #     print(u)
    #     print(training_items)
    #     print(user_pos_test)
    #     exit()
    # # .......checking.......
    result_dict = get_performance(user_pos_test, r, auc, Ks)
    result_dict['uid'] = u
    result_dict['pids'] = pids
    return result_dict


def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(test_cores)

    if args.model_type in ['ripple']:

        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
    elif args.model_type in ['fm', 'nfm']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE
    else:
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    from collections import defaultdict
    user_topk_dict = defaultdict(list)


    DATASET_KEY = 'A_dataset' if args.model_type == 'cke' else 'dataset'
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                feed_dict = data_generator[DATASET_KEY].as_test_feed_dict(model=model,
                                                                   user_batch=user_batch,
                                                                   item_batch=item_batch,
                                                                   drop_flag=drop_flag)
                i_rate_batch = model.eval(sess, feed_dict=feed_dict)
                i_rate_batch = i_rate_batch.reshape((-1, len(item_batch)))

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)
            feed_dict = data_generator[DATASET_KEY].as_test_feed_dict(model=model,
                                                               user_batch=user_batch,
                                                               item_batch=item_batch,
                                                               drop_flag=drop_flag)
            rate_batch = model.eval(sess, feed_dict=feed_dict)
            rate_batch = rate_batch.reshape((-1, len(item_batch)))

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
            u = re['uid']
            pids = re['pids']
            user_topk_dict[u] = pids



    assert count == n_test_users
    pool.close()
    return result,user_topk_dict