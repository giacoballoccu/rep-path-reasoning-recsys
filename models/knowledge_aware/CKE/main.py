'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.knowledge_aware.helper import *
from batch_test import *
from time import time
from CKE import CKE
import wandb
import os
import sys
from utils import *
from models.utils import MetricsLogger
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pickle

if __name__ == '__main__':

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # get argument settings.

    tf.set_random_seed(2019)
    np.random.seed(2019)
    args = parse_args()

    os.makedirs(TMP_DIR[args.dataset], exist_ok=True)
    makedirs(args.dataset)
    with open(os.path.join(TMP_DIR[args.dataset],f'{MODEL}_hparams.json'), 'w') as f:
        import json
        import copy
        args_dict = dict()
        for x,y in copy.deepcopy(args._get_kwargs()):
            args_dict[x] = y
        if 'device' in args_dict:
            del args_dict['device']
        json.dump(args_dict,f)



    metrics = MetricsLogger(args.wandb_entity if args.wandb else None, 
                            f'{MODEL}_{args.dataset}',
                            config=args)
    metrics.register('train_loss')
    metrics.register('train_base_loss')
    metrics.register('train_reg_loss')
    metrics.register('train_kge_loss')
    metrics.register('ndcg')
    metrics.register('hit')
    metrics.register('recall')     
    metrics.register('precision')
    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator['dataset'].n_users
    config['n_items'] = data_generator['dataset'].n_items
    config['n_relations'] = data_generator['dataset'].n_relations
    config['n_entities'] = data_generator['dataset'].n_entities

    if args.model_type in ['kgat', 'cfkg']:

        key = 'A_dataset' if args.model_type == 'kgat' else 'dataset'
        "Load the laplacian matrix."
        config['A_in'] = sum(data_generator[key].lap_list)

        "Load the KG triplets."
        config['all_h_list'] = data_generator[key].all_h_list
        config['all_r_list'] = data_generator[key].all_r_list
        config['all_t_list'] = data_generator[key].all_t_list
        config['all_v_list'] = data_generator[key].all_v_list

        config['n_relations'] = data_generator[key].n_relations

    t0 = time()



    model = CKE(data_config=config, pretrain_data=None, args=args)

    saver = tf.train.Saver()

    """
    *********************************************************
    Save the model parameters.
    """
    ensureDir(TMP_DIR[args.dataset])
    weights_save_path =  os.path.join(TMP_DIR[args.dataset], "weights")
    save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    t0 = time()
    train_time = 0
    test_time = 0

    #print(data_generator['dataset'].N_exist_users, ' ', data_generator['dataset'].n_users, ' ', data_generator['dataset'].n_train)
    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator['dataset'].n_train // args.batch_size + 1

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 1: to train the recommender.
        """
        loader_iter = iter(data_generator['loader'])
        loader_A_iter =  iter(data_generator['A_loader']) if 'A_loader' in data_generator else None

        train_start = time()
        for idx in range(n_batch):
            btime= time()

            try:
                batch_data = next(loader_iter)
            except:
                loader_iter = iter(data_generator['loader'])
                batch_data = next(loader_iter)
            if args.model_type == 'cke':
                try:
                    batch_A_data = next(loader_A_iter)
                except:
                    loader_A_iter = iter(data_generator['A_loader'])
                    batch_A_data = next(loader_A_iter)

                if data_generator['dataset'].batch_style == 'list':
                    batch_data = (*batch_data, *batch_A_data)
                else:
                    batch_data.update(batch_A_data)

                feed_dict = data_generator['A_dataset'].as_train_feed_dict(model, batch_data)
            else:
                feed_dict = data_generator['dataset'].as_train_feed_dict(model, batch_data)
            #    feed_dict = data_generator['dataset'].as_train_feed_dict(model,
            #                                    batch_data)
            #print(batch_data)
            #feed_dict = data_generator['dataset'].as_train_feed_dict(model, batch_data)#*batch_data)
            #print(feed_dict)
            #import time as time_
            #time_.sleep(5)
            _, batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train(sess, feed_dict=feed_dict)

            loss += batch_loss
            base_loss += batch_base_loss
            kge_loss += batch_kge_loss
            reg_loss += batch_reg_loss


        train_time += time()-train_start
        if args.wandb:
            log_dict = {'train_total_loss':loss,
                'train_base_loss':base_loss,
                'train_reg_loss':reg_loss,
                'train_kge_loss':kge_loss,
                'train_time': train_time}
            if  args.model_type != 'kgat':
                wandb.log(log_dict)
            elif  args.model_type == 'kgat' and args.use_kge == False:
                wandb.log(log_dict)
        if np.isnan(loss) == True:
            print('ERROR: loss@phase1 is nan.')
            sys.exit()

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
        """
        if args.model_type in ['kgat']:

            n_A_batch = len(data_generator['A_dataset'].all_h_list) // args.batch_size_kg + 1

            if args.use_kge is True:
                # using KGE method (knowledge graph embedding).
                train_start = time()
                loader_A_iter =  iter(data_generator['A_loader'])
                for idx in range(n_A_batch):
                    btime = time()


                    try:
                        A_batch_data = next(loader_A_iter)
                    except:
                        loader_A_iter = iter(data_generator['A_loader'])
                        A_batch_data = next(loader_A_iter)
                    #substitute for data_generator.generate_train_A_batch()
                    feed_dict = data_generator['A_dataset'].as_train_A_feed_dict(model, A_batch_data)#data_generator.generate_train_A_feed_dict(model, A_batch_data)

                    _, batch_loss, batch_kge_loss, batch_reg_loss = model.train_A(sess, feed_dict=feed_dict)

                    loss += batch_loss
                    kge_loss += batch_kge_loss
                    reg_loss += batch_reg_loss



                train_time += time() - train_start
                log_dict = {'train_total_loss':loss,
                    'train_base_loss':base_loss,
                    'train_reg_loss':reg_loss,
                    'train_kge_loss':kge_loss,
                    'time':  train_time}
                if args.wandb :
                    wandb.log(log_dict)

            if args.use_att is True:
                # updating attentive laplacian matrix.
                model.update_attentive_A(sess)

        if np.isnan(loss) == True:
            print('ERROR: loss@phase2 is nan.')
            sys.exit()

        show_step = 10
        if (epoch + 1) % show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
                print(perf_str)
            continue

        """
        *********************************************************
        Test.
        """
        t2 = time()
        users_to_test = list(data_generator['dataset'].test_user_dict.keys())

        ret, top_k = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
        os.makedirs(LOG_DATASET_DIR[args.dataset], exist_ok=True)
        topk_path = f'{LOG_DATASET_DIR[args.dataset]}/item_topk.pkl'
        with open(topk_path, 'wb') as f:
            pickle.dump(top_k, f)
            print('Saved topK to: ', topk_path)
        """
        *********************************************************
        Performance logging.
        """
        t3 = time()
        metrics.log('train_loss', loss.item())
        metrics.log('train_base_loss', base_loss.item())
        metrics.log('train_kge_loss', kge_loss.item())
        metrics.log('train_reg_loss',reg_loss.item())
        metrics.log('valid_ndcg',ret['ndcg'].item())
        metrics.log('valid_hit',ret['hit_ratio'].item())
        metrics.log('valid_recall',ret['recall'].item())     
        metrics.log('valid_precision',ret['precision'].item())
        metrics.push(['train_loss','train_base_loss', 'train_kge_loss','train_reg_loss',
                        'valid_ndcg','valid_hit','valid_recall','valid_precision'])


        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        metrics_logs ={ }
        for metric_name, metric_values in ret.items():
            if metric_name !='auc':
                for idx, k in enumerate(Ks):
                    metrics_logs[f'{metric_name}@{k}'] = metric_values[idx]
        test_time += t3 -t2
        metrics_logs['test_time'] = test_time
        if args.wandb :
            wandb.log(metrics_logs)

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, base_loss, kge_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=1000)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        #if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
        save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
        print('save the weights in path: ', weights_save_path)

    metrics.write(TEST_METRICS_FILE_PATH[args.dataset])

    metrics.close_wandb()