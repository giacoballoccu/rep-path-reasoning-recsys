'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os
import sys
from models.traditional.helper import *
from batch_test import *
from time import time

from FM import FM


import os
import sys

from batch_test import data_generator

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_pretrained_data(args):
    pre_model = 'mf'
    if args.pretrain == -2:
        pre_model = 'kgat'
    pretrain_path = os.path.join(args.data_path, args.dataset, "preprocessed", "kgat", "pretrain", model.model_type,
                                                  f"{pre_model}.npz")
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # get argument settings.

    tf.set_random_seed(2019)
    np.random.seed(2019)
    args = parse_args()

    STORAGE_DIR = f'{args.model_type}_{args.dataset}'
    import os
    import wandb
    os.makedirs(STORAGE_DIR, exist_ok=True)

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities

    if args.model_type in ['kgat', 'cfkg']:
        "Load the laplacian matrix."
        config['A_in'] = sum(data_generator.lap_list)

        "Load the KG triplets."
        config['all_h_list'] = data_generator.all_h_list
        config['all_r_list'] = data_generator.all_r_list
        config['all_t_list'] = data_generator.all_t_list
        config['all_v_list'] = data_generator.all_v_list

    t0 = time()

    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    if args.pretrain in [-1, -2]:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None

    """
    *********************************************************
    Select one of the models.
    """
    model = FM(data_config=config, pretrain_data=pretrain_data, args=args)

    saver = tf.train.Saver()

    """
    *********************************************************
    Save the model parameters.
    """
    if args.save_flag == 1:
        if args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
            weights_save_path = os.path.join(args.data_path, args.dataset, "preprocessed", "kgat", "weights", args.model_type, args.layer,
                                             str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        elif args.model_type in ['ncf', 'nfm', 'kgat']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            weights_save_path = os.path.join(args.data_path, args.dataset, "preprocessed", "kgat", "weights", args.model_type, layer,
                                             str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the model parameters to fine tune.
    """
    if args.pretrain == 1:
        if args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
            pretrain_path = os.path.join(args.data_path, args.dataset, "preprocessed", "kgat", "weights", args.model_type, args.layer,
                                         f"l{str(args.lr)}_r{'-'.join([str(r) for r in eval(args.regs)])}")
        elif args.model_type in ['ncf', 'nfm', 'kgat']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            pretrain_path = os.path.join(args.data_path, args.dataset, "preprocessed", "kgat","weights", args.model_type, args.layer,
                                         f"l{str(args.lr)}_r{'-'.join([str(r) for r in eval(args.regs)])}")

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from the model to fine tune.
            if args.report != 1:
                users_to_test = list(data_generator['dataset'].test_user_dict.keys())

                ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
                print(pretrain_ret)

                # *********************************************************
                # save the pretrained model parameters of mf (i.e., only user & item embeddings) for pretraining other models.
                if args.save_flag == -1:
                    user_embed, item_embed = sess.run(
                        [model.weights['user_embed'], model.weights['entity_embed']],
                        feed_dict={})
                    temp_save_path = os.path.join(args.data_path, args.dataset, "preprocessed", "kgat", "pretrain", model.model_type,
                                                  f"{model.model_type}.npz")

                    ensureDir(temp_save_path)
                    np.savez(temp_save_path, user_embed=user_embed, item_embed=item_embed)
                    print('save the weights of fm in path: ', temp_save_path)

                # *********************************************************
                # save the pretrained model parameters of kgat (i.e., user & item & kg embeddings) for pretraining other models.
                if args.save_flag == -2:
                    user_embed, entity_embed, relation_embed = sess.run(
                        [model.weights['user_embed'], model.weights['entity_embed'], model.weights['relation_embed']],
                        feed_dict={})

                    temp_save_path = os.path.join(args.data_path, args.dataset, "preprocessed", "kgat", "pretrain", model.model_type,
                                                  f"{model.model_type}.npz")
                    ensureDir(temp_save_path)
                    np.savez(temp_save_path, user_embed=user_embed, entity_embed=entity_embed,
                             relation_embed=relation_embed)
                    print('save the weights of kgat in path: ', temp_save_path)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the final performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator['dataset'].get_sparsity_split()

        users_to_test_list.append(list(data_generator['dataset'].test_user_dict.keys()))
        split_state.append('all')

        save_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(save_path)
        f = open(save_path, 'w')
        f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n' % (args.embed_size, args.lr, args.regs,
                                                                       args.loss_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

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

        ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

        """
        *********************************************************
        Performance logging.
        """
        t3 = time()

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

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=1000)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = os.path.join(args.proj_path, "output", args.dataset, "preprocessed", "kgat", f"{model.model_type.result}")
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s, use_att=%s, use_kge=%s, pretrain=%d\n\t%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs, args.adj_type, args.use_att, args.use_kge, args.pretrain, final_perf))
    f.close()