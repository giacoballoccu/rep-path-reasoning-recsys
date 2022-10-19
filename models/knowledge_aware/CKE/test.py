from batch_test import *
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow.compat.v1 as tf
import numpy as np
from CKE import CKE
import os
import sys
from utils import *
import pickle
tf.disable_v2_behavior()


if __name__ == '__main__':
    # get argument settings.

    tf.set_random_seed(2019)
    np.random.seed(2019)
    args = parse_args()


    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator['dataset'].n_users
    config['n_items'] = data_generator['dataset'].n_items
    config['n_relations'] = data_generator['dataset'].n_relations
    config['n_entities'] = data_generator['dataset'].n_entities
    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    
    model = CKE(data_config=config, pretrain_data=None, args=args)
    users_to_test = list(data_generator['dataset'].test_user_dict.keys())
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()


    if args.pretrained_weights is not None  and os.path.exists(args.pretrained_weights):
        pretrain_path =  args.pretrained_weights    
    else:
        pretrain_path = f'../../../data/{args.dataset}/preprocessed/kgat/tmp/{MODEL}/weights/0.00011e-05-1e-05-0.01'
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('load the pretrained model parameters from: ', pretrain_path)


    ret, top_k = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
    topk_path = f'{TMP_DIR[args.dataset]}/item_topk.pkl'
    with open(topk_path, 'wb') as f:
        pickle.dump(top_k, f)
        print('Saved topK to: ', topk_path)
