import torch
import pandas as pd
import numpy as np
import dgl

def create_graph(nodes, ratings) :
    train_u_nodes, train_v_nodes, val_u_nodes, val_v_nodes, test_u_nodes, test_v_nodes = nodes
    train_ratings, val_ratings, test_ratings = ratings
    
    train_g = dgl.heterograph({
        ('user', 'is_rating', 'item') : (train_u_nodes, train_v_nodes),
        ('item', 'is_rated', 'user') : (train_v_nodes, train_u_nodes),
    })

    val_g = dgl.heterograph({
        ('user', 'is_rating', 'item') : (val_u_nodes, val_v_nodes),
        ('item', 'is_rated', 'user') : (val_v_nodes, val_u_nodes),
    })

    test_g = dgl.heterograph({
        ('user', 'is_rating', 'item') : (test_u_nodes, test_v_nodes),
        ('item', 'is_rated', 'user') : (test_v_nodes, test_u_nodes),
    })
    
    train_g.edges['is_rating'].data['r'] = torch.tensor(train_ratings).reshape(-1, 1)
    val_g.edges['is_rating'].data['r'] = torch.tensor(val_ratings).reshape(-1, 1)
    test_g.edges['is_rating'].data['r'] = torch.tensor(test_ratings).reshape(-1, 1)
    
    train_g.edges['is_rated'].data['r'] = torch.tensor(train_ratings).reshape(-1, 1)
    val_g.edges['is_rated'].data['r'] = torch.tensor(val_ratings).reshape(-1, 1)
    test_g.edges['is_rated'].data['r'] = torch.tensor(test_ratings).reshape(-1, 1)
    
    return train_g, val_g, test_g

def split_dataset(data_train, data_test, dtypes) :
    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)
    
    data_array = np.concatenate([data_array_train, data_array_test], axis=0)
    
    train_u_nodes = data_array_train[:, 0].astype(dtypes['u_nodes'])
    train_v_nodes = data_array_train[:, 1].astype(dtypes['v_nodes'])
    train_ratings = np.subtract(data_array_train[:, 2].astype(dtypes['ratings']), 1)

    test_u_nodes = data_array_test[:, 0].astype(dtypes['u_nodes'])
    test_v_nodes = data_array_test[:, 1].astype(dtypes['v_nodes'])
    test_ratings = np.subtract(data_array_test[:, 2].astype(dtypes['ratings']), 1)
    
    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val
    
    rand_idx = list(range(num_train+num_val))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    
    train_idx = rand_idx[:num_train]
    val_idx = rand_idx[num_train:]
    
    train_u_nodes = data_array_train[train_idx, 0].astype(dtypes['u_nodes'])
    train_v_nodes = data_array_train[train_idx, 1].astype(dtypes['v_nodes'])
    train_ratings = np.subtract(data_array_train[train_idx, 2].astype(dtypes['ratings']), 1)

    val_u_nodes = data_array_train[val_idx, 0].astype(dtypes['u_nodes'])
    val_v_nodes = data_array_train[val_idx, 1].astype(dtypes['v_nodes'])
    val_ratings = np.subtract(data_array_train[val_idx, 2].astype(dtypes['ratings']), 1)
    
    
    train_nodes = np.hstack([train_u_nodes.reshape(-1,1), train_v_nodes.reshape(-1,1)]) # 학습 타겟 대상
    test_nodes = np.hstack([test_u_nodes.reshape(-1,1), test_v_nodes.reshape(-1,1)]) # 학습 타겟 대상
    
    nodes = (train_u_nodes, train_v_nodes, val_u_nodes, val_v_nodes, test_u_nodes, test_v_nodes)
    ratings = (train_ratings, val_ratings, test_ratings)
    
    train_g, val_g, test_g = create_graph(nodes, ratings)
    return train_g, val_g, test_g, train_nodes, test_nodes, train_ratings, test_ratings


def load_ml_100k(data_path) :
    sep = '\t'
    dtypes = {'u_nodes': np.int32, 'v_nodes': np.int32, 'ratings': np.int64, 'timestamp': np.float64}
    
    data_train = pd.read_csv(
    data_path + 'u1.base', sep=sep, header=None,
    names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_test = pd.read_csv(
        data_path + 'u1.test', sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)
    
    ratings_type = len(data_train['ratings'].unique())
    train_g, val_g, test_g, train_nodes, test_nodes, train_ratings, test_ratings = split_dataset(data_train, data_test, dtypes)
    
    return train_g, val_g, test_g, train_nodes, test_nodes, train_ratings, test_ratings, ratings_type