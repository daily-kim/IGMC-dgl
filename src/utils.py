import torch
import dgl
import numpy as np
from easydict import EasyDict
import yaml
import logging


def one_hot(idx, length):
    idx = np.array(idx, dtype=np.int32)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return torch.tensor(x, dtype=torch.float32)


def extract_subgraph(target_u_node, target_v_node, graph, h, train):
    # u_dist, v_dist = [0], [0] # 서브 그래프 내에서의 labeling
    # init node labels in subgraph
    num_users = graph.num_nodes('user')
    num_items = graph.num_nodes('item')

    user_sub_labels = torch.fill_(torch.empty(num_users+1, 1), -1)
    item_sub_labels = torch.fill_(torch.empty(num_items+1, 1), -1)

    # set target node labels
    user_sub_labels[target_u_node] = 0
    item_sub_labels[target_v_node] = 1

    u_visited, v_visited = set([target_u_node]), set([target_v_node])
    u_fringe, v_fringe = torch.tensor([target_u_node], dtype=torch.int64), torch.tensor([
        target_v_node], dtype=torch.int64)

    for dist in range(1, h+1):  # loop for hop
        u_fringe, v_fringe = graph.in_edges((v_fringe), etype='is_rating')[0], \
            graph.in_edges((u_fringe), etype='is_rated')[0]

        u_fringe = set(u_fringe.numpy()) - u_visited
        v_fringe = set(v_fringe.numpy()) - v_visited

        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)

        u_fringe, v_fringe = torch.tensor(
            list(u_fringe)), torch.tensor(list(v_fringe))

        user_sub_labels[u_fringe.numpy()] = 2*dist
        item_sub_labels[v_fringe.numpy()] = 2*dist+1

    subgraph = dgl.node_subgraph(graph, {'user': list(
        u_visited), 'item': list(v_visited)}, store_ids=True)

    # remove edge between target nodes
    target_u_node_idx = (
        subgraph.nodes['user'].data['_ID'] == target_u_node).nonzero(as_tuple=True)[0]
    target_v_node_idx = (
        subgraph.nodes['item'].data['_ID'] == target_v_node).nonzero(as_tuple=True)[0]

    if train:
        edge_idx = subgraph.edge_ids(
            target_u_node_idx, target_v_node_idx, etype=('user', 'is_rating', 'item'))
        subgraph.remove_edges(edge_idx, 'is_rating')

        edge_idx = subgraph.edge_ids(
            target_v_node_idx, target_u_node_idx, etype=('item', 'is_rated', 'user'))
        subgraph.remove_edges(edge_idx, 'is_rated')

    # set node features in subgraph
    user_sub_labels = user_sub_labels[torch.ne(
        user_sub_labels, torch.tensor([-1]))]
    item_sub_labels = item_sub_labels[torch.ne(
        item_sub_labels, torch.tensor([-1]))]

    user_sub_feats = one_hot(user_sub_labels, 4)
    item_sub_feats = one_hot(item_sub_labels, 4)

    subgraph.ndata['feature'] = {
        'user': user_sub_feats, 'item': item_sub_feats}

    return subgraph


def collate_fn(data):
    g_list, label_list = map(list, zip(*data))
    g_batch = dgl.batch(g_list)
    return g_batch, label_list


def get_logger(name, path):

    logger = logging.getLogger(name)

    if len(logger.handlers) > 0:
        return logger  # Logger already exists

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=path)

    console.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


def get_args_from_yaml(yaml_path):

    with open('train_configs/common_configs.yaml') as f:
        common_cfgs = yaml.load(f, Loader=yaml.FullLoader)
    data_cfg = common_cfgs['dataset']
    model_cfg = common_cfgs['model']
    train_cfg = common_cfgs['train']

    with open(yaml_path) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    exp_data_cfg = cfgs.get('dataset', dict())
    exp_model_cfg = cfgs.get('model', dict())
    exp_train_cfg = cfgs.get('train', dict())

    for k, v in exp_data_cfg.items():
        data_cfg[k] = v
    for k, v in exp_model_cfg.items():
        model_cfg[k] = v
    for k, v in exp_train_cfg.items():
        train_cfg[k] = v

    args = EasyDict(
        {
            'key': cfgs['key'],

            'datapath': data_cfg.get('path'),
            'dataset': data_cfg.get('name'),
            # 'keywords': data_cfg.get('keywords'),
            # 'keyword_edge_k': data_cfg.get('keyword_edge_k'),
            # 'additional_feature': data_cfg.get('additional_feature'),
            'max_seq': data_cfg.get('max_seq'),

            # model configs
            'model_type': model_cfg['type'],
            'hop': model_cfg['hop'],
            'in_nfeats': model_cfg.get('in_nfeats'),
            'out_nfeats': model_cfg.get('out_nfeats'),
            'in_efeats': model_cfg.get('in_efeats'),
            'out_efeats': model_cfg.get('out_efeats'),
            'num_heads': model_cfg.get('num_heads'),
            'node_features': model_cfg.get('node_features'),
            'parameters': model_cfg.get('parameters'),
            'num_relations': model_cfg.get('num_relations', 5),
            'edge_dropout': model_cfg['edge_dropout'],
            'num_bases': model_cfg['num_bases'],
            'ARR': model_cfg['ARR'],

            'latent_dims': model_cfg.get('latent_dims'),  # baseline model

            # train configs
            'device': train_cfg['device'],
            'log_dir': train_cfg['log_dir'],
            'log_interval': train_cfg.get('log_interval'),
            'train_lrs': [float(lr) for lr in train_cfg.get('learning_rates')],
            'train_epochs': train_cfg.get('epochs'),
            'batch_size': train_cfg['batch_size'],
            'weight_decay': train_cfg.get('weight_decay', 0),
            'lr_decay_step': train_cfg.get('lr_decay_step'),
            'lr_decay_factor': train_cfg.get('lr_decay_factor'),

        }
    )

    return args


if __name__ == '__main__':
    with open('./train_configs/train_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
        file_list = files['files']
    for f in file_list:
        print(get_args_from_yaml(f))
