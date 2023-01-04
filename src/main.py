import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import yaml
from collections import defaultdict

from preprocessing import load_ml_100k
from utils import get_logger, get_args_from_yaml, collate_fn
from trainer import train
from dataset import IGMCDataset
from model.igmc import IGMC


def main(args):
    with open('./train_configs/train_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
        file_list = files['files']
    for f in file_list:
        date_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        args = get_args_from_yaml(f)
        logger = get_logger(
            name=args.key, path=f"{args.log_dir}/{args.key}.log")
        logger.info('train args')

        for k, v in args.items():
            logger.info(f'{k}: {v}')

        if args.dataset == 'ml-100k':
            train_g, val_g, test_g, train_nodes, test_nodes, train_ratings, test_ratings, ratings_type = load_ml_100k(
                args.datapath + '/'+args.dataset + '/')

        train_dataset = IGMCDataset(
            train_g, train_nodes, train_ratings, ratings_type, args.hop, True)
        test_dataset = IGMCDataset(
            train_g, test_nodes, test_ratings, ratings_type, args.hop, False)
        NUM_WORKER = 16

        train_dataloader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKER, collate_fn=collate_fn)
        test_dataloader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKER, collate_fn=collate_fn)

        model = IGMC(4, num_relations=ratings_type, num_bases=args.num_bases)
        # train(model, train_dataloader, test_dataloader, num_bases=args.num_bases, num_rels=ratings_type, args)
# def train(model, train_dataloader, test_dataloader, num_bases, num_rels, file_name, epochs, ARR, lr, weight_decay):
        train(model, train_dataloader,
              test_dataloader, num_bases=args.num_bases, num_rels=ratings_type, epochs=args.train_epochs, ARR=args.ARR, lr=1e-3, weight_decay=args.weight_decay, logger=logger)
        # test_results = defaultdict(list)
        # best_lr = None
        # # for data_name in args.datasets:
        # sub_args = args
        # # sub_args['data_name'] = data_name
        # best_rmse_list = []
        # for lr in args.train_lrs:
        #     sub_args['train_lr'] = lr
        #     best_rmse = train(model, train_dataloader,
        #                       test_dataloader, num_bases=args.num_bases, num_rels=ratings_type, epochs=args.train_epochs, ARR=args.ARR, lr=lr, weight_decay=args.weight_decay, logger=logger)
        #     # test_results[data_name].append(best_rmse)
        #     best_rmse_list.append(best_rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_name', type=str, default='ml_100k')
    # parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--ARR', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--file_name', type=str)

    args = parser.parse_args()
    main(args)
