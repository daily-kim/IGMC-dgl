import torch
import argparse
from torch.utils.data import Dataset, DataLoader

from preprocessing import *
from utils import *
from trainer import *
from dataset import *
from model import *

def main(args) :
    if args.dataset_name == 'ml_100k' :
        train_g, val_g, test_g, train_nodes, test_nodes, train_ratings, test_ratings, ratings_type = load_ml_100k(args.data_path + args.dataset_name + '/')
    
    train_dataset = IGMCDataset(train_g, train_nodes, train_ratings, ratings_type, args.hop, True)
    test_dataset = IGMCDataset(train_g, test_nodes, test_ratings, ratings_type, args.hop, False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn = collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn = collate_fn)
    
    num_bases = 4
    num_rels = ratings_type
    model = IGMC(4, num_relations = ratings_type, num_bases = 4)
    train(model, train_dataloader, test_dataloader, num_bases, num_rels, args.file_name, args.epochs, args.ARR, args.lr, args.weight_decay)
    
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ml_100k')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--ARR', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--file_name', type=str)
    
    args = parser.parse_args()
    main(args)
    