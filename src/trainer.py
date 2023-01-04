import logging
from tqdm import tqdm
import dgl
import torch
import torch.nn.functional as F
from torch.optim import Adam


def set_logger(file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')
    file_handler = logging.FileHandler(f'./log/{file_name}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def train(model, train_dataloader, test_dataloader, num_bases, num_rels, epochs, ARR, lr, weight_decay, logger):

    e_pbar = tqdm(range(1, epochs+1))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_rmse = int(1e9)
    rmses = []
    for epoch in e_pbar:
        total_loss = 0
        model.train()

        for graph, label in train_dataloader:
            optimizer.zero_grad()

            g = dgl.to_homogeneous(
                graph, ndata=['feature'], edata=['r']).to(device)
            target = torch.tensor(label).to(device).to(torch.float32)

            out = model(g)

            loss = F.mse_loss(out, target)

            for idx, gconv in enumerate(model.convs):
                w = torch.matmul(
                    gconv.linear_r.coeff,
                    gconv.linear_r.W.view(num_bases, -1)
                ).view(num_rels, model.dimensions[idx], model.dimensions[idx+1])
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                loss += ARR * reg_loss

            loss.backward()

            total_loss += loss.item() * 64  # batch_size
            optimizer.step()
            torch.cuda.empty_cache()
        train_loss = total_loss / len(train_dataloader.dataset)

        ######### eval ############
        loss = 0

        model.eval()
        for graph, label in test_dataloader:
            g = dgl.to_homogeneous(
                graph, ndata=['feature'], edata=['r']).to(device)
            target = torch.tensor(label).to(device).to(torch.float32)

            with torch.no_grad():
                out = model(g)

            loss += F.mse_loss(out, target, reduction='sum').item()

            torch.cuda.empty_cache()

        val_rmse = loss / len(test_dataloader.dataset)
        rmses.append(val_rmse)

        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_rmse': rmses[-1],
        }

        e_pbar.set_description(
            'Epoch {} || train loss : {:.6f}  |  test_rmse {:.6f}  | '.format(
                *eval_info.values())
        )
        logger.info('Epoch {} || train loss : {:.6f}  |  test_rmse {:.6f}  | '.format(
            *eval_info.values()))
        # print('Epoch {} || train loss : {:.6f}  |  test_rmse {:.6f}  | '.format(
        #     *eval_info.values()))
        if best_val_rmse > val_rmse:
            best_val_rmse = val_rmse
            # torch.save(model.state_dict(), f'./{file_name}.pt')
