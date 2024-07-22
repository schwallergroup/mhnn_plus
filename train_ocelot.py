import time
import argparse
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from datasets import OCELOTv1, RandomSplitter, OneTarget
from utils import Logger, seed_everything
from models import *


def get_model(name, **kwargs):
    # assert name in [], f"Graph Encoder '{name}' is unavailable!"
    model = globals().get(name)
    if model:
        return model(**kwargs)
    else:
        raise ValueError(f"Model '{name}' not found!")


@torch.no_grad()
def evaluate(args, model, loader, std=None):
    model.eval()
    err = 0.0
    # for MAE
    for batch in loader:
        batch = batch.to(args.device)
        out = model(batch)
        if std is not None:
            err += (out * std - batch.y * std).abs().sum().item()
        else:
            err += (out - batch.y).abs().sum().item()
    mae = err / len(loader.dataset)
    return mae


if __name__ == '__main__':

    print('Task start time:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()

    parser = argparse.ArgumentParser(description='OCELOT training')

    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--target', type=int, default=0, help='target of dataset')

    # Training hyperparameters
    parser.add_argument('--runs', default=3, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--log_steps', type=int, default=1)

    # Model hyperparameters
    parser.add_argument('--method', default='mhnn', help='model type')
    parser.add_argument('--All_num_layers', default=3, type=int, help='number of basic blocks')
    parser.add_argument('--MLP1_num_layers', default=2, type=int, help='layer number of mlps')
    parser.add_argument('--MLP2_num_layers', default=2, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=2, type=int, help='layer number of mlp3')
    parser.add_argument('--MLP4_num_layers', default=2, type=int, help='layer number of mlp4')
    parser.add_argument('--MLP_hidden', default=64, type=int, help='hidden dimension of mlps')
    parser.add_argument('--output_num_layers', default=2, type=int)
    parser.add_argument('--output_hidden', default=64, type=int)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normalization', default='ln', choices=['bn', 'ln', 'None'])
    parser.add_argument('--activation', default='relu', choices=['Id', 'relu', 'prelu'])
    parser.add_argument('--dropout', default=0.0, type=float)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load dataset and normalize targets to mean = 0 and std = 1
    transform = T.Compose([OneTarget(target=args.target)])
    dataset = OCELOTv1(root=args.data_dir, transform=transform)
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, args.target].item(), std[:, args.target].item()

    # load logger
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        # set global seed for this run
        seed = args.seed + run
        seed_everything(seed=seed, workers=True)
        print(f'\nRun No. {run+1}:')
        print(f'Seed: {seed}\n')

        # split data for each run
        splitter = RandomSplitter()
        train_idx, valid_idx, test_idx = splitter.split(dataset, seed=seed)
        train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[valid_idx], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False)

        # initialize model etc.
        # initialize model etc.
        if args.method == 'mhnn':
            model = MHNN(1, args)
        else:
            model = get_model(
                args.method,
                hid_channels=args.MLP_hidden,
                num_layer=args.All_num_layers,
                drop_rate=args.dropout,
            )

        model = model.to(args.device)
        print("# Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.8,
                                                               patience=5,
                                                               min_lr=0.00001)

        # training
        best_val_mae = None
        for epoch in range(1, 1 + args.epochs):
            model.train()
            loss_all = 0.0
            lr = scheduler.optimizer.param_groups[0]['lr']
            for data in train_loader:
                data = data.to(args.device)
                optimizer.zero_grad()
                out = model(data)
                loss = loss_fn(out, data.y)
                loss.backward()
                loss_all += loss.item() * data.num_graphs
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            loss_all /= len(train_loader.dataset)
            valid_mae = evaluate(args, model, valid_loader, std=std)
            scheduler.step(valid_mae)
            if best_val_mae is None or valid_mae < best_val_mae:
                test_mae = evaluate(args, model, test_loader, std=std)
                best_val_mae = valid_mae
            logger.add_result(run, [loss_all, valid_mae, test_mae])

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'lr: {lr:.6f}, '
                      f'Loss: {loss_all:.6f}, '
                      f'Valid MAE: {valid_mae:.6f}, '
                      f'Test MAE: {test_mae:.6f}')

        logger.print_statistics(run)
    logger.print_statistics()

    print('Task end time:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    end_time = time.time()
    print('Total time taken: {} s.'.format(int(end_time - start_time)))
