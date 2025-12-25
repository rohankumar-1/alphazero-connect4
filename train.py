from pandas.core.internals.blocks import new_block_2d

import torch
from torch.utils.data import DataLoader
from zero import selfplay_parallel
from utils import PolicyValueDataset
from model import PolicyValueNetwork
from argparse import ArgumentParser

parser = ArgumentParser("AlphaZero training")
parser.add_argument("--data", "-d")
parser.add_argument("--model", "-m")

if __name__=="__main__":
    args = parser.parse_args()      

    data_path = f"data/iter{int(args.data):03}.safetensors"
    old_model_path = f"models/iter{int(args.model)-1:03}.safetensors"
    new_model_path = f"models/iter{int(args.model):03}.safetensors"

    # PLAY
    selfplay_parallel(games=100, noise=0.3, model_path=old_model_path, outpath=data_path, workers=4)

    # RETRAIN
    dataset = PolicyValueDataset()
    print("Samples in dataset:",  len(dataset))
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    net = PolicyValueNetwork()
    optimizer = torch.optim.AdamW(params=net.parameters())
    net.train_iteration(train_loader, optimizer=optimizer, outpath=new_model_path)