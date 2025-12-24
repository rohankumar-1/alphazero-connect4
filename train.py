
import torch
from torch.utils.data import DataLoader
from zero import selfplay_parallel
from utils import PolicyValueDataset
from model import PolicyValueNetwork


STARTING_ITERATION = 2
ENDING_ITERATION = 10

if __name__=="__main__":

    for i in range(STARTING_ITERATION, ENDING_ITERATION):
        data_path = f"data/iter{i:03}.safetensors"
        model_path = f"models/iter{i:03}.safetensors"
        print("#"*35)
        print()
        print(f"ITERATION {i}")
        print()
        print("#"*35)

        # PLAY
        selfplay_parallel(games=100, noise=0.3, model_path=model_path, outpath=data_path, workers=4)

        # RETRAIN
        dataset = PolicyValueDataset()
        print("Samples in dataset:",  len(dataset))
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        net = PolicyValueNetwork()
        optimizer = torch.optim.AdamW(params=net.parameters())
        net.train_iteration(train_loader, optimizer=optimizer, outpath=model_path)