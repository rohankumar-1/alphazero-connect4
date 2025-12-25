
import torch
from torch.utils.data import Dataset, DataLoader
import random
from state import Game
import os


def _save_to_safetensor(data:list, path:str):
    state_tensors = []
    policy_tensors = []
    value_tensors = []

    for sample in data:
        state_tensors.append(sample['s_t'].squeeze())
        policy_tensors.append(sample['alpha_t'])
        value_tensors.append(sample['z_t'])

    full= {
        "states": torch.stack(state_tensors),        # Shape: [N, 3, 6, 7] (3 channels, one for X, one for O, one for turn)
        "policies": torch.stack(policy_tensors),     # Shape: [N, 7] (7 possible moves)
        "values": torch.tensor(value_tensors)           # Shape: [N, 1]
    }

    torch.save(full, path)


class PolicyValueDataset(Dataset):
    """ 
    The PolicyValueDataset properly formats data samples, performing the following functions: 
    1) Load in the most recent 4 selfplay iterations: games later than that probably use really weak models, so we can safely ignore 
    2) Average out p,v for states that were seen multiple times
    3) Applies data augmentations (flipping state/policy vectors, since connect-four is symmetric)
    4) Returns single samples for Dataloader
    """

    def __init__(self, window=4, data_dir:str="./data/"):

        self.states = []
        self.policies = []
        self.values = []
        
        for file in sorted(os.listdir(data_dir))[-window:]:
            d = torch.load(os.path.join(data_dir, file), weights_only=True)
            self.states.append(d['states'].squeeze())
            self.policies.append(d['policies'])
            self.values.append(d['values'])
        
        self.states = torch.cat(self.states, dim=0)
        self.policies = torch.cat(self.policies, dim=0)
        self.values = torch.cat(self.values, dim=0).squeeze()
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        return self.states[idx], self.policies[idx], self.values[idx]



class SimpleBot:
    def get_best_move(self, game: Game):
        return random.choice(game.get_valid_moves())


class LookaheadBot:
    """ looks ahead one time """
    def get_best_move(self, game):

        for move in game.get_valid_moves():
            game.make_move(move)
            if game.over():
                game.undo_move()
                return move
            game.undo_move()

        return random.choice(game.get_valid_moves())
