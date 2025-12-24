from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PolicyValueDataset
from tqdm import tqdm


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class PolicyValueNetwork(nn.Module):
    def __init__(self, num_res_blocks=2, num_channels=64):
        super().__init__()
        
        # Initial Convolutional Block
        self.start_block = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual Tower
        self.res_tower = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, 7) # 7 columns in Connect Four
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 6 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start_block(x)
        for block in self.res_tower:
            x = block(x)
        
        policy = self.policy_head(x) # Output log-probabilities (use CrossEntropyLoss)
        value = self.value_head(x)   # Output scalar in [-1, 1]
        
        return policy, value


    def predict(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad(): 
            if x.ndimension() == 3:
                x = x.unsqueeze(0)
            policy_logits, value_tensor = self.forward(x)
    
            probs = torch.softmax(policy_logits, dim=1)
            
        return probs.squeeze(0), value_tensor.item()

    def _save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.state_dict()
        }, path)
        print(f"Saved trained model to {path}")

    def _load_checkpoint(self, path):
        t = torch.load(path, weights_only=True)
        self.load_state_dict(t['model_state_dict'])


    def train_iteration(self, trainloader: DataLoader, optimizer, outpath:str, epochs:int=10):
        self.train()
        for epoch in range(epochs):
            
            for (s, p, v) in tqdm(trainloader, desc=f"Epoch {epoch}:", total=len(trainloader)):
                optimizer.zero_grad()
                pred_policy, pred_value = self.forward(s)

                value_loss: torch.Tensor = F.mse_loss(pred_value, v)
                policy_loss: torch.Tensor = -torch.sum(p * F.log_softmax(pred_policy, dim=1)) / s.size(0)
                total_loss = value_loss + policy_loss

                total_loss.backward()
                optimizer.step()
            

        self._save_checkpoint(outpath)



if __name__=="__main__":
    dataset = PolicyValueDataset()
    print("Samples in dataset:",  len(dataset))
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    net = PolicyValueNetwork()
    optimizer = torch.optim.AdamW(params=net.parameters())
    net.train_iteration(train_loader, optimizer=optimizer, outpath="models/iter002.safetensors")