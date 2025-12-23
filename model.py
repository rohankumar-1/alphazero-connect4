
import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicNet(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, X):
        return F.softmax(torch.rand((7,)),dim=0), torch.rand((1,)).item()




if __name__=="__main__":
    n = BasicNet()
    p, v = n(torch.rand((7,)))
    print(p)