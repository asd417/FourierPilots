import torch
from torch import nn
import torch_dct as dct

class MLPCosine(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): 
        X = dct.dct(x)
        return self.net(X)

class MLPDoubleCosine(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): 
        X = dct.dct(dct.dct(x))
        return self.net(X)

class MLPCosineTime(nn.Module):
    def __init__(self, past, feature_dim, hidden=64, out_dim=2):
        super().__init__()
        self.past = past
        self.feat_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(past * feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )

    def forward(self, x : torch.Tensor):
        x = dct.dct(x) # DCT performs on last axis by default
        return self.net(x)

class MLPDoubleCosineTime(nn.Module):
    def __init__(self, past, feature_dim, hidden=64, out_dim=2):
        super().__init__()
        self.past = past
        self.feat_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(past * feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )

    def forward(self, x : torch.Tensor):
        # 1) UNFLATTEN
        #x = x.view(x.shape[0], self.past, self.feat_dim)
        x = dct.dct(dct.dct(x))
        #x = x.reshape(x.shape[0], self.past * self.feat_dim)
        # 3) FLATTEN BACK and feedforward
        return self.net(x)

class MLPFourier(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): 
        X = torch.fft.fft(x,norm="ortho")          
        return self.net(X.real)

class MLPFourierTime(nn.Module):
    def __init__(self, past, feature_dim, hidden=64, out_dim=2):
        super().__init__()
        self.past = past
        self.feat_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(past * feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )

    def forward(self, x : torch.Tensor):
        # 1) UNFLATTEN
        x = x.view(x.shape[0], self.past, self.feat_dim)
        x = torch.fft.fft(x,dim=1).real
        x = x.reshape(x.shape[0], self.past * self.feat_dim)
        # 3) FLATTEN BACK and feedforward
        return self.net(x)
    
class MLPFourierTimeFlat(nn.Module):
    def __init__(self, past, feature_dim, hidden=64, out_dim=2):
        super().__init__()
        self.past = past
        self.feat_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(past * feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )

    def forward(self, x : torch.Tensor):
        # 1) UNFLATTEN
        #x = x.view(x.shape[0], self.past, self.feat_dim)
        x = torch.fft.fft(x,dim=1).real
        #x = x.reshape(x.shape[0], self.past * self.feat_dim)
        # 3) FLATTEN BACK and feedforward
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): return self.net(x)

class MLPTime(nn.Module):
    def __init__(self, past, features, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(past * features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits (no activation)
        )
    def forward(self, x): return self.net(x)

