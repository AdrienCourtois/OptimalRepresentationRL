import torch
import torch.nn as nn
import torch.nn.functional as F

class Phi(nn.Module):
    def __init__(self, MDP, n_out, n_hidden=512):
        super(Phi, self).__init__()

        self.MDP = MDP

        self.n_out = n_out
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(self.MDP.n_states, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_out)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CEVIMModel(nn.Module):
    def __init__(self, MDP, n_out=100):
        super(CEVIMModel, self).__init__()

        self.MDP = MDP
        self.n_out = n_out

        self.phi = Phi(self.MDP, self.n_out)
        self.theta = nn.Linear(self.n_out, 1)
        
        self.optim_phi = self.phi.optim
        self.optim_theta = torch.optim.Adam(self.theta.parameters(), lr=1e-4)
    
    def forward(self, x):
        x = F.leaky_relu(self.phi(x))
        x = self.theta(x)

        return x