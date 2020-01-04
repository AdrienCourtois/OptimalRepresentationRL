import torch
import torch.nn as nn

class AVFRepresentation(nn.Module):
    def __init__(self, MDP, d=16):
        super(AVFRepresentation, self).__init__()
        
        self.MDP = MDP
        self.d = d

        self.fc1 = nn.Linear(self.MDP.n_states, 512)
        self.fc2 = nn.Linear(512, d)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00025)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x