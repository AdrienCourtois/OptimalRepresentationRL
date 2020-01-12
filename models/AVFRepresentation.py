import torch
import torch.nn as nn
import torch.nn.functional as F

class AVFRepresentation(nn.Module):
    def __init__(self, MDP, d=100):
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
    
    def load(self, path):
        temp = torch.load(path)

        self.fc1.weight.value = temp['fc1.weight']
        self.fc1.bias.value = temp['fc1.bias']

        self.fc2.weight.value = temp['fc2.weight']
        self.fc2.bias.value = temp['fc2.bias']