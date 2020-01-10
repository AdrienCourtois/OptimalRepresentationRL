import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, MDP, n_hidden=512):
        super(Model, self).__init__()

        self.MDP = MDP
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(self.MDP.n_states, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, 1)

        self.optim = torch.optim.Adam(self.parameters())
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def get_naive_V(MDP, n_iter=1000, n_hidden=512):
    r = torch.from_numpy(MDP.R).mean(2)
    model = Model(MDP, n_hidden=n_hidden)

    for i in range(n_iter):
        V = model(torch.eye(MDP.n_states))[:,0]

        # Compute TV
        Q = r + MDP.gamma * (MDP.P * V[None,None]).sum(2)
        TV = Q.max(1).values

        loss = nn.SmoothL1Loss()(V, TV)

        # Update
        loss.backward()
        model.optim.step()
        model.optim.zero_grad()
    
    V = model(torch.eye(MDP.n_states))[:,0]

    return V