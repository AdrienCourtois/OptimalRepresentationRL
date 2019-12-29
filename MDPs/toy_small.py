import torch

class MDP:
    def __init__(self):
        self.n_actions = 4
        self.n_states = 10
        self.gamma = 0.99

        self.P = torch.rand(self.n_states, self.n_actions, self.n_states)
        for x in range(self.n_states):
            for a in range(self.n_actions):
                self.P[x, a] = self.P[x, a] / torch.norm(self.P[x,a], p=1)
        
        self.r = torch.zeros(self.n_states, 1)
        self.r[-1, 0] = 1