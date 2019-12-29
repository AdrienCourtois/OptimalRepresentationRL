from torch.autograd import Variable
import torch
import torch.nn as nn

class AVFPolicy(nn.Module):
    def __init__(self, MDP):
        super(AVFPolicy, self).__init__()

        self.policy = Variable(torch.rand(MDP.n_states, MDP.n_actions), requires_grad=True)

        self.optimizer = torch.optim.Adam([self.policy])
        self.is_cuda = False
    
    def cuda(self):
        self.is_cuda = True
        self.policy = self.policy.cuda()

        return self
    
    def get_policy(self): # x \in \mathcal{X}
        return torch.softmax(self.policy, -1)