import torch
from torch.autograd import Variable
import numpy as np

from models.AVFRepresentation import AVFRepresentation


def AVF(MDP, avf_m, batch_size=32, n_iter=200000, d=100, verbose=False):
    # Recording
    losses = []

    # Theta AVF
    theta = Variable(torch.rand(len(avf_m), d), requires_grad=True)
    optim_theta = torch.optim.Adam([theta], lr=1e-5)

    # Theta objective
    obj_theta = Variable(torch.rand(d), requires_grad=True)
    optim_obj_theta = torch.optim.Adam([obj_theta], lr=1e-5)

    r = torch.from_numpy(MDP.R).mean(2)
    representation = AVFRepresentation(MDP, d=d)

    # Precomputation of V_AVF
    V_avf = torch.zeros(len(avf_m), MDP.n_states, 1)
    for j in range(len(avf_m)):
        V_avf[j] = avf_m.compute_V(avf_m[j])
    
    # Optimization
    for i in range(n_iter):
        # Retrieve batch
        idx = np.random.randint(0, len(avf_m), batch_size)

        # Compute V
        V = V_avf[idx] 
        
        # Compute \hat{V}
        Phi = representation(torch.eye(MDP.n_states)).t()
        V_hat = (Phi.t()[None]*theta[idx][:,None]).sum(2, keepdim=True)
        
        # Computation of the loss
        loss = nn.SmoothL1Loss()(V_hat, V)

        # Bellman objective
        V_obj = (Phi.t() * obj_theta[None]).sum(1)
        V_obj_target = (r + MDP.gamma * (MDP.P * V_obj[None,None]).sum(2)).max(1).values.float()

        loss = loss + nn.SmoothL1Loss()(V_obj, V_obj_target)

        # Update phi
        loss.backward()
        representation.optimizer.step()
        representation.optimizer.zero_grad()

        # Update theta objective
        optim_obj_theta.step()
        optim_obj_theta.zero_grad()

        # Update theta
        optim_theta.step()
        optim_theta.zero_grad()
        
        if i % 5000 == 0 and i > 0 and verbose:
            print(i)

        losses.append(loss.item())
    
    return representation, obj_theta, theta, losses