import torch
import torch.nn as nn
from models.CEVIMModel import CEVIMModel

def CEVIM(MDP, n_iter=500, K=100, n_out=100, verbose=False):
    # Recording
    losses_bellman = []
    losses_cum = []

    # Model definition
    alphas = (1-MDP.gamma) * np.power(MDP.gamma, np.arange(K-2, -2, -1)) / (1 - MDP.gamma ** (K+1))
    model = CEVIMModel(MDP, n_out=n_out)

    r = torch.from_numpy(MDP.R).mean(2)

    # Algorithm
    for i in range(n_iter):
        cum_loss = 0

        # Update of the last layer
        for k in range(K):
            V = model(torch.eye(MDP.n_states))[:,0]

            # Compute TV
            Q = r + MDP.gamma * (MDP.P * V[None,None]).sum(2)
            TV = Q.max(1).values
            
            # Loss
            bellman_loss = nn.SmoothL1Loss()(V, TV)
            cum_loss += alphas[k] * bellman_loss

            # Update theta
            bellman_loss.backward(retain_graph=True)
            model.optim_theta.step()
            model.optim_theta.zero_grad()

            losses_bellman.append(bellman_loss.item())

        # Update the rest of the network
        cum_loss.backward()
        model.optim_phi.step()
        model.optim_phi.zero_grad()

        if i % 100 == 0 and i > 0 and verbose:
            print(i, "iterations done over", n_iter)

        losses_cum.append(cum_loss.item())
    
    return model, (losses_bellman, losses_cum)