import torch

def compute_V_optimal(MDP, niter=10000):
    r = torch.from_numpy(MDP.R).mean(2)
    V = torch.rand(MDP.n_states)
    gamma = MDP.gamma

    # Compute optimal V
    for k in range(niter):
        q = r + gamma * (MDP.P * V[None,None]).sum(2)
        
        V = q.max(1).values
    
    return V