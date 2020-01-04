import torch
import numpy as np
import matplotlib.pyplot as plt

from models.AVFRepresentation import AVFRepresentation
from MDPs.MDPTwoRoom import MDPTwoRoom

MDP = MDPTwoRoom()
avf_m = AVFManager.load("/content/gdrive/My Drive/Colab Notebooks/models/AVFs1000_1000_2Rooms_2.pkl")
representation = AVFRepresentation(MDP)

# Hyper parameters
batch_size = 32
n_iter = 20000

losses = []

for i in range(n_iter):
    # Retrieve batch
    idx = np.random.randint(0, len(avf_m), batch_size)

    # Compute V
    V = torch.zeros(batch_size, MDP.n_states, 1)
    for j in range(batch_size):
        V[j] = avf_m.compute_V(avf_m[idx[j]])

    # Compute theta
    Phi = representation(torch.eye(MDP.n_states)).t()
    A = torch.mm(torch.mm(Phi, Phi.t()).pinverse(rcond=1e-10), Phi)
    theta = torch.matmul(V.permute(0,2,1), A.t()[None])
    
    # Compute \hat{V}
    V_hat = (Phi.t()[None] * theta).sum(2, keepdim=True)

    # Computation of the loss
    loss = nn.MSELoss()(V_hat, V)

    # Update
    loss.backward()
    representation.optimizer.step()
    representation.optimizer.zero_grad()

    if i % 1000 == 0 and i > 0:
        print(i)

    losses.append(loss.item())

plt.plot(losses)
plt.show()