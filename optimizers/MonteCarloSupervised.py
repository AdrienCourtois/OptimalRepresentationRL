# Approximate MonteCarlo

class ValueFunction(nn.Module):
    def __init__(self, representation, MDP):
        super(ValueFunction, self).__init__()

        self.representation = representation
        self.MDP = MDP

        self.r = torch.from_numpy(self.MDP.R).mean(2)
        self.p = self.MDP.P

        self.theta = Variable(torch.rand(self.representation.d), requires_grad=True)
        self.optim = torch.optim.Adam([self.theta], lr=1e-3)
    
    def get_V(self):
        Phi = self.representation(torch.eye(self.MDP.n_states)).detach()
        V = (Phi * self.theta[None]).sum(1)

        return V
    
    def forward(self, x):
        # Returns pi[x]!
        # Input must be batch-like, onehot

        V = self.get_V()
        x_idx = x.argmax(1)
        pi = torch.zeros(x.size(0), self.MDP.n_actions)

        for i in range(x.size(0)):
            pi[i] = self.r[x_idx[i]] + self.MDP.gamma * (self.p[x_idx[i],:] * V[None]).sum(1)
        
        # renormalize
        pi = torch.softmax(pi, -1)

        return pi
    
    def select_action(self, state):
        pi = self(state)
        action = pi.argmax(1)

        return action

    def evaluate(self):
        observation = self.MDP.reset()
        reward_episode = 0
        nb = 0
        done = False
        g = 1
            
        while not done:
            action = self.select_action(torch.eye(self.MDP.n_states)[observation][None])
            observation, reward, done, info = self.MDP.step(int(action))
            observation = observation
            reward_episode += g * reward
            g *= self.MDP.gamma

            nb += 1

            if nb > 100:
                break
        
        print(f'Reward: {reward_episode} : {nb}')
    
    def get_objective(self):
        V = self.get_V()
        Q = self.r + self.MDP.gamma * (self.p * V[None,None]).sum(2)
        
        return Q.max(1).values.detach().float()

V = ValueFunction(representation, MDP)
metrics = []

for t in range(5000):
    initial_state = np.random.randint(0, MDP.n_states)

    observation = MDP.reset(initial_state)
    reward_episode = 0
    nb = 0
    done = False
    g = 1
        
    while not done:
        action = V.select_action(torch.eye(MDP.n_states)[observation][None])
        observation, reward, done, info = MDP.step(int(action))
        observation = observation
        reward_episode += g * reward

        g *= MDP.gamma

        nb += 1

        if nb > 100:
            break
    
    V_ = V.get_V()
    
    loss = nn.SmoothL1Loss()(V_[initial_state], torch.tensor(reward_episode))
    loss.backward()

    V.optim.step()
    V.optim.zero_grad()

    metrics.append(torch.norm(V_true - V.get_V()).item())

plt.plot(metrics)
plt.show()