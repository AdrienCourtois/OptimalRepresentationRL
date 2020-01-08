class ValueFunction(nn.Module):
    def __init__(self, representation, MDP):
        super(ValueFunction, self).__init__()

        self.representation = representation
        self.MDP = MDP

        self.r = torch.from_numpy(self.MDP.R).mean(2)
        self.p = self.MDP.P

        self.theta = torch.rand(self.representation.d)
        self.alpha = 0.8
    
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

        if np.random.rand() >= self.alpha:
            action = torch.multinomial(pi, 1)
        else:
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
        
        print(f'Reward: {reward_episode} : {nb}')
    
    def linear_update(self, s0, s1, r, alpha):
        V = self.get_V()
        phi = self.representation(torch.eye(self.MDP.n_states))[s0]

        alpha *= 0.001

        self.theta = self.theta - alpha * (V[s0].detach() - r - self.MDP.gamma * V[s1].detach()) * phi.detach()

V = ValueFunction(representation, MDP)

mses = []

count_states = torch.zeros(MDP.n_states)

for i in range(1000):
    state = MDP.reset()
    reward_episode = 0
    nb = 0
    done = False
    g = 1

    while not done:
        s0 = state
        count_states[s0] += 1

        action = V.select_action(torch.eye(MDP.n_states)[state][None])
        state, reward, done, info = MDP.step(int(action))
        reward_episode += g * reward
        nb += 1

        g *= MDP.gamma

        # UPDATE
        V.linear_update(s0, state, reward, 1/((count_states[s0]+1)**(0.8)))
    
        if count_states.sum() > 100000:
            break

        mse = torch.norm(V_true-V.get_V())
        mses.append(mse.item())

plt.plot(mses)
plt.show()
V.evaluate()