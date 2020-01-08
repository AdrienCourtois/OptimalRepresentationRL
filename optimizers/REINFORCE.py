class ValueFunction(nn.Module):
    def __init__(self, representation, MDP):
        super(ValueFunction, self).__init__()

        self.representation = representation
        self.MDP = MDP

        self.r = torch.from_numpy(self.MDP.R).mean(2)
        self.p = self.MDP.P

        self.theta = Variable(torch.rand(self.representation.d), requires_grad=True)
        self.optim = torch.optim.Adam([self.theta])

        self.alpha = 0.8
    
    def get_V(self):
        Phi = Variable(self.representation(torch.eye(self.MDP.n_states)), requires_grad=False)
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
            
        while not done:
            action = self.select_action(observation)
            observation, reward, done, info = self.MDP.step(int(action))
            observation = observation
            reward_episode += reward
            nb += 1
        
        print(f'Reward: {reward_episode} : {nb}')

import pandas as pd
import seaborn as sns
import itertools

class REINFORCE:
    
    def __init__(self, MDP):
        self.MDP = MDP
        self.model = ValueFunction(representation, MDP)
        self.gamma = self.MDP.gamma
        
        # the optimizer used by PyTorch (Stochastic Gradient, Adagrad, Adam, etc.)
        self.optimizer = self.model.optim
    
    def train(self, n_trajectories, n_update):
        """Training method

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expected gradient
        n_update : int
            The number of gradient updates
            
        """
        
        rewards = []
        for episode in range(n_update):
            rewards.append(self.optimize_model(n_trajectories))
            
            # ALPHA UPDATE #
            #self.model.alpha *= 1+0.44*1e-2

            print(f'Episode {episode + 1}/{n_update}: rewards {round(rewards[-1].mean(), 2)} +/- {round(rewards[-1].std(), 2)}')

            # Early stopping
            if rewards[-1].mean() > 490 and episode != n_update -1:
                print('Early stopping !')
                break
        
        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards[i]) for i in range(len(rewards))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');
        
    def evaluate(self, render=False):
        """Evaluate the agent on a single trajectory            
        """
        
        observation = self.onehot(self.MDP.reset())
        reward_episode = 0
        done = False
            
        while not done:
            action = self.model.select_action(torch.from_numpy(observation[None]).float()).numpy().item()
            observation, reward, done, info = self.MDP.step(int(action))
            observation = self.onehot(observation)
            reward_episode += reward
        
        print(f'Reward: {reward_episode}')

    def _compute_returns(self, rewards):
        n = len(rewards)
        gammas = np.power(self.gamma, np.arange(n))
        rewards = np.array(rewards)

        R = np.array([np.sum(gammas[:n-i] * rewards[i:]) for i in range(n)])

        return R
    
    def onehot(self, state):
        return np.eye(self.MDP.n_states)[state]
        
    def optimize_model(self, n_trajectories):
        loss = 0
        final_reward = []

        for i in range(n_trajectories):
            done = False
            state = self.onehot(self.MDP.reset())

            rewards = []
            states = [state]
            actions = []

            # Generation of a trajectory
            while not done:
                action = self.model.select_action(torch.from_numpy(state[None]).float()).numpy().item()
                state, reward, done, info = self.MDP.step(action)

                state = self.onehot(state)

                states.append(state)
                rewards.append(reward)
                actions.append(action)

            # Calculate the loss term
            states = torch.from_numpy(np.array(states)).float()
            returns = torch.from_numpy(self._compute_returns(rewards)).float()

            log_policy = torch.log(self.model(states))
            
            for i in range(len(actions)):
                loss += -log_policy[i][actions[i]] * returns[i]
            
            # Storage of the total reward
            final_reward.append(returns[0].numpy())
        
        # Optimization step
        loss /= n_trajectories

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return np.array(final_reward)

agent = REINFORCE(MDP)
agent.train(n_trajectories=50, n_update=200)
agent.evaluate()