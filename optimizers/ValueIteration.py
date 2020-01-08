r = torch.from_numpy(MDP.R).mean(2)
V = torch.rand(MDP.n_states)
gamma = MDP.gamma

# Compute optimal V
for k in range(10000):
    q = torch.zeros(MDP.n_states, MDP.n_actions)

    for a in range(MDP.n_actions):
        q[:,a] = r[:,a] + gamma * (MDP.P[:,a,:] * V[None]).sum(1)
    
    V = q.max(1).values

# Compute optimal pi
pi = (r + gamma * (MDP.P * V[None,None]).sum(2)).argmax(1)

# Evaluate
observation = MDP.reset()
reward_episode = 0
nb = 0
done = False
    
while not done:
    action = pi[observation]
    observation, reward, done, info = MDP.step(int(action))
    observation = observation
    reward_episode += reward

    MDP.render()

    nb += 1

print(f'Reward: {reward_episode} : {nb}')

# Save
V_true = V
print(V_true)