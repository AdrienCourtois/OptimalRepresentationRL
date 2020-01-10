import torch

def evaluate_V(V, MDP, state_start=None, max_iter=2000, random=False, render=False):
    r = torch.from_numpy(MDP.R).mean(2)

    # Compute policy
    Q = (r + MDP.gamma * (MDP.P * V[None,None]).sum(2))
    pi = Q.argmax(1) if not random else torch.softmax(Q, 1)

    error = 0

    observation = MDP.reset(state_start)
    reward_episode = 0
    nb = 0
    done = False
    g = 1
        
    while not done:
        action = pi[observation] if not random else torch.multinomial(pi[observation], 1)
        observation, reward, done, _ = MDP.step(int(action))
        observation = observation
        reward_episode += g * reward
        g *= MDP.gamma

        nb += 1

        if nb >= max_iter:
            error = 1
            break
        
        if render:
            MDP.render()

    return error, reward_episode