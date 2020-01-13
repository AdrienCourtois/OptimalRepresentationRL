import torch

def evaluate_policy(pi, MDP, state_start=None, max_iter=2000, random=False, render=False):
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