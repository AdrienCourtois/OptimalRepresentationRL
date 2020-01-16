import torch
import numpy as np
import sys
from utils.evaluate_policy import evaluate_policy

def make_adjacency(layout):
    directions = [np.array((0, -1)),  # LEFT
                  np.array((0, 1)),  # RIGHT
                  np.array((-1, 0)),  # UP
                  np.array((1, 0))]  # DOWN

    grid = np.array([list(map(lambda c: 0 if c == 'w' else 1, line))
                     for line in layout.splitlines()])
    state_to_grid_cell = np.argwhere(grid)
    grid_cell_to_state = {tuple(state_to_grid_cell[s].tolist()): s
                          for s in range(state_to_grid_cell.shape[0])}

    nstates = state_to_grid_cell.shape[0]
    nactions = len(directions)

    P = np.zeros((nstates, nactions, nstates))

    for state, idx in enumerate(state_to_grid_cell):
        for action, d in enumerate(directions):
            if grid[tuple(idx + d)]:
                dest_state = grid_cell_to_state[tuple(idx + d)]
                P[state, action, dest_state] = 1.
            else:
                P[state, action, state] = 1.

    return P, state_to_grid_cell


class MDPFourRoom:
    def __init__(self):

        # 000 001 002 003 004 xxx 005 006 007 008 009
        # 010 011 012 013 014 xxx 015 016 017 018 019
        # 020 021 022 023 024 025 026 027 028 029 030
        # 031 032 033 034 035 xxx 036 037 038 039 040
        # 041 042 043 044 045 xxx 046 047 048 049 050
        # xxx 051 xxx xxx xxx xxx 052 053 054 055 056
        # 057 058 059 060 061 xxx xxx xxx 062 xxx xxx
        # 063 064 065 066 067 xxx 068 069 070 071 072
        # 073 074 075 076 077 xxx 078 079 080 081 082
        # 083 084 085 086 087 088 089 090 091 092 093
        # 094 095 096 097 098 xxx 099 100 101 102 103

        self.desc = np.array([
            [" ", " ", " ", " ", " ", "x", " ", " ", " ", " ", "G"],
            [" ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " "],
            ["x", " ", "x", "x", "x", "x", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", "x", "x", "x", " ", "x", "x"],
            [" ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
            ["S", " ", " ", " ", " ", "x", " ", " ", " ", " ", " "]])
        
        self.layout = """wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        
        self.n_rows = 11
        self.n_cols = 10
        self.begin_state = 94
        self.end_state = 9

        self.n_states = 104
        self.n_actions = 4 # 0: left, 1: right, 2: top, 3: bottom
        self.gamma = 0.99

        self.P, _ = make_adjacency(self.layout)

        # End
        self.P[self.end_state] = 0
        self.P[self.end_state,:,self.end_state] = 1
        
        # R definition
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R[8,1,:] = 1 # qd tu vas à droite à partir de la case 8, c'est bien
        self.R[19,2,:] = 1 # qd tu vas en haut à partir de la case 19 c'est bien
        
        self.P = torch.from_numpy(self.P).float()
        
        self.r = torch.from_numpy(self.R.mean(axis=0).mean(axis=0))
        self.r = self.r.view(-1, 1).float()


        # transition
        self.state = self.begin_state
        
        self.is_cuda = False
    
    def step(self, action):
        if action != 0 and action != 1 and action != 2 and action != 3:
            raise "Error wrong action, must be in [|0,3|]"
        
        # Compute the next state
        p = self.P[self.state, action]
        next_state = p.argmax()

        # Compute the reward
        reward = self.R[self.state, action, next_state]

        # Checking if it is done
        done = next_state == self.end_state

        # Internal updates
        self.state = next_state.item()
        self.nb_actions += 1

        if done:
            self.reset()
        
        return next_state, reward, done, {}
    
    def reset(self, s=None):
        if s is None:
            self.state = self.begin_state
        else:
            self.state = s
        
        self.nb_actions = 0
        
        return self.state
    
    def render(self):
        outfile = sys.stdout

        out = self.desc.copy()
        out_test = np.zeros(out.shape)
        out_test[out != "x"] = np.arange(out[out != "x"].size)
        out[(out != "x") & (out_test == self.state)] = "H"

        outfile.write("---------\n")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        outfile.write("---------\n")
    
    def cuda(self):
        self.is_cuda = True

        self.P = self.P.cuda()
        self.r = self.r.cuda()

        return self
    
    def evaluate(self, pi, max_iter=500):
        rewards = []
        nb_errors = 0

        for i in range(self.n_states):
            error, reward = evaluate_policy(pi, self, state_start=i, max_iter=max_iter)

            nb_errors += error
            rewards.append(reward)
        
        return nb_errors, rewards
    
    def evaluate_start(self, pi, max_iter=500):
        error, reward = evaluate_policy(pi, self, max_iter=max_iter)

        return error, reward

    def sample(self, n):
        states = torch.zeros(n, dtype=torch.long)
        rewards = torch.zeros(n, self.n_actions, dtype=torch.float)
        next_states = torch.zeros(n, self.n_actions, dtype=torch.long)

        for i in range(n):
            # Sample states
            rd = np.random.randint(0, self.n_states)

            states[i] = rd

            # Sample reward, next_state
            for a in range(self.n_actions):
                self.reset(s=rd)
                next_state, reward, _, _ = self.step(a)
            
                rewards[i][a] = reward
                next_states[i][a] = next_state
            
        return states, rewards, next_states