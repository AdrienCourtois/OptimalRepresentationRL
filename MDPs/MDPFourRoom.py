import torch
import numpy as np
import sys
from utils.evaluate_V import evaluate_V

def make_adjacency(layout):
    """Convert a grid world layout to an adjacency matrix.
    Args:
        layout (np.ndarray): Grid layout as an array where 0 means a wall and 1 is empty.
    Returns:
        tuple: First element is aulti-dimensional np.ndarray of size (A X S X S) where A=4 is the 
        number of actions, and S is the number of states. The action set is: 
        UP (0), DOWN (1), LEFT (2), RIGHT (3). The second element of the tuple is a np.ndarray
        mapping state (integer) to cell coordinates in the original layout.
    """
    directions = [np.array((-1, 0)),  # UP
                  np.array((1, 0)),  # DOWN
                  np.array((0, -1)),  # LEFT
                  np.array((0, 1))]  # RIGHT

    grid = np.array([list(map(lambda c: 0 if c == 'w' else 1, line)) for line in layout.splitlines()])
    state_to_grid_cell = np.argwhere(grid)
    grid_cell_to_state = {tuple(state_to_grid_cell[s].tolist()): s
                          for s in range(state_to_grid_cell.shape[0])}

    nstates = state_to_grid_cell.shape[0]
    nactions = len(directions)
    P = np.zeros((nactions, nstates, nstates))
    for state, idx in enumerate(state_to_grid_cell):
        for action, d in enumerate(directions):
            if grid[tuple(idx + d)]:
                dest_state = grid_cell_to_state[tuple(idx + d)]
                P[action, state, dest_state] = 1.

    return P, state_to_grid_cell

class MDPFourRoom:
    def __init__(self):
        # (000) (001) (002) (003) (004) (005) (006) (007) (008) (009) (010) (011) (012)
        # (013)  014   015   016   017   018  (019)  020   021   022   023   024  (025)
        # (026)  027   028   029   030   031  (032)  033   034   035   036   037  (038)
        # (039)  040   041   042   043   044   045   046   047   048   049   050  (051)
        # (052)  053   054   055   056   057  (058)  059   060   061   062   063  (064)
        # (065)  066   067   068   069   070  (071)  072   073   074   075   076  (077)
        # (078) (079)  080  (081) (082) (083) (084)  085   086   087   088   089  (090)
        # (091)  092   093   094   095   096  (097) (098) (099)  100  (101) (102) (103)
        # (104)  105   106   107   108   109  (110)  111   112   113   114   115  (116)
        # (117)  118   119   120   121   122  (123)  124   125   126   127   128  (129)
        # (130)  131   132   133   134   135   136   137   138   139   140   141  (142)
        # (143)  144   145   146   147   148  (149)  150   151   152   153   154  (155)
        # (156) (157) (158) (159) (160) (161) (162) (163) (164) (165) (166) (167) (168)

        self.desc = np.array([
            ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x"],
            ["x", " ", " ", " ", " ", " ", "x", " ", " ", " ", " ", "G", "x"],
            ["x", " ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " ", "x"],
            ["x", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "x"],
            ["x", " ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " ", "x"],
            ["x", " ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " ", "x"],
            ["x", "x", " ", "x", "x", "x", "x", " ", " ", " ", " ", " ", "x"],
            ["x", " ", " ", " ", " ", " ", "x", "x", "x", " ", "x", "x", "x"],
            ["x", " ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " ", "x"],
            ["x", " ", " ", " ", " ", " ", "x", " ", " ", " ", " ", " ", "x"],
            ["x", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "x"],
            ["x", "S", " ", " ", " ", " ", "x", " ", " ", " ", " ", " ", "x"],
            ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x"]])

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
        self.n_rows = 13
        self.n_cols = 13
        self.begin_state = 144
        self.end_state = 24
        self.forbidden_states = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 19, 25,
            26, 32, 38,
            39, 51,
            52, 58, 64,
            65, 71, 77,
            78, 79, 81, 82, 83, 84, 90,
            91, 97, 98, 99, 101, 102, 103,
            104, 110, 116,
            117, 123, 129,
            130, 142,
            143, 149, 155,
            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]

        self.n_states = 169
        self.n_actions = 4 # 0: left, 1: right, 2: top, 3: bottom
        self.gamma = 0.99

        # P definition
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))

        for s in range(self.n_states):
            self.P[s,0,(s-1) % self.n_states] = 1
            self.P[s,1,(s+1) % self.n_states] = 1
            self.P[s,2,(s-self.n_cols) % self.n_states] = 1
            self.P[s,3,(s+self.n_cols) % self.n_states] = 1
        
        # End
        self.P[self.end_state] = np.zeros((self.n_actions, self.n_states))
        self.P[self.end_state,:,self.end_state] = 1

        # In a wall probability -> stay in place
        for x in range(self.n_states):
            for a in range(self.n_actions):
                if self.P[x,a].argmax() in self.forbidden_states:
                    self.P[x,a,self.P[x,a].argmax()] = 0
                    self.P[x,a,x] = 1
        
        # R definition
        self.R[23,1,:] = 1 # qd tu vas à droite à partir de la case 23, c'est bien
        self.R[37,2,:] = 1 # qd tu vas en haut à partir de la case 37 c'est bien
        
        self.P = torch.from_numpy(self.P).float()
        
        self.r = torch.from_numpy(self.R.mean(axis=0).mean(axis=0))
        self.r = self.r.view(-1, 1).float()


        # transition
        self.state = self.begin_state
        self.nb_actions = 0
        
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
        done = next_state == self.end_state or next_state in self.forbidden_states

        # Internal updates
        self.state = next_state
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

        out = self.desc.copy().tolist()
        out = [[c for c in line] for line in out]
        r, c = int(self.state / self.n_cols), int(self.state % self.n_cols)

        def ul(x):
            return "_" if x == " " else x
        
        out[r][c] = "H"

        outfile.write("---------\n")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        outfile.write("---------\n")
    
    def cuda(self):
        self.is_cuda = True

        self.P = self.P.cuda()
        self.r = self.r.cuda()

        return self
    
    def evaluate(self, V):
        rewards = []
        nb_errors = 0

        for i in range(self.n_states):
            if i in self.forbidden_states:
                continue
            
            error, reward = evaluate_V(V, self, state_start=i)

            nb_errors += error
            rewards.append(reward)
        
        return nb_errors, rewards

    def sample(self, n):
        states = torch.zeros(n, dtype=torch.int)
        rewards = torch.zeros(n, self.n_actions, dtype=torch.float)
        next_states = torch.zeros(n, self.n_actions, dtype=torch.int)

        for i in range(n):
            # Sample states
            rd = np.random.randint(0, self.n_states)

            while rd in self.forbidden_states:
                rd = np.random.randint(0, self.n_states)

            states[i] = rd

            # Sample reward, next_state
            self.reset(s=rd)

            for a in range(self.n_actions):
                next_state, reward, _, _ = self.step(a)
            
                rewards[i][a] = reward
                next_states[i][a] = next_state
            
        return states, rewards, next_states