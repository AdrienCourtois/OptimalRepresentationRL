import torch
import numpy as np
import sys

class MDPTwoRoom:
    def __init__(self):
        # 0  1  2  3  x  5  6  7  8
        # 9  10 11 12 x  14 15 16 17
        # 18 19 20 21 22 23 24 25 26
        # 27 28 29 30 x  32 33 34 35  
        # 36 37 38 39 x  41 42 43 44 (45) 

        self.desc = np.array([[" ", " ", " ", " ", "x", " ", " ", " ", " "],
                             [" ", " ", " ", " ", "x", " ", " ", " ", " "],
                             [" ", " ", " ", " ", " ", " ", " ", " ", " "],
                             [" ", " ", " ", " ", "x", " ", " ", " ", " "],
                             ["S", " ", " ", " ", "x", " ", " ", " ", "G"]])

        self.n_rows = 5
        self.n_cols = 9
        self.begin_state = 36
        self.end_state = 44
        self.dump_state = 45
        self.forbidden_states = [4, 13, 31, 40]

        self.n_states = 46
        self.n_actions = 4 # 0: left, 1: right, 2: top, 3: bottom
        self.gamma = 0.99

        # P definition
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))

        for s in range(self.n_states):
            self.P[s,0,(s-1) % self.n_states] = 1
            self.P[s,1,(s+1) % self.n_states] = 1
            self.P[s,2,(s-self.n_cols) % self.n_states] = 1
            self.P[s,3,(s+self.n_cols) % self.n_states] = 1
        
        for x in [0, 9, 18, 27, 36]:
            self.P[x,0,:] = 0
            self.P[x,0,self.dump_state] = 1
        
        for x in [8, 17, 26, 35, 44]:
            self.P[x,1,:] = 0
            self.P[x,1,self.dump_state] = 1

        for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            self.P[x,2,:] = 0
            self.P[x,2,self.dump_state] = 1
        
        for x in [36, 37, 38, 39, 40, 41, 42, 43, 44]:
            self.P[x,3,:] = 0
            self.P[x,3,self.dump_state] = 1
        
        self.P = torch.from_numpy(self.P).float()
        
        # R definition
        self.R = np.zeros(self.P.shape)
        self.R[43,1,:] = 1 # qd tu vas à droite à partir de la case 43, c'est bien
        self.R[35,3,:] = 1 # qd tu vas en bas à partir de la case 35 c'est bien
        #self.R[:,:,22] = 0.5 # not good, loops around the door
        
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
        done = next_state == self.end_state or next_state in self.forbidden_states or next_state == self.dump_state or self.nb_actions >= 200

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