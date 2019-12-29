import torch
import numpy as np
import pickle

class AVFManager:
    def __init__(self, MDP):
        self.MDP = MDP
        self.AVFs = []
        self.deltas = torch.from_numpy(np.array([])).float()
    

    ###############
    #    Utils    #
    ###############
    def onehot_state(self, x):
        s = torch.zeros(self.MDP.n_states)
        s[x] = 1

        return s

    def compute_P(self, pi):
        # pi can be either deterministic or probabilistic
        # deterministic: pi must be in the format X -> A
        # probabilistic: pi must be in the format X -> P(A)

        P = torch.zeros(self.MDP.n_states, self.MDP.n_states)

        if len(pi.size()) == 1: # deterministic
            for x in range(self.MDP.n_states):
                P[:, x] = self.MDP.P[x, pi[x], :]
        else:
            for x in range(self.MDP.n_states):
                P[x] = torch.sum(pi * self.MDP.P[:,:,x], 1)
        
        return P

    def compute_V(self, pi):
        # pi can be either deterministic or probabilistic
        # deterministic: pi must be in the format X -> A
        # probabilistic: pi must be in the format X -> P(A)

        P = self.compute_P(pi)
        mat = torch.eye(self.MDP.n_states) - self.MDP.gamma * P
        mat = mat.inverse()
        V = torch.mm(mat, self.MDP.r)

        return V

    def scoreAVF(self, delta, pi):
        # pi can be either deterministic or probabilistic
        # deterministic: pi must be in the format X -> A
        # probabilistic: pi must be in the format X -> P(A)

        V = self.compute_V(pi)
        score = torch.mm(delta.t(), V)

        return score.item()


    ###############
    # ALGORITHM 1 #
    ###############

    def GradientBasedAVF(self, delta, niter=1000):
        # Implementation of Algorithm 1 to compute an AVF given a direction delta
        # using a gradient-based policy-learning algorithm.
        # Around 1s on CPU for default params.
        # delta: Direction in R^n where n=n_states
        # Returns a deterministic policy

        pi = AVFPolicy(self.MDP)

        for i in range(niter):
            # Compute P
            P = self.compute_P(pi.get_policy())
            
            mat = torch.eye(self.MDP.n_states) - self.MDP.gamma * P
            mat = mat.inverse()

            # Compute loss
            loss = -torch.mm(torch.mm(delta.t(), mat), self.MDP.r)

            # Optimization
            loss.backward()
            pi.optimizer.step()
            pi.optimizer.zero_grad()
        
        return pi.get_policy().max(1).indices


    ###############
    # ALGORITHM 2 #
    ###############

    def PolicyBasedAVF(self, delta, niter=1000, piter=100):
        # Policy based algorithm used to compute the AVF.
        # It is really slow in practise and does not always find a good result.
        # delta: Direction in R^n where n=n_states
        # Returns a deterministic policy

        pi = torch.from_numpy(np.random.randint(0, self.MDP.n_actions, self.MDP.n_states))
        d = torch.rand(self.MDP.n_states, 1)
        V = torch.zeros(self.MDP.n_states)
        Q = torch.rand(self.MDP.n_states, self.MDP.n_actions)

        for i in range(niter):
            # Compute P
            P = self.compute_P(pi)
            
            # Fixed point iteration on d
            d_before = d.clone()
            d = delta + self.MDP.gamma * torch.mm(P.t(), d)
            p_iter_count = 0

            while torch.norm(d_before - d) > 0 and p_iter_count < piter:
                d_before = d.clone()
                d = delta + self.MDP.gamma * torch.mm(P.t(), d)
                p_iter_count += 1
            
            # Computation of V
            for j in range(piter):
                for x in range(self.MDP.n_states):
                    if d[x] > 0:
                        V[x] = self.MDP.r[x] + self.MDP.gamma * torch.mm(P, Q).max(1).values[x]
                    else:
                        V[x] = self.MDP.r[x] + self.MDP.gamma * torch.mm(P, Q).min(1).values[x]
            
                # Computation of Q
                for x in range(self.MDP.n_states):
                    for a in range(self.MDP.n_actions):
                        Q[x, a] = self.MDP.r[x] + self.MDP.gamma * torch.sum(self.MDP.P[x, a, :] * V)

            # Computation of pi
            old_pi = pi.clone()
            for x in range(self.MDP.n_states):
                if d[x] > 0:
                    pi[x] = Q[x].argmax()
                else:
                    pi[x] = Q[x].argmin()
            
        return pi


    ###############
    # ALGORITHM 3 #
    ###############

    def NaiveAVF(self, delta, verbose=False):
        # WARNING: This algorithm is extremely computationaly expensive
        # ONLY use it for very little MDP
        # delta: Direction in R_n where n=n_states
        # Returns the best deterministic policy

        nb_policy = self.MDP.n_actions ** self.MDP.n_states

        if nb_policy > 2000000:
            raise "Too long to compute"
        
        best_policy = torch.zeros(self.MDP.n_states, dtype=torch.int)
        best_score = self.scoreAVF(delta, best_policy)
        
        for k in range(nb_policy):
            # Retrieve the policy
            policy = torch.zeros(self.MDP.n_states, dtype=torch.int)
            n = k

            for i in range(self.MDP.n_states):
                policy[i] = n % self.MDP.n_actions
                n = int(n / self.MDP.n_actions)
            
            # Compute the score
            score = self.scoreAVF(delta, policy)

            # Update the score
            if score > best_score:
                best_policy = policy.clone()
                best_score = score
        
        if k % 100000 == 0 and k > 0 and verbose:
            print(k, "iterations done,", nb_policy-k, "left")
    
        return best_policy


    ################
    # Compute AVFs #
    ################

    def compute(self, k, method=0, **kargs):
        # Method can be "gradient" (=0), "policy" (=1) or "naive" (=2)

        # Selection of the algorithm
        algo = None

        if method == "gradient" or method == 0:
            algo = self.GradientBasedAVF
        elif method == "policy" or method == 1:
            algo = self.PolicyBasedAVF
        elif method == "naive" or method == 2:
            algo = self.NaiveAVF
        else:
            raise "Method can be \"gradient\" (=0), \"policy\" (=1) or \"naive\" (=2)"
        
        
        n = len(self.deltas)
        self.deltas = torch.cat((self.deltas, torch.rand(k, self.MDP.n_states, 1, dtype=torch.float)))
        
        for i in range(n,n+k):
            self.AVFs.append(algo(self.deltas[i], **kargs))
    
    def __getitem__(self, idx):
        # Returns the asked AVF

        if idx >= len(self.AVFs):
            raise "Index error, tried to access element idx " + str(idx) + "/" + str(len(self.AVFs))

        return self.AVFs[idx]
    
    def score(self, idx):
        # Returns the score of the AVF number `idx`

        return self.scoreAVF(self.deltas[idx], self.AVFs[idx])
    
    #######
    # I/O #
    #######

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)