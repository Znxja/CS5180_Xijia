import numpy as np

class RandomPolicy:

    def __init__(self, T: int):
        self.T = T
        
    def act(self,obs):
        return 1 if np.random.random() < 1/self.T else 0

class ThresholdPolicy:

    def __init__(self, u_min: int): 
        self.u_min = u_min
    
    def act(self, obs):
        return 1 if obs['U_t'] >= self.u_min else 0

class OptimalPolicy:

    def __init__(self, T: int, K: int):
        self.T = T
        self.K = K
        self.threshold = {}
        self.W = {}
        self.V = {}
    
        w_next = 0
        for t in range(T,0,-1):
            self.threshold[t] = w_next
            total_v = 0
            for u in range(1, K+1):
                v = max(w_next, u)
                self.V[(t,u)] = v
                total_v += v
            self.W[t] = total_v/K
            w_next = self.W[t]

    def act(self, obs):
        t = obs['t']
        u = obs['U_t']
        return 1 if u >= self.threshold[t] else 0
    
    def get_policy_table(self):
        rows = []
        for t in range(1, self.T+1):
            row = {'t': t, 'W_t+1':round(self.threshold[t],4)}
            for u in range(1, self.K+1):
                action = 'Accept' if u >= self.threshold[t] else 'Reject'
                row[f'u={u}'] = action
            row['W_t'] = round(self.W[t],4)
            rows.append(row)
        return rows