import gymnasium as gym
from gymnasium.spaces import Discrete, Dict
import numpy as np

class ApartmentEnv(gym.Env):
    def __init__(self, T: int, K: int, seed=None, noise_std: float = 0.0):
        super().__init__()
        self.T = T
        self.K = K
        self.noise_std = noise_std
        self.action_space = Discrete(2)
        self.observation_space = Dict({'t':Discrete(T+1), 
                                           'U_t':Discrete(K+1)})
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        self.t = 1
        self.U_t = int(self.np_random.randint(1, self.K+1))
        self.done = False
        obs = self.get_obs()
        info = {'true U_t': self.U_t}
        return obs, info
    
    def step(self, action: int):
        assert not self.done, 'episode is done, please call reset()'
        assert action in (0,1), 'invalid action, either 0 or 1'
        terminated = False
        truncated  = False
        reward = 0
        info = {'true U_t': self.U_t}
        if action == 1:
            reward = self.U_t
            terminated = True
            self.done = True
        else:
            if self.t >= self.T:
                reward = 0
                terminated = True
                self.done = True
            else:
                self.t += 1
                self.U_t = int(self.np_random.randint(1,self.K+1))
                reward = 0
        obs = self.get_obs()
        return obs, reward, terminated, truncated, info
    

    def get_obs(self):
        if self.noise_std > 0:
            noisy_U = self.U_t + self.np_random.normal(0, self.noise_std)
            return {'t': self.t, 'U_t':float(noisy_U)}
        else:
            return {'t':self.t, 'U_t':self.U_t}
    
    def __str__(self):
        return f'ApartmentEnv(T={self.T}, K={self.K}, noise_std={self.noise_std})'