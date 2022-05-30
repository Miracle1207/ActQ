import numpy as np
import gym
from gym import spaces

class discrete_mujoco_wrapper(gym.Wrapper):
    def __init__(self, env, k, act_type):
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.k = k
        self.act_type = act_type

    def action_space(self):
        if self.act_type == 'uniform' or self.act_type == 'random' or self.act_type == 'ib' or self.act_type == 'actq' or self.act_type == 'pg':
            self.act_size = np.zeros(shape=(self.action_dim, 2))
            discrete_action_space = spaces.MultiDiscrete(self.action_dim*[self.k])
            return discrete_action_space
        else:
            return self.env.action_space

    # def step(self,action):
    #     actions = np.zeros(self.action_dim)
    #     if self.act_type == 'uniform':
    #         for a_i in range(self.action_dim):
    #             actions[a_i] = self.act_size[a_i][0] + (self.act_size[a_i][1]-self.act_size[a_i][0])/(self.k-1)*action[a_i]
    #         return self.env.step(actions)








