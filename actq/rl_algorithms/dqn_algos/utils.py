import numpy as np
import random

random_action_space = np.array([[0],
                                [0],
                                [0]])
# linear exploration schedule
class linear_schedule:
    def __init__(self, total_timesteps, final_ratio, init_ratio=1.0):
        self.total_timesteps = total_timesteps
        self.final_ratio = final_ratio
        self.init_ratio = init_ratio

    def get_value(self, timestep):
        frac = min(float(timestep) / self.total_timesteps, 1.0)
        return self.init_ratio - frac * (self.init_ratio - self.final_ratio)

# select actions
def select_actions(action_value, explore_eps):
    action_value = action_value.cpu().numpy().squeeze()
    # select actions
    action = np.argmax(action_value,axis=1) if random.random() > explore_eps else np.random.randint(action_value.shape[1], size=action_value.shape[0])
    return action

def step_wrapper(actions, env, dist, k=0, action_space=0):
    true_env = env.env.env
    action_dim = env.env.action_space.shape[0]
    new_action = np.zeros(action_dim)
    # todo 需要对B进行一个规范化到 action space 范围中
    for a_i in range(action_dim):
        if dist == "uniform":
            interval = (true_env.action_space.high[a_i] - true_env.action_space.low[a_i]) / (k - 1)
            new_action[a_i] = true_env.action_space.low[a_i] + interval * actions[a_i]
        elif dist == "random":
            new_action[a_i] = random_action_space[actions[a_i]]
        elif dist == 'actq':
            new_action[a_i] = action_space[a_i][actions[a_i]]

    return new_action
# record the reward info of the dqn experiments
class reward_recorder:
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards 
        self.buffer = [0.0]
        self._episode_length = 1
    
    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        if self.get_length >= self.history_length:
            self.buffer.pop(0)
        # append new one
        self.buffer.append(0.0)
        self._episode_length += 1

    # get length of buffer
    @property
    def get_length(self):
        return len(self.buffer)
    
    @property
    def mean(self):
        return np.mean(self.buffer)
    
    # get the length of total episodes
    @property 
    def num_episodes(self):
        return self._episode_length
