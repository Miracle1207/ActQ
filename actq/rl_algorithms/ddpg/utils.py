import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical

# add ounoise here
class ounoise():
    def __init__(self, std, action_dim, mean=0, theta=0.15, dt=1e-2, x0=None):
        self.std = std
        self.mean = mean
        self.action_dim = action_dim
        self.theta = theta
        self.dt = dt
        self.x0 = x0
    
    # reset the noise
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.action_dim)
    
    # generate noise
    def noise(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
                self.std * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.x_prev = x
        return x

def select_actions(pi, action_space=0):
    actions = Categorical(pi).sample()
    return actions.detach().cpu().numpy().squeeze()
    # indexs = torch.nn.functional.gumbel_softmax(pi, tau=1, hard=True, dim=2)
    # actions = torch.sum(indexs*action_space, dim=2)
    # return actions, indexs


def evaluate_actions(pi, actions):
    cate_dist = Categorical(pi)
    log_prob = cate_dist.log_prob(torch.Tensor(actions)).unsqueeze(-1)
    log_prob = torch.sum(log_prob, dim=1)
    entropy = cate_dist.entropy().mean()
    return log_prob

def step_wrapper(actions, env, action_space):
    action_dim = env.env.action_space.shape[0]
    new_action = np.zeros(action_dim)
    # todo 需要对B进行一个规范化到 action space 范围中
    for a_i in range(action_dim):
        new_action[a_i] = action_space[a_i][actions[a_i]]

    return new_action

def buffer_step_wrapper(actions, action_space):
    acions_one_hot = torch.nn.functional.one_hot(torch.tensor(actions), num_classes=action_space.shape[1])
    repeat_action_space = action_space.unsqueeze(0).repeat(len(actions),1,1)
    try:
        new_action = torch.sum(acions_one_hot * repeat_action_space, dim=2)
    except:
        print(acions_one_hot, repeat_action_space)
    return new_action