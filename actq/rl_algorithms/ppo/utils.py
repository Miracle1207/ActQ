import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
import random
k=5
negative_action_space = np.array([-1, -0.7, -0.33, -0.16, 0])
positive_action_space = np.array([0, 0.21, 0.43, 0.66, 0.9])
random_space = np.array([-1, -0.54, -0.12, 0.55, 0.89])
action_space_1 = np.array([-1, -0.66, -0.4, -0.12, 0.55])
action_space_2 = np.array([-0.4, -0.12, 0.2, 0.55, 1])

def select_actions(pi, dist_type, env_type, action_space=None):
    if env_type == 'atari' or dist_type == 'uniform' or dist_type == 'random' or dist_type == 'ib' or dist_type == 'actq':
        actions = Categorical(pi).sample()
    elif dist_type == "actQ":
        indexs = torch.nn.functional.gumbel_softmax(pi, tau=1, hard=True, dim=2)
        actions = torch.sum(indexs*action_space, dim=2)
        return actions, indexs

    else:
        if dist_type == 'gauss':
            mean, std = pi
            actions = Normal(mean, std).sample()
        elif dist_type == 'beta':
            alpha, beta = pi
            actions = Beta(alpha.detach().cpu(), beta.detach().cpu()).sample()

    # return actions
    return actions.detach().cpu().numpy().squeeze()

def evaluate_actions(pi, actions, dist_type, env_type):
    if env_type == 'atari':
        cate_dist = Categorical(pi)
        log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
        entropy = cate_dist.entropy().mean()
    else:
        if dist_type == 'gauss':
            mean, std = pi
            normal_dist = Normal(mean, std)
            log_prob = normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
            entropy = normal_dist.entropy().mean()
        elif dist_type == 'beta':
            alpha, beta = pi
            beta_dist = Beta(alpha, beta)
            log_prob = beta_dist.log_prob(actions).sum(dim=1, keepdim=True)
            entropy = beta_dist.entropy().mean()
        else:
            cate_dist = Categorical(pi)
            log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
            log_prob = torch.sum(log_prob, dim=1)
            entropy = cate_dist.entropy().mean()
    return log_prob, entropy

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
            new_action[a_i] = action_space_2[actions[a_i]]
        elif dist == 'ib' or dist == 'actq':
            new_action[a_i] = action_space[a_i][actions[a_i]]

    return new_action





