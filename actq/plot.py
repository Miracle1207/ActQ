import os
# import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 50
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

FONT_SIZE=13

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    # shape = a.shape[:-1] + (a.shape[-1], window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def plot_curves(x_list, y_lists, xaxis, title, label, color):
    # plt.figure(figsize=(8,2))
    maxx = x_list[-1]
    minx = 0
    x = x_list
    y_mean = np.mean(y_lists, axis=0)
    y_std_error = np.std(y_lists, axis=0) / np.sqrt(y_lists.shape[0])
    y_upper = y_mean + y_std_error
    y_lower = y_mean - y_std_error

    # for (i, (x, y)) in enumerate(xy_list):
    #     color = COLORS[i]
    #     plt.scatter(x, y, s=2)
    x_filter, y_mean = window_func(x, y_mean, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    _, y_upper = window_func(x, y_upper, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    _, y_lower = window_func(x, y_lower, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes

    plt.plot(x_filter, y_mean, label=label, color=color)

    plt.fill_between(x_filter, y_lower, y_upper, alpha=0.15, color=color)
    # minx = 10000
    #plt.xlim(minx, maxx)
    #plt.ylim(-210, -120)
    plt.title(title, fontsize=FONT_SIZE)
    plt.xlabel(xaxis, fontsize=FONT_SIZE)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.grid(True)
    plt.ylabel("Episode Rewards", fontsize=FONT_SIZE)
    plt.legend()
    # 横坐标指数表示
    # plt.xscale('symlog')
    plt.tight_layout()

# read csv
# path = "/home/mqr/TempoRL-master/experiments/featurized_results/sparsemountain/tdqn/"
# files = os.listdir(path)
# # 测出sample个数
# rew_1 = pd.read_csv(path + files[1])
# x_val = rew_1['Step'].values

# read txt
# path = "/home/mqr/TempoRL-master/experiments/featurized_results/sparsemountain/tdqn/"
# files = os.listdir(path)
# rew_1 = np.loadtxt(path+files[0]+"/reward.txt")
# x_val = np.loadtxt(path+files[0]+"/step.txt")
# len_x = len(x_val)
# # number of files
# len_file = len(files)
# rew = np.array(np.zeros(shape = (len_file, len_x)))
# for file_i in range(len_file):
#     rew[file_i] = pd.read_csv(path+files[file_i])['Value'].values

'''------------------------n seeds plot----------------------------------------------------------------'''
# todo main
# 多个seed放一起:
# rew_10 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run38/reward.txt")
# step_20 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run38/reward.txt")
# rew_20 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run38/reward.txt")
#
# rews_uni0 = np.vstack((rew_10,rew_20))
#
# # ppo-uniform
# rew_1 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/gmm/run2/k=5-reward.txt")
# step_2 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/gmm/run2/k=5-steps.txt")
#
# rew_2 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/gmm/run4/k=5-reward.txt")
#
# rews_uni = np.vstack((rew_1,rew_2))
#
# # ppo-continuous
# rew_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run39/reward.txt")
# step_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run39/steps.txt")
#
# rew_4 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run38/reward.txt")
#
# rews_con = np.vstack((rew_3,rew_4))
#
# # sac-gmm
# gmm_rew3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run5/reward.txt")
# gmm_step = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run5/steps.txt")
#
# gmm_rew4 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run6/reward.txt")
# gmm_rew = np.vstack((gmm_rew3, gmm_rew4))
# #
''''''
# ppo-continuous
sac_rew1 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run75/reward.txt")
sac_steps = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run75/steps.txt")

sac_rew2 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run84/reward.txt")
sac_rew3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run85/reward.txt")
# sac_rew4 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/data/run1/reward.txt")
sac_rews = np.vstack((sac_rew1,sac_rew2, sac_rew3))
plot_curves(x_list=sac_steps, y_lists=sac_rews, xaxis="train steps", title="Hopper-v2 ActQ-PPO", label="continuous",color=COLORS[4])

# # uniform
sac_rew10 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run86/reward.txt")
sac_steps0 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run86/steps.txt")

sac_rew20 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run87/reward.txt")
sac_rew30 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run88/reward.txt")
sac_rews0 = np.vstack((sac_rew10,sac_rew20, sac_rew30))
plot_curves(x_list=sac_steps0, y_lists=sac_rews0, xaxis="train steps", title="Hopper-v2 ActQ-PPO", label="uniform-k=5",color=COLORS[0])

# # sac_rew10 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run65/reward.txt")
# # sac_steps0 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run65/steps.txt")
# #
# # sac_rew20 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run66/reward.txt")
# # sac_rew30 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run68/reward.txt")
# # sac_rew40 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run73/reward.txt")
# # sac_rews0 = np.vstack((sac_rew10,sac_rew20, sac_rew30, sac_rew40))
# # plot_curves(x_list=sac_steps0, y_lists=sac_rews0, xaxis="train steps", title="Hopper-v2 ActQ-PPO", label="uniform-k=10",color=COLORS[10])
#
# # sac_rew10 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run70/reward.txt")
# # sac_steps0 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run70/steps.txt")
# #
# # sac_rew20 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run69/reward.txt")
# # sac_rew30 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run71/reward.txt")
# # sac_rew40 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run74/reward.txt")
# # sac_rews0 = np.vstack((sac_rew10,sac_rew20, sac_rew30, sac_rew40))
# # plot_curves(x_list=sac_steps0, y_lists=sac_rews0, xaxis="train steps", title="Hopper-v2 ActQ-PPO", label="uniform-k=20",color=COLORS[2])
#
# # # train
sac_rew11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run45/reward.txt")
sac_steps1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run45/steps.txt")

sac_rew21 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run46/reward.txt")
sac_rew22 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run47/reward.txt")
sac_rews1 = np.vstack((sac_rew11, sac_rew21, sac_rew22))
plot_curves(x_list=sac_steps1, y_lists=sac_rews1, xaxis="train steps", title="Hopper-v2 ActQ-PPO", label="actq-k=5",color=COLORS[1])

# sac_rew11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run44/reward.txt")
# sac_steps1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run44/steps.txt")
#
# sac_rew21 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run49/reward.txt")
# sac_rew22 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run50/reward.txt")
# sac_rews1 = np.vstack((sac_rew11, sac_rew21, sac_rew22))
# plot_curves(x_list=sac_steps1, y_lists=sac_rews1, xaxis="train steps", title="Hopper-v2 ActQ-PPO", label="actq-k=5 lr=1e-4",color=COLORS[2])
''''''
# sac_rew11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run31/reward.txt")
# sac_steps1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run31/steps.txt")
#

# sac_rew21 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run32/reward.txt")
# sac_rew22 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run33/reward.txt")
# sac_rew23 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run40/reward.txt")
# sac_rews1 = np.vstack((sac_rew11, sac_rew21, sac_rew22, sac_rew23))
# plot_curves(x_list=sac_steps1, y_lists=sac_rews1, xaxis="train steps", title="Hopper-v2 ActQ-PPO", label="actq-k=10",
#             color=COLORS[3])

# sac_rew11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run35/reward.txt")
# sac_steps1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run35/steps.txt")
#
# sac_rew21 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run36/reward.txt")
# sac_rew22 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run37/reward.txt")
# sac_rew23 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run39/reward.txt")
# sac_rews1 = np.vstack((sac_rew11, sac_rew21, sac_rew22, sac_rew23))
# plot_curves(x_list=sac_steps1, y_lists=sac_rews1, xaxis="train steps", title="Hopper-v2 ActQ-PPO", label="actq-k=20",
#             color=COLORS[7])

# # test
# sac_rew112 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run59/reward.txt")
# sac_steps12 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run59/steps.txt")
#
# sac_rew212 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run60/reward.txt")
# sac_rew213 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run57/reward.txt")
# sac_rews12 = np.vstack((sac_rew112, sac_rew212, sac_rew213))

# # PPO uniform


# sac_rew1122 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run46/reward.txt")
# sac_steps122 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run46/steps.txt")
#
# sac_rew2122 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run6/reward.txt")
# sac_rews122 = np.vstack((sac_rew1122, sac_rew2122))


# # plot_curves(x_list=step_20, y_lists=rews_uni0, xaxis="train steps", title="Hopper-v2 SAC-GMM no-related", label="lr=1e-3", color=COLORS[0])
# plot_curves(x_list=step_3, y_lists=rews_con, xaxis="train steps", title="Hopper-v2 PPO", label="PPO-continuous",color=COLORS[2])
# # plot_curves(x_list=step_2, y_lists=rews_uni, xaxis="train steps", title="Hopper-v2 SAC-GMM no-related", label="lr=1e-4",color=COLORS[1])
# # plot_curves(x_list=gmm_step, y_lists=gmm_rew, xaxis="train steps", title="Hopper-v2 PPO", label="PPO-uniform k=5",color=COLORS[1])
# actq
#
# # plot_curves(x_list=sac_steps11, y_lists=sac_rews11, xaxis="train steps", title="Hopper-v2 PPO", label="actQ k=3",color=COLORS[1])

# plot_curves(x_list=sac_steps12, y_lists=sac_rews12, xaxis="train steps", title="Hopper-v2 ActQ-PPO", label="testing",color=COLORS[1])

# #

'''-----------------------------------single seed plot-----------------------------------------------------'''
# 单个种子:

'''
server-178
'''
# rew_2 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/category/run1/k=5-reward.txt")
# step_2 = np.loadtxt("/home/qirui/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run182/k=10-steps.txt")
#
# rew_2 = np.expand_dims(rew_2, axis=0)
# plot_curves(x_list=step_2, y_lists=rew_2, xaxis="train steps", title="Hopper-v2 SAC-GMM relu-0.002", label="k=10 lr=7e-4", color=COLORS[0])
#
#
# rew_3 = np.loadtxt("/home/qirui/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run170/k=3-reward.txt")
# step_3 = np.loadtxt("/home/qirui/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run170/k=3-steps.txt")
#
# rew_3 = np.expand_dims(rew_3, axis=0)
# plot_curves(x_list=step_3, y_lists=rew_3, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="k=3 lr=5e-4", color=COLORS[1])
#
# rew_4 = np.loadtxt("/home/qirui/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run171/k=5-reward.txt")
# step_4 = np.loadtxt("/home/qirui/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run171/k=5-steps.txt")
#
# rew_4 = np.expand_dims(rew_4, axis=0)
# plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="k=5 lr=5e-4", color=COLORS[2])

# rew_11 = np.loadtxt("/home/qirui/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run174/k=10-reward.txt")
# step_11 = np.loadtxt("/home/qirui/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run174/k=10-steps.txt")
#
# rew_11 = np.expand_dims(rew_11, axis=0)
# plot_curves(x_list=step_11, y_lists=rew_11, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="k=10 lr=5e-4", color=COLORS[3])
#
# rew_1 = np.loadtxt("/home/qirui/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run169/k=20-reward.txt")
# step_1 = np.loadtxt("/home/qirui/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run169/k=20-steps.txt")
#
# rew_1 = np.expand_dims(rew_1, axis=0)
# plot_curves(x_list=step_1, y_lists=rew_1, xaxis="train steps", title="Hopper-v2 SAC-GMM tanh", label="k=20 lr=7e-4", color=COLORS[4])

'''
server-80.25
'''
# # seed 125
# rew_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run75/reward.txt")
# step_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run75/steps.txt")
#
# rew_119 = np.expand_dims(rew_119, axis=0)
# plot_curves(x_list=step_119, y_lists=rew_119, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s125-con lr=3e-4", color=COLORS[1])
#
# rew_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run85/reward.txt")
# step_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run85/steps.txt")
#
# rew_119 = np.expand_dims(rew_119, axis=0)
# plot_curves(x_list=step_119, y_lists=rew_119, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s125-con lr=3e-4", color=COLORS[2])
#
# rew_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run86/reward.txt")
# step_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run86/steps.txt")
#
# rew_119 = np.expand_dims(rew_119, axis=0)
# plot_curves(x_list=step_119, y_lists=rew_119, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s125-con lr=3e-4", color=COLORS[4])

# # rew_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run75/reward.txt")
# # step_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run75/steps.txt")
# #
# # rew_119 = np.expand_dims(rew_119, axis=0)
# # plot_curves(x_list=step_119, y_lists=rew_119, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s125-con lr=2e-4", color=COLORS[5])
#
# rew_33 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run62/reward.txt")
# step_33 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run62/steps.txt")
#
# rew_33 = np.expand_dims(rew_33, axis=0)
# plot_curves(x_list=step_33, y_lists=rew_33, xaxis="train steps", title="Hopper-v2 PPO", label="s125-uniform-k=5", color=COLORS[0])
#
# # # rew_111 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run65/reward.txt")
# # # step_111 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run65/steps.txt")
# # #
# # # rew_111 = np.expand_dims(rew_111, axis=0)
# # # plot_curves(x_list=step_111, y_lists=rew_111, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s125-uniform-k=10", color=COLORS[10])
# # #
# # # rew_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run70/reward.txt")
# # # step_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run70/steps.txt")
# # #
# # # rew_3 = np.expand_dims(rew_3, axis=0)
# # # plot_curves(x_list=step_3, y_lists=rew_3, xaxis="train steps", title="Hopper-v2 PPO", label="s125-uniform-k=20", color=COLORS[2])
# # #
#
#
# rew_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run45/reward.txt")
# step_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run45/steps.txt")
#
# rew_31 = np.expand_dims(rew_31, axis=0)
# plot_curves(x_list=step_31, y_lists=rew_31, xaxis="train steps", title="Hopper-v2 PPO", label="s125-actq-k=5 new ", color=COLORS[2])
#
# rew_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run54/reward.txt")
# step_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run54/steps.txt")
#
# rew_31 = np.expand_dims(rew_31, axis=0)
# plot_curves(x_list=step_31, y_lists=rew_31, xaxis="train steps", title="Hopper-v2 PPO", label="s125-blr-2e-4 ", color=COLORS[1])
#
# rew_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run55/reward.txt")
# step_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run55/steps.txt")
#
# rew_31 = np.expand_dims(rew_31, axis=0)
# plot_curves(x_list=step_31, y_lists=rew_31, xaxis="train steps", title="Hopper-v2 PPO", label="s125-blr-1e-4", color=COLORS[3])
#
# rew_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run56/reward.txt")
# step_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run56/steps.txt")
#
# rew_31 = np.expand_dims(rew_31, axis=0)
# plot_curves(x_list=step_31, y_lists=rew_31, xaxis="train steps", title="Hopper-v2 PPO", label="s125-blr-3e-4", color=COLORS[7])
#
# rew_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run62/reward.txt")
# step_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run62/steps.txt")
#
# rew_31 = np.expand_dims(rew_31, axis=0)
# plot_curves(x_list=step_31, y_lists=rew_31, xaxis="train steps", title="Hopper-v2 PPO", label="s125-blr-5e-4", color=COLORS[8])
#
# rew_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run63/reward.txt")
# step_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run63/steps.txt")
#
# rew_31 = np.expand_dims(rew_31, axis=0)
# plot_curves(x_list=step_31, y_lists=rew_31, xaxis="train steps", title="Hopper-v2 PPO", label="s125-blr-1e-3", color=COLORS[9])

# rew_11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run31/reward.txt")
# step_11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run31/steps.txt")
# #
# rew_11 = np.expand_dims(rew_11, axis=0)
# plot_curves(x_list=step_11, y_lists=rew_11, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s125-actq-k=10", color=COLORS[3])
#
# rew_11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run35/reward.txt")
# step_11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run35/steps.txt")
# #
# rew_11 = np.expand_dims(rew_11, axis=0)
# plot_curves(x_list=step_11, y_lists=rew_11, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s125-actq-k=20", color=COLORS[7])



'''
seed 222
'''

# rew_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/data/run1/reward.txt")
# step_119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/data/run1/steps.txt")
#
# rew_119 = np.expand_dims(rew_119, axis=0)
# plot_curves(x_list=step_119, y_lists=rew_119, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s222-con", color=COLORS[4])
#
# rew_33 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/data/run2/reward.txt")
# step_33 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/data/run2/steps.txt")
#
# rew_33 = np.expand_dims(rew_33, axis=0)
# plot_curves(x_list=step_33, y_lists=rew_33, xaxis="train steps", title="Hopper-v2 PPO", label="s222-uniform-k=5", color=COLORS[0])
#
# rew_111 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run73/reward.txt")
# step_111 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run73/steps.txt")
#
# rew_111 = np.expand_dims(rew_111, axis=0)
# plot_curves(x_list=step_111, y_lists=rew_111, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s222-uniform-k=10", color=COLORS[10])
#
# rew_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run74/reward.txt")
# step_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run74/steps.txt")
#
# rew_3 = np.expand_dims(rew_3, axis=0)
# plot_curves(x_list=step_3, y_lists=rew_3, xaxis="train steps", title="Hopper-v2 PPO", label="s222-uniform-k=20", color=COLORS[2])
#
# rew_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run38/reward.txt")
# step_31 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run38/steps.txt")
#
# rew_31 = np.expand_dims(rew_31, axis=0)
# plot_curves(x_list=step_31, y_lists=rew_31, xaxis="train steps", title="Hopper-v2 PPO", label="s222-actq-k=5 ", color=COLORS[1])
#
# rew_11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run40/reward.txt")
# step_11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run40/steps.txt")
# #
# rew_11 = np.expand_dims(rew_11, axis=0)
# plot_curves(x_list=step_11, y_lists=rew_11, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s222-actq-k=10", color=COLORS[3])
#
# rew_11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run39/reward.txt")
# step_11 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run39/steps.txt")
# #
# rew_11 = np.expand_dims(rew_11, axis=0)
# plot_curves(x_list=step_11, y_lists=rew_11, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s222-actq-k=20", color=COLORS[7])

'''
seed 123
'''
#
# rew_112 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run40/reward.txt")
# step_112 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run40/steps.txt")
#
# rew_112 = np.expand_dims(rew_112, axis=0)
# plot_curves(x_list=step_112, y_lists=rew_112, xaxis="train steps", title="Hopper-v2 PPO", label="s123-con", color=COLORS[4])
#
# rew_112 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run76/reward.txt")
# step_112 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run76/steps.txt")
#
# rew_112 = np.expand_dims(rew_112, axis=0)
# plot_curves(x_list=step_112, y_lists=rew_112, xaxis="train steps", title="Hopper-v2 PPO", label="s123-con lr=1e-4", color=COLORS[5])
#
# #
# rew_11119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run63/reward.txt")
# step_11119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run63/steps.txt")
#
# rew_11119 = np.expand_dims(rew_11119, axis=0)
# plot_curves(x_list=step_11119, y_lists=rew_11119, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s123-uniform-k=5", color=COLORS[0])
#
# # rew_11119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run82/reward.txt")
# # step_11119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run82/steps.txt")
# #
# # rew_11119 = np.expand_dims(rew_11119, axis=0)
# # plot_curves(x_list=step_11119, y_lists=rew_11119, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s123-uniform-k=5 lr=1e-4", color=COLORS[1])
# #
# # # rew_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run66/reward.txt")
# # # step_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run66/steps.txt")
# # #
# # # rew_3 = np.expand_dims(rew_3, axis=0)
# # # plot_curves(x_list=step_3, y_lists=rew_3, xaxis="train steps", title="Hopper-v2 PPO", label="s123-uniform-k=10", color=COLORS[10])
# # #
# # # rew_33 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run69/reward.txt")
# # # step_33 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run69/steps.txt")
# # #
# # # rew_33 = np.expand_dims(rew_33, axis=0)
# # # plot_curves(x_list=step_33, y_lists=rew_33, xaxis="train steps", title="Hopper-v2 PPO", label="s123-uniform-k=20", color=COLORS[2])
# # #
# # #
# # rew_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run28/reward.txt")
# # step_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run28/steps.txt")
# #
# # rew_4 = np.expand_dims(rew_4, axis=0)
# # plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s123-actq-k=5", color=COLORS[7])
#
# rew_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run46/reward.txt")
# step_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run46/steps.txt")
#
# rew_4 = np.expand_dims(rew_4, axis=0)
# plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s123-actq-k=5", color=COLORS[2])
#
# rew_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run67/reward.txt")
# step_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run67/steps.txt")
#
# rew_4 = np.expand_dims(rew_4, axis=0)
# plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s123-blr-3e-4", color=COLORS[8])
#
# rew_1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run68/reward.txt")
# step_1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run68/steps.txt")
#
# rew_1 = np.expand_dims(rew_1, axis=0)
# plot_curves(x_list=step_1, y_lists=rew_1, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s123-blr-5e-4", color=COLORS[3])

# rew_1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run36/reward.txt")
# step_1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run36/steps.txt")
#
# rew_1 = np.expand_dims(rew_1, axis=0)
# plot_curves(x_list=step_1, y_lists=rew_1, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s123-actq-k=20", color=COLORS[7])


'''
seed 2
'''
#
# rew_112 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run38/reward.txt")
# step_112 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run38/steps.txt")
#
# rew_112 = np.expand_dims(rew_112, axis=0)
# plot_curves(x_list=step_112, y_lists=rew_112, xaxis="train steps", title="Hopper-v2 PPO", label="s2-con", color=COLORS[4])
# #
# # rew_112 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run81/reward.txt")
# # step_112 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run81/steps.txt")
# #
# # rew_112 = np.expand_dims(rew_112, axis=0)
# # plot_curves(x_list=step_112, y_lists=rew_112, xaxis="train steps", title="Hopper-v2 PPO", label="s2-con lr=1e-4", color=COLORS[5])
# #
# # #
# rew_11119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run64/reward.txt")
# step_11119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run64/steps.txt")
#
# rew_11119 = np.expand_dims(rew_11119, axis=0)
# plot_curves(x_list=step_11119, y_lists=rew_11119, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s2-uniform-k=5", color=COLORS[0])
#
# # rew_11119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run83/reward.txt")
# # step_11119 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run83/steps.txt")
# #
# # rew_11119 = np.expand_dims(rew_11119, axis=0)
# # plot_curves(x_list=step_11119, y_lists=rew_11119, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s2-uniform-k=5 lr=1e-4", color=COLORS[1])
#
# #
# # rew_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run68/reward.txt")
# # step_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run68/steps.txt")
# #
# # rew_3 = np.expand_dims(rew_3, axis=0)
# # plot_curves(x_list=step_3, y_lists=rew_3, xaxis="train steps", title="Hopper-v2 PPO", label="s2-uniform-k=10", color=COLORS[10])
# #
# # rew_12 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run71/reward.txt")
# # step_12 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/ppo/models/Hopper_v2/ppo/run71/steps.txt")
# #
# # rew_12 = np.expand_dims(rew_12, axis=0)
# # plot_curves(x_list=step_12, y_lists=rew_12, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s2-uniform-k=20", color=COLORS[2])
#
#
# # rew_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run30/reward.txt")
# # step_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run30/steps.txt")
# #
# # rew_4 = np.expand_dims(rew_4, axis=0)
# # plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s2-actq-k=5", color=COLORS[7])
# #
# rew_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run47/reward.txt")
# step_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run47/steps.txt")
#
# rew_4 = np.expand_dims(rew_4, axis=0)
# plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s2-actq-k=5", color=COLORS[2])
#
# rew_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run69/reward.txt")
# step_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run69/steps.txt")
#
# rew_4 = np.expand_dims(rew_4, axis=0)
# plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s2-actq-blr-3e-4", color=COLORS[8])
#
# rew_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run70/reward.txt")
# step_4 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run70/steps.txt")
#
# rew_4 = np.expand_dims(rew_4, axis=0)
# plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s2-actq-blr-5e-4", color=COLORS[9])

# rew_1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run33/reward.txt")
# step_1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run33/steps.txt")
#
# rew_1 = np.expand_dims(rew_1, axis=0)
# plot_curves(x_list=step_1, y_lists=rew_1, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s2-actq-k=10", color=COLORS[3])
#
#
# rew_1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run37/reward.txt")
# step_1 = np.loadtxt("/home/mqr/code/sac/rl_algorithms/ppo/models/Hopper_v2/ppo/run37/steps.txt")
#
# rew_1 = np.expand_dims(rew_1, axis=0)
# plot_curves(x_list=step_1, y_lists=rew_1, xaxis="train steps", title="Hopper-v2 PPO-actQ", label="s123-actq-k=20", color=COLORS[7])

# rew_120 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run230/k=5-reward.txt")
# step_120 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run230/k=5-steps.txt")
# 
# rew_120 = np.expand_dims(rew_120, axis=0)
# plot_curves(x_list=step_120, y_lists=rew_120, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="random run230 change", color=COLORS[0])
# 
# rew_122 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run217/k=5-reward.txt")
# step_122 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run217/k=5-steps.txt")
# 
# rew_122 = np.expand_dims(rew_122, axis=0)
# plot_curves(x_list=step_122, y_lists=rew_122, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="random run217 [-0.9,0]", color=COLORS[8])
# 
# rew_123 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run229/k=5-reward.txt")
# step_123 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run229/k=5-steps.txt")
# 
# rew_123 = np.expand_dims(rew_123, axis=0)
# plot_curves(x_list=step_123, y_lists=rew_123, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="random run229 random", color=COLORS[9])
# plt.legend()
'''
170-gpu
'''
# rew_1 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/data/gaussian/run7/k=2-reward.txt")
# step_1 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/data/gaussian/run7/k=2-steps.txt")
#
# rew_1 = np.expand_dims(rew_1, axis=0)
# plot_curves(x_list=step_1, y_lists=rew_1, xaxis="train steps", title="Hopper-v2 SAC", label="178 seed=2", color=COLORS[0])
#
# rew_2 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/data/gaussian/run8/k=2-reward.txt")
# step_2 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/data/gaussian/run8/k=2-steps.txt")
#
# rew_2 = np.expand_dims(rew_2, axis=0)
# plot_curves(x_list=step_2, y_lists=rew_2, xaxis="train steps", title="Hopper-v2 SAC", label="178 seed=3", color=COLORS[1])
#
# rew_19 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/data/gaussian/run9/k=2-reward.txt")
# step_19 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/data/gaussian/run9/k=2-steps.txt")
#
# rew_19 = np.expand_dims(rew_19, axis=0)
# plot_curves(x_list=step_19, y_lists=rew_19, xaxis="train steps", title="Hopper-v2 SAC", label="178 seed=100", color=COLORS[9])
#
# rew_29 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/data/gaussian/run10/k=2-reward.txt")
# step_29 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/data/gaussian/run10/k=2-steps.txt")
#
# rew_29 = np.expand_dims(rew_29, axis=0)
# plot_curves(x_list=step_29, y_lists=rew_29, xaxis="train steps", title="Hopper-v2 SAC", label="178 seed=101", color=COLORS[10])
#
# rew_20 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run6/k=2-reward.txt")
# step_20 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run6/k=2-steps.txt")
#
# rew_20 = np.expand_dims(rew_20, axis=0)
# plot_curves(x_list=step_20, y_lists=rew_20, xaxis="train steps", title="Hopper-v2 SAC", label="170 seed=1", color=COLORS[2])
#
# rew_21 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run7/k=2-reward.txt")
# step_21 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run7/k=2-steps.txt")
#
# rew_21 = np.expand_dims(rew_21, axis=0)
# plot_curves(x_list=step_21, y_lists=rew_21, xaxis="train steps", title="Hopper-v2 SAC", label="170 seed=2", color=COLORS[3])
#
# rew_22 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run8/k=2-reward.txt")
# step_22 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run8/k=2-steps.txt")
#
# rew_22 = np.expand_dims(rew_22, axis=0)
# plot_curves(x_list=step_22, y_lists=rew_22, xaxis="train steps", title="Hopper-v2 SAC", label="170 seed=4", color=COLORS[7])
#
# rew_23 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run9/k=2-reward.txt")
# step_23 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run9/k=2-steps.txt")
#
# rew_23 = np.expand_dims(rew_23, axis=0)
# plot_curves(x_list=step_23, y_lists=rew_23, xaxis="train steps", title="Hopper-v2 SAC", label="170 seed=5", color=COLORS[8])
#
# rew_231 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run10/k=2-reward.txt")
# step_231 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run10/k=2-steps.txt")
#
# rew_231 = np.expand_dims(rew_231, axis=0)
# plot_curves(x_list=step_231, y_lists=rew_231, xaxis="train steps", title="Hopper-v2 SAC", label="170 seed=6", color=COLORS[11])
#
# rew_232 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run11/k=2-reward.txt")
# step_232 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run11/k=2-steps.txt")
#
# rew_232 = np.expand_dims(rew_232, axis=0)
# plot_curves(x_list=step_232, y_lists=rew_232, xaxis="train steps", title="Hopper-v2 SAC", label="170 seed=7", color=COLORS[12])
#
# rew_233 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run14/k=2-reward.txt")
# step_233 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run14/k=2-steps.txt")
#
# rew_233 = np.expand_dims(rew_233, axis=0)
# plot_curves(x_list=step_233, y_lists=rew_233, xaxis="train steps", title="Hopper-v2 SAC", label="170 seed=8", color=COLORS[13])
#
# rew_234 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run13/k=2-reward.txt")
# step_234 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/Gaussian/run13/k=2-steps.txt")
#
# rew_234 = np.expand_dims(rew_234, axis=0)
# plot_curves(x_list=step_234, y_lists=rew_234, xaxis="train steps", title="Hopper-v2 SAC", label="170 seed=9", color=COLORS[14])


# rew_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run187/k=3-reward.txt")
# step_3 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run187/k=3-steps.txt")
#
# rew_3 = np.expand_dims(rew_3, axis=0)
# plot_curves(x_list=step_3, y_lists=rew_3, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="k=3 lr=1e-3", color=COLORS[1])

# rew_4 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run183/k=5-reward.txt")
# step_4 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run183/k=5-steps.txt")
#
# rew_4 = np.expand_dims(rew_4, axis=0)
# plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="k=5 lr=7e-4", color=COLORS[2])
#
# rew_11 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run184/k=10-reward.txt")
# step_11 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run184/k=10-steps.txt")
#
# rew_11 = np.expand_dims(rew_11, axis=0)
# plot_curves(x_list=step_11, y_lists=rew_11, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="k=10 lr=7e-4", color=COLORS[3])
#
# rew_12 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run188/k=20-reward.txt")
# step_12 = np.loadtxt("/home/mqr/code/rl-algorithm/rl_algorithms/sac/models/Hopper_v2/GaussianMixture/run188/k=20-steps.txt")
#
# rew_12 = np.expand_dims(rew_12, axis=0)
# plot_curves(x_list=step_12, y_lists=rew_12, xaxis="train steps", title="Hopper-v2 SAC-GMM", label="k=20 lr=7e-4", color=COLORS[4])

plt.show()

'''----------------------------------------------------------------------------------------'''
# rew_2 = np.loadtxt("models/Hopper_v2/GaussianMixture/run33/k=20-reward.txt")
# step_2 = np.loadtxt("models/Hopper_v2/GaussianMixture/run33/k=20-steps.txt")
# rew_3 = np.loadtxt("models/Hopper_v2/GaussianMixture/run4/k=50-reward.txt")
# step_3 = np.loadtxt("models/Hopper_v2/GaussianMixture/run4/k=50-steps.txt")
# rew_4 = np.loadtxt("models/Hopper_v2/GaussianMixture/run5/k=100-reward.txt")
# step_4 = np.loadtxt("models/Hopper_v2/GaussianMixture/run5/k=100-steps.txt")
#
# # x_val = np.loadtxt("/home/qirui/code/pytorch-sac/models/Hopper_v2/GaussianMixture/run2/k=50-steps.txt")
# # rew = np.loadtxt("/home/qirui/code/pytorch-sac/models/Hopper_v2/GaussianMixture/run2/k=50-reward.txt")
# # gaussian = np.loadtxt("/home/qirui/code/pytorch-sac/models/Hopper_v2/GaussianMixture/run1/k=50-reward.txt")
# # gaussian_step = np.loadtxt("/home/qirui/code/pytorch-sac/models/Hopper_v2/GaussianMixture/run1/k=50-steps.txt")
#
# rew_1 = np.expand_dims(rew_1, axis=0)
# rew_2 = np.expand_dims(rew_2, axis=0)
# rew_3 = np.expand_dims(rew_3, axis=0)
# rew_4 = np.expand_dims(rew_4, axis=0)
# plot_curves(x_list=step_2, y_lists=rew_2, xaxis="train steps", title="Hopper-v2", label="seed1: SAC + GMM k=%d"%20, color='r')
# plot_curves(x_list=step_3, y_lists=rew_3, xaxis="train steps", title="Hopper-v2", label="seed1: SAC + GMM k=%d"%50, color='g')
# plot_curves(x_list=step_4, y_lists=rew_4, xaxis="train steps", title="Hopper-v2", label="seed1: SAC + GMM k=%d"%100, color='c')


