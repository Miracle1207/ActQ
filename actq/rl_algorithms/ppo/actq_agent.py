import numpy as np
import torch
from torch import optim
from log_path import make_logpath
import os,sys
sys.path.append(os.path.abspath('../..'))
from rl_utils.running_filter.running_filter import ZFilter
from models import cnn_net, mlp_net, Q_net, B_mlp_net
from utils import select_actions, evaluate_actions, step_wrapper
from datetime import datetime
import copy
import d4rl
# Import the summary writer
from torch.utils.tensorboard import SummaryWriter # Create an instance of the object
writer = SummaryWriter()

class actq_agent:
    def __init__(self, envs, args):
        self.envs = envs 
        self.args = args
        self.action_max = self.envs.env.env.action_space.high[0]
        self.action_min = self.envs.env.env.action_space.low[0]
        # start to build the network.

        # 初始化为uniform
        self.B = np.arange(self.action_min, self.action_max+0.001, (self.action_max - self.action_min) / (self.args.k-1), "float32")
        self.B = np.repeat(self.B.reshape(1, self.args.k), self.envs.action_space.shape[0], axis=0)
        self.B = torch.tensor(self.B, requires_grad=True, device='cuda' if self.args.cuda else 'cpu')
        # variable B
        self.B_optim = torch.optim.Adam([self.B], lr=self.args.lr)
        self.net = B_mlp_net(envs.observation_space.shape[0], envs.action_space.nvec, value_type="B")
        self.old_B = copy.deepcopy(self.B)
        self.B_diff = torch.zeros(1)
        self.B_sets = []

        self.old_net = copy.deepcopy(self.net)
        # if use the cuda...
        if self.args.cuda:
            self.net.cuda()
            self.old_net.cuda()
        # define the optimizer...
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
        # running filter...
        if self.args.env_type == 'mujoco':
            num_states = self.envs.observation_space.shape[0]
            self.running_state = ZFilter((num_states, ), clip=5)
        # check saving folder..
        # if not os.path.exists(self.args.save_dir):
        #     os.mkdir(self.args.save_dir)
        self.model_path, _ = make_logpath(self.args.env_name, "ppo")
        # get the observation
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        if self.args.env_type == 'mujoco':
            self.obs[:] = np.expand_dims(self.running_state(self.envs.reset()), 0)
        else:
            self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.args.num_workers)]


    # start to train the network...
    def learn(self):
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        final_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        rew = []
        epochs = []

        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    # get tensors
                    obs_tensor = self._get_tensors(self.obs)
                    values, pis = self.net(obs_tensor, self.B.flatten().unsqueeze(0))

                actions = select_actions(pis, self.args.dist, self.args.env_type)
                input_actions = actions

                # start to store information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                mb_values.append(values.detach().cpu().numpy().squeeze())

                input_actions = step_wrapper(input_actions, self.envs, dist="actq", action_space=self.B)

                # start to excute the actions in the environment
                obs, rewards, dones, _ = self.envs.step(input_actions)
                # update dones
                if self.args.env_type == 'mujoco':
                    dones = np.array([dones])
                    rewards = np.array([rewards])
                self.dones = dones
                mb_rewards.append(rewards)
                # clear the observation
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0
                        if self.args.env_type == 'mujoco':
                            # reset the environment
                            obs = self.envs.reset()
                self.obs = obs if self.args.env_type == 'atari' else np.expand_dims(self.running_state(obs), 0)
                # process the rewards part -- display the rewards on the screen
                episode_rewards += rewards
                masks = np.array([0.0 if done_ else 1.0 for done_ in dones], dtype=np.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            if self.args.env_type == 'mujoco':
                mb_values = np.expand_dims(mb_values, 1)
            # compute the last state value
            with torch.no_grad():
                obs_tensor = self._get_tensors(self.obs)
                last_values, _ = self.net(obs_tensor, self.B.flatten().unsqueeze(0))
                last_values = last_values.detach().cpu().numpy().squeeze()
            # start to compute advantages...

            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1]
                    nextvalues = mb_values[t + 1]
                delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values
            # after compute the returns, let's process the rollouts
            mb_obs = mb_obs.swapaxes(0, 1).reshape(self.batch_ob_shape)
            if self.args.env_type == 'atari':
                mb_actions = mb_actions.swapaxes(0, 1).flatten()
            mb_returns = mb_returns.swapaxes(0, 1).flatten()
            mb_advs = mb_advs.swapaxes(0, 1).flatten()
            # before update the network, the old network will try to load the weights
            self.old_net.load_state_dict(self.net.state_dict())
            self.old_B = copy.deepcopy(self.B)
            # start to update the network

            pl, vl, ent, b_diff = self._update_network(mb_obs, mb_actions, mb_returns, mb_advs, update)

            rew.append(final_rewards.mean())
            np.savetxt(str(self.model_path) + "/reward.txt", rew)
            epochs.append((update + 1)*self.args.nsteps*self.args.num_workers)
            np.savetxt(str(self.model_path) + "/steps.txt", epochs)
            now_step = (update + 1)*self.args.nsteps*self.args.num_workers
            writer.add_scalar('policy_loss', pl, now_step)
            writer.add_scalar('value_loss', vl, now_step)
            writer.add_scalar('reward', final_rewards.mean(), now_step)
            writer.add_scalar('B_difference', b_diff, now_step)
                # print("B loss:", B_loss)

            # display the training information
            if update % self.args.display_interval == 0:
                print('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}, PL: {:.3f},VL: {:.3f}, Ent: {:.3f}'.format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                    final_rewards.mean(), final_rewards.min(), final_rewards.max(), pl, vl, ent) )

                print("B:", self.B)
        writer.close()

    # update the network
    def _update_network(self, obs, actions, returns, advantages, episode):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size
        for ep_i in range(self.args.epoch):

            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_actions = actions[mbinds]
                mb_returns = returns[mbinds]
                mb_advs = advantages[mbinds]
                # convert minibatches to tensor
                mb_obs = self._get_tensors(mb_obs)
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32)
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32).unsqueeze(1)
                mb_advs = torch.tensor(mb_advs, dtype=torch.float32).unsqueeze(1)
                # normalize adv
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                if self.args.cuda:
                    mb_actions = mb_actions.cuda()
                    mb_returns = mb_returns.cuda()
                    mb_advs = mb_advs.cuda()
                # start to get values
                mb_B = self.B.flatten().unsqueeze(0).repeat(nbatch_train, 1)
                old_mb_B = self.old_B.flatten().unsqueeze(0).repeat(nbatch_train, 1)
                mb_values, pis = self.net(mb_obs, mb_B)

                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_pis = self.old_net(mb_obs, old_mb_B)
                    # _, old_pis = self.old_net(mb_obs, mb_B)
                    # get the old log probs
                    old_log_prob, _ = evaluate_actions(old_pis, mb_actions, self.args.dist, self.args.env_type)
                    old_log_prob = old_log_prob.detach()
                # evaluate the current policy
                log_prob, ent_loss = evaluate_actions(pis, mb_actions, self.args.dist, self.args.env_type)
                prob_ratio = torch.exp(log_prob - old_log_prob)

                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                # surr1
                surr1 = prob_ratio * mb_advs
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef

                # clear the grad buffer
                self.optimizer.zero_grad()
                self.B_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_([self.B], self.args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
                self.B_optim.step()
                self.optimizer.step()

                self.B_diff = torch.norm(self.old_B - self.B)


        self.B_sets.append(self.B.flatten().cpu().detach().numpy())
        np.savetxt(str(self.model_path) + "/B.txt", self.B_sets, fmt='%.2f')

        return policy_loss.item(), value_loss.item(), ent_loss.item(), self.B_diff.item()

    # convert the numpy array to tensors
    def _get_tensors(self, obs):
        if self.args.env_type == 'atari':
            obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
        # decide if put the tensor on the GPU
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr
