import sys
import numpy as np
from models import net
from utils import linear_schedule, select_actions, reward_recorder, step_wrapper
import os,sys
sys.path.append(os.path.abspath('../..'))

from rl_utils.experience_replay.experience_replay import replay_buffer
import torch
from datetime import datetime

import copy
# Import the summary writer
from torch.utils.tensorboard import SummaryWriter # Create an instance of the object
writer = SummaryWriter()

# define the dqn agent
class dqn_agent:
    def __init__(self, env, args):
        # define some important 
        self.env = env
        self.args = args
        # define discrete action space B
        self.action_max = self.env.env.env.action_space.high[0]
        self.action_min = self.env.env.env.action_space.low[0]

        # 初始化为uniform
        self.B = np.arange(self.action_min, self.action_max + 0.001,
                           (self.action_max - self.action_min) / (self.args.k - 1), "float32")
        self.B = np.repeat(self.B.reshape(1, self.args.k), self.env.action_space.shape[0], axis=0)
        self.B = torch.tensor(self.B, requires_grad=True, device='cuda' if self.args.cuda else 'cpu')
        # variable B
        self.B_optim = torch.optim.Adam([self.B], lr=self.args.lr)
        self.net = net(self.env.observation_space.shape[0], self.args.hidden_size, self.env.action_space.nvec, self.args.use_dueling, self.args.dist)
        self.old_B = copy.deepcopy(self.B)
        self.B_diff = torch.zeros(1)
        self.B_sets = []

        # copy the self.net as the 
        self.target_net = copy.deepcopy(self.net)
        # make sure the target net has the same weights as the network
        self.target_net.load_state_dict(self.net.state_dict())
        if self.args.cuda:
            self.net.cuda()
            self.target_net.cuda()
        # define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        # define the replay memory
        self.buffer = replay_buffer(self.args.buffer_size)
        # define the linear schedule of the exploration
        self.exploration_schedule = linear_schedule(int(self.args.total_timesteps * self.args.exploration_fraction), \
                                                    self.args.final_ratio, self.args.init_ratio)
        # create the folder to save the models
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # set the environment folder
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)



    '''
    ======================================================================
    '''
    def learn(self):
        num_updates = self.args.total_timesteps // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        final_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        rew = []
        epochs = []

        for update in range(num_updates):
            obs = np.array(self.env.reset())
            for step in range(self.args.nsteps):
                explore_eps = self.exploration_schedule.get_value(update * self.args.nsteps + step)
                with torch.no_grad():
                    obs_tensor = self._get_tensors(obs)
                    action_value = self.net(obs_tensor, self.B.flatten().unsqueeze(0))
                # select actions
                index = select_actions(action_value, explore_eps)
                # excute actions
                action = step_wrapper(index, self.env, self.args.dist, k=self.args.k, action_space=self.B)
                obs_, rewards, dones, _ = self.env.step(action)
                obs_ = np.array(obs_)
                # tryint to append the samples
                self.buffer.add(obs, index, rewards, obs_, float(dones))
                obs = obs_

                # update dones
                if self.args.env_type == 'mujoco':
                    dones = np.array([dones])
                    rewards = np.array([rewards])
                self.dones = dones

                # clear the observation
                for n, done in enumerate(dones):
                    if done:

                        if self.args.env_type == 'mujoco':
                            # reset the environment
                            obs = np.array(self.env.reset())

                # process the rewards part -- display the rewards on the screen
                episode_rewards += rewards
                masks = np.array([0.0 if done_ else 1.0 for done_ in dones], dtype=np.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

            # before update the network, the old network will try to load the weights
            self.target_net.load_state_dict(self.net.state_dict())
            # start to sample the samples from the replay buffer
            for update_i in range(self.args.num_updates):
                batch_samples = self.buffer.sample(self.args.batch_size)
                b_diff, td_loss = self._update_network(batch_samples)
            self.B_sets.append(self.B.flatten().cpu().detach().numpy())
            np.savetxt(str(self.model_path) + "/B.txt", self.B_sets)

            rew.append(final_rewards.mean())
            np.savetxt(str(self.model_path) + "/reward.txt", rew)
            epochs.append((update + 1)*self.args.nsteps*self.args.num_workers)
            np.savetxt(str(self.model_path) + "/steps.txt", epochs)
            now_step = (update + 1)*self.args.nsteps*self.args.num_workers
            writer.add_scalar('policy_loss', td_loss, now_step)
            writer.add_scalar('reward', final_rewards.mean(), now_step)
            writer.add_scalar('B_difference', b_diff, now_step)


            # display the training information
            if update % self.args.display_interval == 0:
                print('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}, Loss: {:.3f}'.format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                    final_rewards.mean(), final_rewards.min(), final_rewards.max(),td_loss) )

                print("B:", self.B)
        writer.close()


    '''
    =======================================================================
    '''


    # update the network
    def _update_network(self, samples):
        obses, actions, rewards, obses_next, dones = samples
        # convert the data to tensor
        obses = self._get_tensors(obses)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        obses_next = self._get_tensors(obses_next)
        dones = torch.tensor(1 - dones, dtype=torch.float32).unsqueeze(-1)
        # convert into gpu
        if self.args.cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()
        # calculate the target value
        Bs = self.B.flatten().unsqueeze(0).repeat(self.args.batch_size, 1)
        with torch.no_grad():
            # if use the double network architecture
            if self.args.use_double_net:
                q_value_ = self.net(obses_next, Bs)
                action_max_idx = torch.argmax(q_value_, dim=1, keepdim=True)
                target_action_value = self.target_net(obses_next, Bs)
                target_action_max_value = target_action_value.gather(1, action_max_idx)
            else:
                target_action_value = self.target_net(obses_next, Bs)
                # target_action_max_value, _ = torch.max(target_action_value, dim=1, keepdim=True)
                target_action_max_value = torch.sum(torch.max(target_action_value, dim=2)[0], dim=1, keepdim=True)
        # target
        expected_value = rewards + self.args.gamma * target_action_max_value * dones
        # get the real q value

        action_value = self.net(obses, Bs)
        real_value = torch.sum(action_value.gather(2, actions),dim=1)
        loss = (expected_value - real_value).pow(2).mean()
        # start to update
        self.optimizer.zero_grad()
        self.B_optim.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.B_optim.step()
        self.optimizer.zero_grad()

        self.B_diff = torch.norm(self.old_B - self.B)
        self.old_B = copy.deepcopy(self.B)

        return self.B_diff.item(), loss.item()

    # get tensors
    def _get_tensors(self, obs):
        if obs.ndim == 3:
            obs = np.transpose(obs, (2, 0, 1))
            obs = np.expand_dims(obs, 0)
        elif obs.ndim == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.tensor(obs, dtype=torch.float32)
        if self.args.cuda:
            obs = obs.cuda()
        return obs
