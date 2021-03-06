import torch
from torch import nn
from torch.nn import functional as F

"""
this network also include gaussian distribution and beta distribution

"""

class mlp_net(nn.Module):
    def __init__(self, state_size, num_actions, dist_type):
        super(mlp_net, self).__init__()
        self.num_actions = num_actions
        self.dist_type = dist_type
        self.fc1_v = nn.Linear(state_size, 64)
        self.fc2_v = nn.Linear(64, 64)
        self.fc1_a = nn.Linear(state_size, 64)
        self.fc2_a = nn.Linear(64, 64)
        # check the type of distribution
        if self.dist_type == 'gauss':
            self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))
            self.action_mean = nn.Linear(64, num_actions)
            self.action_mean.weight.data.mul_(0.1)
            self.action_mean.bias.data.zero_()
        elif self.dist_type == 'beta':
            self.action_alpha = nn.Linear(64, num_actions)
            self.action_beta = nn.Linear(64, num_actions)
            # init..
            self.action_alpha.weight.data.mul_(0.1)
            self.action_alpha.bias.data.zero_()
            self.action_beta.weight.data.mul_(0.1)
            self.action_beta.bias.data.zero_()
        elif self.dist_type == "uniform" or self.dist_type == "random" or self.dist_type == "ib" or self.dist_type == "pg":
            self.action = nn.Linear(64, sum(num_actions))
            self.action.weight.data.mul_(0.1)
            self.action.bias.data.zero_()

        # define layers to output state value
        self.value = nn.Linear(64, 1)
        self.value.weight.data.mul_(0.1)
        self.value.bias.data.zero_()

    def forward(self, x):
        x_v = torch.tanh(self.fc1_v(x))
        x_v = torch.tanh(self.fc2_v(x_v))
        state_value = self.value(x_v)
        # output the policy...
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))
        if self.dist_type == 'gauss':
            mean = self.action_mean(x_a)
            sigma_log = self.sigma_log.expand_as(mean)
            sigma = torch.exp(sigma_log)
            pi = (mean, sigma)
        elif self.dist_type == 'beta':
            alpha = F.softplus(self.action_alpha(x_a)) + 1
            beta = F.softplus(self.action_beta(x_a)) + 1
            pi = (alpha, beta)
        elif self.dist_type == "uniform" or self.dist_type == "random" or self.dist_type == "ib" or self.dist_type == "pg":
            # action = F.softmax(self.action(x_a), dim=1)
            x_a = self.action(x_a)
            action = F.softmax(x_a)
            action = torch.reshape(action, (-1, self.num_actions.size, self.num_actions[0]))
            pi = action

        return state_value, pi


class B_mlp_net(nn.Module):
    def __init__(self, state_size, num_actions, value_type):
        super(B_mlp_net, self).__init__()
        self.num_actions = num_actions
        self.value_type = value_type
        if self.value_type == "N0_B":
            self.fc1_v = nn.Linear(state_size, 64)
        else:
            self.fc1_v = nn.Linear(state_size+sum(num_actions), 64)
        self.fc2_v = nn.Linear(64, 64)
        self.fc1_a = nn.Linear(state_size+sum(num_actions), 64)
        self.fc2_a = nn.Linear(64, 64)
        # check the type of distribution
        self.action = nn.Linear(64, sum(num_actions))
        self.action.weight.data.mul_(0.1)
        self.action.bias.data.zero_()

        # define layers to output state value
        self.value = nn.Linear(64, 1)
        self.value.weight.data.mul_(0.1)
        self.value.bias.data.zero_()

    def forward(self, obs, B):
        if self.value_type == "N0_B":
            x = obs
        else:
            x = torch.cat((obs, B), 1)
        x_v = torch.tanh(self.fc1_v(x))
        x_v = torch.tanh(self.fc2_v(x_v))
        state_value = self.value(x_v)
        # output the policy...
        x = torch.cat((obs, B), 1)
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))

        x_a = self.action(x_a)
        action = F.softmax(x_a, dim=1)
        action = torch.reshape(action, (-1, self.num_actions.size, self.num_actions[0]))
        pi = action

        return state_value, pi

class discrete_actor_net(nn.Module):
    def __init__(self, state_size, num_actions):
        super(discrete_actor_net, self).__init__()
        self.num_actions = num_actions
        self.fc1_a = nn.Linear(state_size+sum(num_actions), 64)
        self.fc2_a = nn.Linear(64, 64)
        # check the type of distribution
        self.action = nn.Linear(64, sum(num_actions))
        self.action.weight.data.mul_(0.1)
        self.action.bias.data.zero_()

    def forward(self, obs, actions):
        x = torch.cat((obs,actions),1)
        # output the policy...
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))

        x_a = self.action(x_a)
        a_prob = F.softmax(x_a, dim=1)
        a_prob = torch.reshape(a_prob, (-1, self.num_actions.size, self.num_actions[0]))

        return a_prob

class Q_net(nn.Module):
    def __init__(self, state_size, num_actions):
        super(Q_net, self).__init__()
        self.num_actions = num_actions
        self.fc1_a = nn.Linear(state_size+num_actions.shape[0], 64)
        self.fc2_a = nn.Linear(64, 64)
        # check the type of distribution
        self.value = nn.Linear(64, 1)
        self.value.weight.data.mul_(0.1)
        self.value.bias.data.zero_()

    def forward(self, obs, actions):
        x = torch.cat((obs,actions),1)
        # output the policy...
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))

        value = self.value(x_a)

        return value

# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 512) 
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x

# in the initial, just the nature CNN
class cnn_net(nn.Module):
    def __init__(self, num_actions):
        super(cnn_net, self).__init__()
        self.cnn_layer = deepmind()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        value = self.critic(x)
        pi = F.softmax(self.actor(x), dim=1)
        return value, pi
