import torch
import torch.nn as nn
import torch.nn.functional as F

# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)

        return x

# in the initial, just the nature CNN
class net(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_actions, use_dueling=False, actq="uniform"):
        super(net, self).__init__()
        # if use the dueling network
        self.use_dueling = use_dueling
        self.actq = actq
        self.num_actions = num_actions
        # define the network
        self.fc1 = nn.Linear(obs_dim + sum(num_actions), hidden_dim)
        # if not use dueling
        if not self.use_dueling:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.action_value = nn.Linear(hidden_dim, sum(num_actions))
        else:
            # the layer for dueling network architecture
            self.action_fc = nn.Linear(hidden_dim, hidden_dim)
            self.state_value_fc = nn.Linear(hidden_dim, hidden_dim)
            self.action_value = nn.Linear(hidden_dim, sum(num_actions))
            self.state_value = nn.Linear(hidden_dim, 1)

    def forward(self, obs, B):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if self.actq == "uniform":
            x = obs
        else:
            x = torch.cat((obs, B), 1)
        x = F.relu(self.fc1(x))
        if not self.use_dueling:
            x = F.relu(self.fc2(x))
            action_value_out = self.action_value(x)
        else:
            # get the action value
            action_fc = F.relu(self.action_fc(x))
            action_value = self.action_value(action_fc)
            # get the state value
            state_value_fc = F.relu(self.state_value_fc(x))
            state_value = self.state_value(state_value_fc)
            # action value mean
            action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
            action_value_center = action_value - action_value_mean
            # Q = V + A
            action_value_out = state_value + action_value_center
        action = F.softmax(action_value_out, dim=1)
        action = torch.reshape(action, (-1, self.num_actions.size, self.num_actions[0]))
        return action
