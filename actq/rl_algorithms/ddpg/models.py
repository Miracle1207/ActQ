import torch
import torch.nn as nn
import torch.nn.functional as F

# define the actor network
class actor(nn.Module):
    def __init__(self, obs_dims, action_dims):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(obs_dims, 400)
        self.fc2 = nn.Linear(400, 300)
        self.action_out = nn.Linear(300, action_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.action_out(x))
        return actions

class critic(nn.Module):
    def __init__(self, obs_dims, action_dims):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(obs_dims, 400)
        self.fc2 = nn.Linear(400 + action_dims, 300)
        self.q_out = nn.Linear(300, 1)

    def forward(self, x, actions):
        x = F.relu(self.fc1(x))
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value

class B_actor(nn.Module):
    def __init__(self, state_size, num_actions):
        super(B_actor, self).__init__()
        self.num_actions = num_actions
        self.fc1_a = nn.Linear(state_size+sum(num_actions), 64)
        self.fc2_a = nn.Linear(64, 64)
        # check the type of distribution
        self.action = nn.Linear(64, sum(num_actions))
        self.action.weight.data.mul_(0.1)
        self.action.bias.data.zero_()


    def forward(self, obs, B):
        # output the policy...
        x = torch.cat((obs, B), 1)
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))

        x_a = self.action(x_a)
        action = F.softmax(x_a, dim=1)
        action = torch.reshape(action, (-1, self.num_actions.size, self.num_actions[0]))
        pi = action

        return pi