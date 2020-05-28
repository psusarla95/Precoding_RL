import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. /np.sqrt(fan_in)
    return(-lim, lim)


class Actor(nn.Module):
    """
    Actor-policy model (mu)
    """

    def __init__(self, state_size, action_size, seed, fc1=64, fc2=8, fc3=16):
        """Initialize parameters and build model.
            Params
            ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc11 =nn.Linear(in_features=state_size-2, out_features=fc1)
        self.fc12 = nn.Linear(in_features=2, out_features=fc2)

        self.fc21 = nn.Linear(in_features=fc1, out_features=fc2)
        self.fc22 = nn.Linear(in_features=fc2, out_features=fc2)

        #self.fc3 =nn.Linear(in_features=fc3, out_features=fc3)
        self.fc3 = nn.Linear(in_features=fc3, out_features=action_size)
        self.bn11 =nn.BatchNorm1d(state_size-2) #probably this is not needed
        self.bn12 = nn.BatchNorm1d(2)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc11.weight.data.uniform_(*hidden_init(self.fc11))
        self.fc12.weight.data.uniform_(*hidden_init(self.fc12))

        self.fc21.weight.data.uniform_(*hidden_init(self.fc21))
        self.fc22.weight.data.uniform_(*hidden_init(self.fc22))

        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))#0,2*np.pi)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state_tensor):
        """
        Build an actor-ppolicy network that maps states to actions
        """
        #input layer
        t = state_tensor

        # (1) hidden linear layer
        t1 = self.bn11(t[:,:-2])
        t1 = self.fc11(t1)
        t1 = F.relu(t1)

        t2 = self.bn12(t[:,-2:])
        t2 = self.fc12(t2)
        t2 = F.relu(t2)

        # (2) hidden linear layer
        t1 = self.fc21(t1)
        t1 = F.relu(t1)

        t2 = self.fc22(t2)
        t2 = F.relu(t2)

        #print(t1.shape, t2.shape)

        # (3) Output layer
        t = self.fc3(torch.cat((t1,t2), dim=1))
        t = torch.tanh(t)
        #t = F.relu(t)

        #t = self.fc4(t)
         #torch.softmax(t, dim=1)

        return t


class Critic(nn.Module):
    """
    Build a Critic-Value network - learns a (state, action)-value distribution
    """
    def __init__(self, state_size, action_size, seed, fcs1_units = 64, fcs2_units = 8, fc2_units=64):
        """
        Initialize parameters and build model
        :param state_size (int): dimension of each state
        :param action_size (int): dimension of each action
        :param seed (int): random seed
        :param fcs1_units (int): number of nodes in the first hidden layer
        :param fc2_units (int): number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 =nn.Linear(in_features=state_size-2, out_features=fcs1_units)
        self.fcs2 = nn.Linear(in_features=2, out_features=fcs2_units)

        self.fc2 = nn.Linear(in_features=fcs1_units+fcs2_units+action_size, out_features=fc2_units)
        #self.fc3 = nn.Linear(in_features=fc2_units, out_features=fc3_units)
        self.fc3 =nn.Linear(in_features=fc2_units, out_features=1)


        self.bn11 =nn.BatchNorm1d(state_size-2)
        self.bn12 = nn.BatchNorm1d(2)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fcs2.weight.data.uniform_(*hidden_init(self.fcs2))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state_tensor, action_tensor):
        #xs =F.relu(self.fcs1(state_tensor))

        #xs = self.bn1(xs)
        #x = torch.cat((xs, action_tensor), dim=1)
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #return self.fc4(x)
        t = state_tensor

        t1 = self.bn11(t[:,:-2])
        t2 = self.bn12(t[:,-2:])

        t1 = self.fcs1(t1)
        t1 = F.relu(t1)

        t2 = self.fcs2(t2)
        t2 = F.relu(t2)

        t = torch.cat((t1,t2,action_tensor), dim=1)
        t = self.fc2(t)
        t = F.relu(t)

        return self.fc3(t)
