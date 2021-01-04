import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        hidden_layers = [400,400,256]

        #add the first hidden layer
        fc1=200
        fc2=state_size-3
        #self.fc11 = nn.Linear(in_features=state_size, out_features=fc1)
        #self.fc12 = nn.Linear(in_features=3, out_features=fc2)
        #self.fc22 = nn.Linear(in_features=fc2, out_features=fc1)
        #nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.fc22.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.zeros_(self.fc11.bias)
        #nn.init.zeros_(self.fc12.bias)
        #nn.init.zeros_(self.fc22.bias)
        #add the second hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])

        #add a variable number of hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.hidden_layers.apply(self._init_weights)

        self.softmax = nn.Softmax(dim=1)
        #create an output layer
        self.output =nn.Linear(hidden_layers[-1], action_size)
        #nn.init.xavier_uniform_(self.output.weight)
        nn.init.kaiming_uniform_(self.output.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_normal_(self.output.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.output.bias)
        #self.dropout = nn.Dropout(drop_p)

    def forward(self, state):
        """
        Build a network that maps state->action values
        :param state: input observation to the network
        :return: probability action outputs from the network architecture
        """
        # (1) hidden linear layer
        #t1 = self.fc11(state[:,:-3])
        #t1 = F.relu(t1)

        #t2 = self.fc12(state[:, -3:])
        #t2 = F.relu(t2)
        #t2 = self.fc22(t2)
        #t2= F.relu(t2)

        #state = torch.cat((t1,t2), dim=1)
        #forward through each layer in"hidden layer",with ReLU activation unit between them
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
            #state = torch.sigmoid(linear(state))
            #state = self.dropout(state)
        state = self.softmax(self.output(state))
        #state = self.output(state)
        distribution = Categorical(state)
        return distribution
        #return state

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            #nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)


class Critic(nn.Module):
    def __init__(self, state_size, seed):
        super(Critic, self).__init__()
        self.state_size = state_size
        #self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        hidden_layers = [400,400,256]

        #add the first hidden layer
        fc1=200
        fc2=state_size-3
        #self.fc11 = nn.Linear(in_features=state_size, out_features=fc1)
        #self.fc12 = nn.Linear(in_features=3, out_features=fc2)
        #self.fc22 = nn.Linear(in_features=fc2, out_features=fc1)
        #nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.fc22.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.zeros_(self.fc11.bias)
        #nn.init.zeros_(self.fc12.bias)
        #nn.init.zeros_(self.fc22.bias)
        #add the second hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])

        #add a variable number of hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.hidden_layers.apply(self._init_weights)

        #create an output layer
        self.output =nn.Linear(hidden_layers[-1], 1)
        #nn.init.xavier_uniform_(self.output.weight)
        nn.init.kaiming_uniform_(self.output.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_normal_(self.output.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.output.bias)
        #self.dropout = nn.Dropout(drop_p)

    def forward(self, state):
        """
        Build a network that maps state->action values
        :param state: input observation to the network
        :return: probability action outputs from the network architecture
        """
        # (1) hidden linear layer
        #t1 = self.fc11(state[:,:-3])
        #t1 = F.relu(t1)

        #t2 = self.fc12(state[:, -3:])
        #t2 = F.relu(t2)
        #t2 = self.fc22(t2)
        #t2= F.relu(t2)

        #state = torch.cat((t1,t2), dim=1)
        #forward through each layer in"hidden layer",with ReLU activation unit between them
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
            #state = torch.sigmoid(linear(state))
            #state = self.dropout(state)
        state = self.output(state)
        #state = self.output(state)
        return state

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            #nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)