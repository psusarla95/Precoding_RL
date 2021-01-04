import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """ Actor (Policy) Model- Value based method"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        hidden_layers = [400,400,256]
        drop_p =0.5

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
        return state

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            #nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)


'''

class QNetwork(nn.Module):
    """
    DQN model (mu)
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
            Params
            ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        fc11 = 128
        fc12 = 64
        fc21 = 128
        fc22 = 64
        fc3 = 64

        # add the first hidden layer
        self.fc11 =nn.Linear(in_features=state_size-2, out_features=fc11)
        nn.init.xavier_uniform_(self.fc11.weight)
        nn.init.zeros_(self.fc11.bias)

        self.fc12 = nn.Linear(in_features=2, out_features=fc12)
        nn.init.xavier_uniform_(self.fc12.weight)
        nn.init.zeros_(self.fc12.bias)

        self.fc21 = nn.Linear(in_features=fc11, out_features=fc21)
        nn.init.xavier_uniform_(self.fc21.weight)
        nn.init.zeros_(self.fc21.bias)

        self.fc22 = nn.Linear(in_features=fc12, out_features=fc22)
        nn.init.xavier_uniform_(self.fc22.weight)
        nn.init.zeros_(self.fc22.bias)

        #self.fc3 =nn.Linear(in_features=fc3, out_features=fc3)
        self.fc3 = nn.Linear(in_features=(fc21+fc22), out_features=fc3)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        self.fc4 = nn.Linear(in_features=fc3, out_features=action_size)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, state_tensor):
        """
        Build an actor-ppolicy network that maps states to actions
        """
        #input layer
        t = state_tensor

        # (1) hidden linear layer
        #t1 = self.bn11(t[:,:-2])
        t1 = t[:, :2]
        t1 = self.fc11(t1)
        t1 = F.relu(t1)

        #t2 = self.bn12(t[:,-2:])
        t2 = t[:,2:]
        t2 = self.fc12(t2)
        t2 = F.relu(t2)

        # (2) hidden linear layer
        t1 = self.fc21(t1)
        t1 = F.relu(t1)

        t2 = self.fc22(t2)
        t2 = F.relu(t2)

        #print(t1.shape, t2.shape)

        # (3) Third - Combination layer
        t = self.fc3(torch.cat((t1,t2), dim=1))
        t = torch.relu(t)
        # (4) Output layer
        t = self.fc4(t)
        t = torch.softmax(t, dim=1)

        return t

'''
