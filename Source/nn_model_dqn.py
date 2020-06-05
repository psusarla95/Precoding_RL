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

        hidden_layers = [400,400,64]
        drop_p =0.5

        #add the first hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])

        #add a variable number of hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.hidden_layers.apply(self._init_weights)

        #create an output layer
        self.output =nn.Linear(hidden_layers[-1], action_size)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        #self.dropout = nn.Dropout(drop_p)

    def forward(self, state):
        """
        Build a network that maps state->action values
        :param state: input observation to the network
        :return: probability action outputs from the network architecture
        """

        #forward through each layer in"hidden layer",with ReLU activation unit between them
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
            #state = self.dropout(state)
        state = self.output(state)
        return state

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


'''
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. /np.sqrt(fan_in)
    return(-lim, lim)

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

        fc1 = 64
        fc2 = 8
        fc3 = 16

        # add the first hidden layer
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

        # self.fc3.weight.data.uniform_(*hidden_init(self.fc3))#0,2*np.pi)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state_tensor):
        """
        Build an actor-ppolicy network that maps states to actions
        """
        #input layer
        t = state_tensor

        # (1) hidden linear layer
        #t1 = self.bn11(t[:,:-2])
        t1 = t[:, :-2]
        t1 = self.fc11(t1)
        t1 = F.relu(t1)

        #t2 = self.bn12(t[:,-2:])
        t2 = t[:,-2:]
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
'''