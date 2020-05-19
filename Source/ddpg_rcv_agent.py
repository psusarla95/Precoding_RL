import random

import torch
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE =int(1e6)

device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """
        Interacts and learns from environment
        Params
            ======
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                random_seed (int): random seed
    """

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed =random.seed(seed)

        #Actor Network(local & target network) -learns state-action distribution
        self.actor_local = Actor(state_size, action_size, seed).to(device)