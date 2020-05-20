import random
import copy
import numpy as np
from collections import namedtuple, deque

from Source.nn_model import Actor, Critic

import torch
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE =int(1e6)
LR_ACTOR =2e-4
LR_CRITIC = 2e-4
WEIGHT_DECAY = 0 #L2 weight decay
BATCH_SIZE = 128 #minibatch size
GAMMA = 0.99 #discount factor
TAU=1e-3 #soft update hyper-parameter

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
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #Critic Network (local and target network)
        self.critic_local =Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        #Noise process
        self.noise = OUNoise(action_size, seed)

        #Replay memory
        self.memory  = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done):
        "Save experiences in replay memory and sample from buffer to learn"

        #Save experiences in memory
        self.memory.add(state, action, reward, next_state, done)

        #Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """ Returns actions for given state as per current policy"""
        state_tensor = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action =self.actor_local(state_tensor).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1,1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards,next_states, dones =experiences

        #update critic
        #get predicted next-state actions andQ values from target models
        actions_next =self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        #compute Q targets for current state
        Q_targets =rewards + (gamma*Q_targets_next*(1-dones))

        #compute critic loss
        Q_expected =self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        #minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1) #clip local gradients of critic
        self.critic_optimizer.step()

        #update actor
        actions_pred = self.actor_local(states)
        actor_loss =-self.critic_local(states, actions_pred).mean()

        #Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #update target network
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """ Ornstein-Uhlenbeck process"""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process"""

        self.mu = mu*np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed =random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (=noise) to mean (mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample"""
        x =self.state
        dx =self.theta*(self.mu -x) + self.sigma*np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """ Fixed size buffer to store experience tuples"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a replay buffer object
        :param action_size (int): size of each action tuple
        :param buffer_size (int): maximum size of the buffer
        :param batch_size (int): size of each training batch
        :param seed (int): a random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.bach_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return current size of internal memory"""
        return len(self.memory)