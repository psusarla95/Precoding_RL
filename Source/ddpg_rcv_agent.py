import random
import copy
import math
import numpy as np
from collections import namedtuple, deque

from Source.nn_model_2 import Actor, Critic

import torch
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE =int(1e6)
LR_ACTOR =2e-4
LR_CRITIC = 2e-4
WEIGHT_DECAY = 0 #L2 weight decay
BATCH_SIZE = 64 #minibatch size
GAMMA = 0.99 #discount factor
TAU=1e-2 #soft update hyper-parameter

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
            #print("learning")

    def act(self, state, add_noise=True):
        """ Returns actions for given state as per current policy"""
        state_tensor = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            #print(state_tensor.size())
            action =self.actor_local(state_tensor).cpu().data.numpy()
        #print("old_action: {}", action)
        self.old_action = action

        if add_noise:
            self.actor_local.train()
            action += self.noise.sample()
        #print("noise action: {}", action)
        #if(math.isnan(action)):
        #    print(action, state_tensor, state)
        action = np.clip(action, -1,1)
        return (action+1)/2 #setting range to [0,1]#np.clip(action, 0,1)*2*math.pi

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
        self.critic_local.train()
        self.critic_target.train()
        self.actor_local.train()
        self.actor_target.train()

        states, actions, rewards,next_states, dones =experiences

        #update critic
        #get predicted next-state actions andQ values from target models
        actions_next =self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        #print("Q_targets: ", Q_targets_next.shape)

        #compute Q targets for current state
        Q_targets =rewards + (gamma*Q_targets_next*(1-dones))

        #compute critic loss
        Q_expected =self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        #minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) #clip local gradients of critic
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

    def save_checkpoint(self, last_timestep):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'
        Arguments:
            last_timestep:  Last timestep in training before saving
            replay_buffer:  Current replay buffer
        """
        checkpoint_name =  'ep_{}.pth.tar'.format(last_timestep)
        #logger.info('Saving checkpoint...')
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor_local.state_dict(),
            'critic': self.critic_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            #'replay_buffer': self.memory,
        }
        #logger.info('Saving model at timestep {}...'.format(last_timestep))
        torch.save(checkpoint, checkpoint_name)
        #gc.collect()
        #logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))

    def load_checkpoint(self, last_timestep):
        """
        Saving the networks and all parameters from a given path. If the given path is None
        then the latest saved file in 'checkpoint_dir' will be used.
        Arguments:
            checkpoint_path:    File to load the model from
        """

        checkpoint_path = 'ep_{}.pth.tar'.format(last_timestep)
        key = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(checkpoint_path, map_location=key)
        start_timestep = checkpoint['last_timestep'] + 1
        self.actor_local.load_state_dict(checkpoint['actor'])
        self.critic_local.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        #replay_buffer = self.memory#checkpoint['replay_buffer']

#        gc.collect()
#        logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
        return start_timestep#, replay_buffer


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
        self.batch_size = batch_size
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
