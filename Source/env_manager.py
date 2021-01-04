import gym
import gym_combrf
import torch
import numpy as np

"""
This class is mainly for doing some pre-processing, converting numpy arrays from env to tensors and viceversa

"""

class EnvManager():
    def __init__(self, device, env_name, seed):
        self.device = device
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.done = False

    def reset(self, tx_num, ch_randval, eps):
        self.obs =self.env.reset(tx_num, ch_randval, eps)
        return torch.tensor(self.obs, device=self.device, dtype=torch.float32)#.unsqueeze(0)

    def test_reset(self, tx_loc, sc, ch_randval, rbdir_ndx, tbdir_ndx=None):
        self.obs =self.env.test_reset(tx_loc, tbdir_ndx, rbdir_ndx, sc, ch_randval)
        return torch.tensor(self.obs, device=self.device, dtype=torch.float32)#.unsqueeze(0)

    def close(self):
        self.env.close()

    def step(self, action_tensor, ch_randval=None):
        action = action_tensor.item()
        #print(action)
        #next_state, reward, done, temp_rwd,  _ = self.env.step(action, ch_randval)
        next_state, reward, done, _ = self.env.step(action, ch_randval)

        next_state_tensor = torch.tensor(next_state, device=self.device, dtype=torch.float32)#.unsquueze(0)
        reward_tensor = torch.tensor([reward], device=self.device)
        done_tensor = torch.tensor([done], device=self.device)
        #temp_rwd_tensor = torch.tensor([temp_rwd], device=self.device)

        #return next_state_tensor, reward_tensor, done_tensor, temp_rwd_tensor, _
        return next_state_tensor, reward_tensor, done_tensor, _

    def num_actions_available(self):
        return self.env.action_space.n

    def state_size(self):
        #self.obs = self.env.reset(np.exp(1j * 2 * np.pi * 0.6),0)
        #return len(self.env.obs_space.nvec)+1#self.obs.shape[1]
        #return len(self.env.obs_space.nvec)
        #return 2 + len(self.env.obs_space.nvec) -1
        return self.env.N_tx*2+self.env.N_rx*2+ len(self.env.obs_space.nvec)
