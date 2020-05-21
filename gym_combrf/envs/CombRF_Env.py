import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np
import random
import math
from Source.MIMO import MIMO
from Source.misc_fun.channel_mmW import *
from Source.misc_fun.geometry import *
from Source.misc_fun.codebook import DFT_Codebook
# This is the 3D plotting toolkit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
CombRF_Env - RF base station Environment

RX Base Station - RL agent
TX base station, Channel - CombRF Environment

Model Characteristics:
- TX base station has fixed P_tx with N_t fixed antennas, RX base station has N_r fixed antennas with ULA
- Considers a random TX location with RX base station coverage area (U)
- TX base station follows a codebook unknown to RX base station
- Each episode starts with a fixed random TX location 
- for each episode:
    - choose a random transmitter beam direction with a fixed width (T_br, T_bw)
    - T_br with T_bw passes through the channel with a random R_br and computes RSSI_i, forall i>=1
    - effective channel of T_br with H is computed (H_eff) and sent to RX base station
    - considers the state (H_eff, RSSI_i) and shared with RX base station agent  
- Considers a MIMO model with mmwave frequency

- Main parameters of the RF model
  RSSI- Receiver Signal Strength Indicator 
  H_eff - Affective channel from transmitter
  C- Overall capacity of the selected channel type and environment
  N_rx - No. of receiver codebook directions  

- Observation space - [0,1,2,.....,179] -> [-120, -119, -118,......,60]
- Action space - [0,1,2,.......5624] -> [(-60,-60,1,1), ......(RBS,TBS,RBeamWidth,TBeamWidth).......(60,60,3,3)]

- Transmit Power= 30 dB, N_tx= 8, N_rx=8
'''

class CombRF_Env(gym.Env):
    def __init__(self):
        self.N_tx = 8 #number of TX antennas
        self.N_rx = 8 #number of RX antennas
        self.P_tx = 30 #Power in dB
        self.SF_time =20 #msec - for 60 KHz carrier frequency in 5G
        self.alpha = 0 #angle of rotation of antenna w.r.t y-axis

        self.rx_loc = np.array([[0,0,0]]) #RX is at origin
        self.tx_loc = None
        self.freq = 30e9
        self.df = 60 * 1e3  # 75e3  # carrier spacing frequency
        self.nFFT = 1  # 2048  # no. of subspace carriers
        self.T_sym = 1 / self.df
        self.B = self.nFFT * self.df
        self.sc_xyz = np.array([]) #No reflection points for now
        self.ch_model ='fsp' #free-space path loss model
        self.N = self.N_rx #Number of receiver codebook directions

        # noise
        N0dBm = -174  # mW/Hz
        self.N0 = db2lin(N0dBm) * (10 ** -3)  # in WHz-1
        gau = np.zeros((self.N_rx, 1), dtype=np.complex)
        for i in range(gau.shape[0]):
            gau[i] = complex(np.random.randn(), np.random.randn())
        self.noise = np.sqrt(self.N0 / 2) * gau

        self.tx_codebook = DFT_Codebook([1, self.N_tx, 1]) #input: n_xyz = [ant_x, ant_y, ant_z]

        self.state = None #initial observation
        self.rate = None #data rate, could be replaced with SNR as well
        self.cap = None #capacity of the channel for given conditions
        self.rbdir_count = 0
        self.rwd_sum = 0

        self.rx_stepsize = 50 #in m
        self.rx_xcov = np.arange(-500, 550, self.rx_stepsize)#coverage along x axis
        self.rx_ycov = np.arange(-500, 550, self.rx_stepsize) #coverage along y axis
        self.tx_beam = None

        self.aoa_min = 0
        self.aoa_max= 2*math.pi
        self.action_space = spaces.Box(low=self.aoa_min, high=self.aoa_max, shape=(1,), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        #derive channel from obs space
        #h = self.obs[:-1]
        h = np.array(self.obs[:self.N_rx*self.N_tx], dtype=np.complex) #pick the real part of eff channel from observations
        h.imag = self.obs[self.N_rx*self.N_tx:2*self.N_rx*self.N_tx]
        h = h.reshape(self.N_rx, self.N_tx, 1)

        wRF = ula.steervec(self.N_rx, action[0], 0)
        rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(h[:, :, 0])).dot(self.tx_beam) + (np.conj(wRF.T).dot(self.noise))[0]

        #compute reward based on previous rssi value
        rwd = self.get_reward(rssi_val)
        self.rwd_sum = self.rwd_sum + rwd
        self.obs = np.concatenate((self.obs[:-2], np.array([rssi_val.real]), np.array([rssi_val.imag])), axis=0)
        self.rbdir_count = self.rbdir_count + 1
        done = self._gameover()

        return self.obs, rwd, done, {}

    def reset(self):
        #select random TX loc from RX coverage area
        self.tx_loc = np.array([[random.choice(self.rx_xcov), random.choice(self.rx_ycov), 0]])

        #select random tx beam from its codebook
        self.tx_beam = random.choice(self.tx_codebook)

        channel = Channel(self.freq, self.tx_loc, self.rx_loc, self.sc_xyz, 'model', self.ch_model, 'nrx', self.N_rx,
                               'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)

        channel.generate_paths()
        self.h = channel.get_h() #channel coefficient
        self.cap = self.get_capacity() #Compute capacity of channel for given location

        r_bdir = self.action_space.sample()[0] #select a random receive direction
        wRF = ula.steervec(self.N_rx, r_bdir, 0)
        self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.h[:, :, 0])).dot(self.tx_beam) + (np.conj(wRF.T).dot(self.noise))[0]

        # state should be a factor of affective channel at transmitter + current RSSI value between TX and RX
        # A random state - comes from random fixed TX location, random TX beam from its codebook, random RX beam from its codebook
        #self.obs = np.concatenate((self.h.ravel(), np.array([self.rssi_val])), axis=0)
        self.obs = np.concatenate((self.h.real.ravel(), self.h.imag.ravel(), np.array([self.rssi_val.real]), np.array([self.rssi_val.imag])), axis=0)
        self.rbdir_count = 0
        return self.obs

    def render(self, mode='human', close=False):
        return

    def get_capacity(self):
        #C= log2(1+P|h*n_r*n_t|^2 /N0
        C = np.log2(1+ (db2lin(self.P_tx)*np.square(np.linalg.norm(self.h*np.sqrt(self.N_tx*self.N_rx))))/self.N0)*1e-9
        return C

    def get_reward(self, rssi_val):
        #transmission energy
        Es = db2lin(self.P_tx) #* (1e-3 / self.B)
        SNR = Es * np.abs(rssi_val)**2 / self.N0
        self.rate =  np.log2(1 + SNR) * 1e-9  # in Gbit/s (self.B / self.nFFT) *
        print(self.rate, self.cap)
        return self.rate/self.cap #np.abs(rssi_val)**2 /np.square(np.linalg.norm(self.h*self.N_tx*self.N_rx))#

    def _gameover(self):
        if (self.rbdir_count == self.N_rx):
           return True
        else:
            return False
