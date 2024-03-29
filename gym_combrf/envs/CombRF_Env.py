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

        #self.tx_codebook = DFT_Codebook([1, self.N_tx, 1]) #input: n_xyz = [ant_x, ant_y, ant_z]

        self.state = None #initial observation
        self.rate = 0.0 #data rate, could be replaced with SNR as well
        self.cap = None #capacity of the channel for given conditions
        self.rbdir_count = 0
        self.rwd_sum = 0

        self.rx_stepsize = 50 #in m
        self.rx_xcov = np.arange(-250, 550, self.rx_stepsize)#coverage along x axis
        self.rx_ycov = np.arange(-250, 550, self.rx_stepsize) #coverage along y axis
        self.tx_beam = None

        self.aoa_min = 0
        self.aoa_max= 2*np.pi
        self.action_space = spaces.Box(low=self.aoa_min, high=self.aoa_max, shape=(1,), dtype=np.float32)
        self.action = None
        self.BeamSet = Generate_BeamDir(self.N_tx)  # Set of all beam directions

        self.obs_space = spaces.MultiDiscrete([len(self.rx_xcov),  # ue_xloc
                                               len(self.rx_ycov),  # ue_yloc
                                               self.N_tx,  # tx_bdir
                                               ])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _action(self, action_val):
        self.action = np.array([((action_val[0]+1)/2)*(self.aoa_max- self.aoa_min)+ self.aoa_min])


    #def _reverse_action(self):
    #    self.action -= self.action_space.low

    def step(self, action):
        self._action(action[0])

        #print(self.action)
        assert self.action_space.contains(self.action), "%r (%s) invalid" % (action, type(action))

        #derive channel from obs space
        #h = self.obs[:-1]
        #h = np.array(self.obs[0][:self.N_rx*self.N_tx], dtype=np.complex) #pick the real part of eff channel from observations
        #h.imag = self.obs[0][self.N_rx*self.N_tx:2*self.N_rx*self.N_tx]
        #h = h.reshape(self.N_rx, self.N_tx, 1)

        wRF = ula.steervec(self.N_rx, self.action[0], 0)
        #rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(h[:, :, 0])).dot(self.tx_beam) + (np.conj(wRF.T).dot(self.noise))[0]
        self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) + \
                        (np.conj(wRF.T).dot(self.noise))[0]

        #compute reward based on previous rssi value
        rwd, done = self.get_reward_goal(self.rssi_val)
        self.rwd_sum = self.rwd_sum + rwd
        #self.obs = np.array([np.concatenate((self.obs[0][:-2], np.array([rssi_val.real]), np.array([rssi_val.imag])), axis=0)])

        self.rx_bdir = self.action[0]
        #self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])

        self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
                                             self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])

        self.rbdir_count = self.rbdir_count + 1
        #done = self._gameover()

        return self.obs, rwd, done, {}

    def reset(self):

        tx_loc_xndx, tx_loc_yndx, tx_dir_ndx = self.obs_space.sample()

        #select random TX loc from RX coverage area
        self.tx_loc = np.array([[self.rx_xcov[tx_loc_xndx], self.rx_ycov[tx_loc_yndx], 0]])
        if(np.all(self.tx_loc == [0,0,0])):
            self.tx_loc = np.array([[40,40,0]])
        #select random tx beam from its codebook
        #self.tx_beam = random.choice(self.tx_codebook)


        self.channel = Channel(self.freq, self.tx_loc, self.rx_loc, self.sc_xyz, 'model', self.ch_model, 'nrx', self.N_rx,
                               'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)

        self.channel.generate_paths()
        self.h = self.channel.get_h() #channel coefficient
        self.cap = self.get_capacity() #Compute capacity of channel for given location

        #project TX in the transmitter direction
        self.tx_bdir = self.BeamSet[tx_dir_ndx]
        #self.tx_beam = ula.steervec(self.N_tx, self.channel.az_aod[0], self.channel.el_aod[0])
        self.tx_beam = ula.steervec(self.N_tx, self.tx_bdir, 0)

        self.rx_bdir = self.action_space.sample()[0] #select a random receive direction
        wRF = ula.steervec(self.N_rx, self.rx_bdir, 0)
        self.eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
        self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) + \
                        (np.conj(wRF.T).dot(self.noise))[0]
        #self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.h[:, :, 0])).dot(self.tx_beam)+ (np.conj(wRF.T).dot(self.noise))[0]

        # state should be a factor of affective channel at transmitter + current RSSI value between TX and RX
        # A random state - comes from random fixed TX location, random TX beam from its codebook, random RX beam from its codebook
        #self.obs = np.concatenate((self.h.ravel(), np.array([self.rssi_val])), axis=0)
        #self.obs = np.array([np.concatenate((self.h.real.ravel(), self.h.imag.ravel(), np.array([self.rssi_val.real]), np.array([self.rssi_val.imag])), axis=0)])
        #self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        self.norm_tx_xloc = np.array(
            [(self.tx_loc[0][0] + np.max(self.rx_xcov)) / (np.max(self.rx_xcov) - np.min(self.rx_xcov))])
        self.norm_tx_yloc = np.array(
            [(self.tx_loc[0][1] + np.max(self.rx_ycov)) / (np.max(self.rx_ycov) - np.min(self.rx_ycov))])
        self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]), \
                                             self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])

        #print(r_bdir, self.tx_loc)
        self.rbdir_count = 1
        return self.obs

    def render(self, mode='human', close=False):
        return

    def get_capacity(self):
        #C= log2(1+P|h*n_r*n_t|^2 /N0
        C = np.log2(1+ (db2lin(self.P_tx)*((np.square(np.linalg.norm(self.h))*self.N_tx*self.N_rx)+ np.linalg.norm(self.noise[0])**2))/self.N0)*1e-9
        return C

    def get_reward_goal(self, rssi_val):
        #transmission energy
        Es = db2lin(self.P_tx) #* (1e-3 / self.B)
        SNR = Es * np.abs(rssi_val)**2 / self.N0
        rate =  np.log2(1 + SNR) * 1e-9  # in Gbit/s (self.B / self.nFFT) *

        #if(self.rate/self.cap >= 1):
        #    return 1.0
        #else:
        #    return 0.0#self.rate/self.cap #np.abs(rssi_val)**2 /np.square(np.linalg.norm(self.h*self.N_tx*self.N_rx))#
        #print(rate, self.rate)
        rwd=0.0
        done = False

        if(rate > self.rate):
            rwd = 1.0#rate/self.cap
        elif (self.rbdir_count >= 9): #((rate >= self.rate) and (self.rbdir_count ==2)) or
            rwd = 0.0
            done = True

        self.rate = rate
        return rwd, done

    def _gameover(self):
        if (self.rbdir_count == self.N_rx) or (self.tx_bdir == self.rx_bdir) or (abs(self.tx_bdir-self.rx_bdir)==np.pi):#or (self.rate/self.cap >=1):
           return True
        else:
            return False

def Generate_BeamDir(N):
    #if np.min(self.ue_xloc) < 0 and np.max(self.ue_xloc) > 0:

    min_ang = 0#-math.pi/2
    max_ang = np.pi#math.pi/2
    step_size = (max_ang-min_ang)/N
    beam_angles = np.arange(min_ang+step_size, max_ang+step_size, step_size)

    BeamSet = []#np.zeros(N)#np.fft.fft(np.eye(N))

    #B_i = (i)pi/(N-1), forall 0 <= i <= N-1; 0< beta < pi/(N-1)
    val = min_ang
    for i in range(N):
        BeamSet.append(np.arctan2(np.sin(beam_angles[i]), np.cos(beam_angles[i])))#(i+1)*(max_ang-min_ang)/(N)

    return np.array(BeamSet) #eval(strBeamSet_list)#np.ndarray.tolist(BeamSet)
