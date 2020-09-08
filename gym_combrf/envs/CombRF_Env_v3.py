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
from Source.misc_fun.utils import Generate_Beams
# This is the 3D plotting toolkit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
CombRF_Env_v3 - RF base station precoder Environment
This environment is designed mainly to learn a precoder with a codebook containing varying beamwidth and beamdir

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

- Observation space - [TX_LOC, RSSI]
- Action space - [0,1,2,.......8] -> [(0), ......(RBeamDir).......(pi)]

- Transmit Power= 0 dB, N_tx= 8, N_rx=8
'''

class CombRF_Env_v3(gym.Env):
    def __init__(self):
        self.N_tx = 16 #number of TX antennas
        self.N_rx = 16 #number of RX antennas
        self.P_tx = 0 #Power in dB
        #self.SF_time =20 #msec - for 60 KHz carrier frequency in 5G
        #self.alpha = 0 #angle of rotation of antenna w.r.t y-axis

        self.rx_loc = np.array([[0,0,0]]) #RX is at origin
        self.tx_loc = None
        self.freq = 30e9
        self.df = 60 * 1e3  # 75e3  # carrier spacing frequency
        self.nFFT = 1  # 2048  # no. of subspace carriers
        self.T_sym = 1 / self.df
        self.B = self.nFFT * self.df
        self.sc_xyz = np.array([])#np.array([[0,100,0], [10,50,0], [40,60,0], [70,80,0]])#np.array([[0,100,0]])#np.array([[0,100,0],[250,0,0],[-200,-150,0]]) #reflection points for now
        #self.obs_xyz = np.array([[150,0,0], [150, 150*np.tan(np.pi/16),0], [150, -150*np.tan(np.pi/16),0]])
        self.ch_model ='uma-los' #free-space path loss model
        self.N = self.N_rx #Number of receiver codebook directions

        # noise
        N0dBm = -174  # mW/Hz
        self.N0 = db2lin(N0dBm) * (10 ** -3)  # in WHz-1
        gau = np.zeros((self.N_rx, 1), dtype=np.complex)
        for i in range(gau.shape[0]):
            gau[i] = complex(np.random.randn(), np.random.randn())
        self.noise = np.sqrt(self.N0 / 2) * gau


        self.state = None #initial observation
        self.rate = 0.0 #data rate, could be replaced with SNR as well
        self.cap = None #capacity of the channel for given conditions
        self.rbdir_count = 0
        self.rwd_sum = 0

        self.rx_stepsize = 50 #in m
        self.rx_xcov = np.arange(-200, 250, self.rx_stepsize)#coverage along x axis
        self.rx_ycov = np.arange(-200, 250, self.rx_stepsize) #coverage along y axis
        self.tx_beam = None

        self.aoa_min = 0
        self.aoa_max= 2*math.pi
        self.width_vec = np.array([4*np.pi/self.N_rx, 2*np.pi/self.N_rx, np.pi/self.N_rx])
        print("width_vec: ", self.width_vec)
        self.BeamSet = Generate_Beams(self.N_tx, self.width_vec)  # Set of all beam directions
        self.action_space = spaces.MultiDiscrete([len(self.BeamSet), #first beam
                                                  len(self.BeamSet), #second beam
                                                  len(self.BeamSet), #third beam
                                                 ])
        self.action = None
        self.num_beams = int(self.N_tx*np.sum([2**(x) for x in range(len(self.width_vec))]))
        self.obs_space = spaces.MultiDiscrete([len(self.rx_xcov),  # ue_xloc
                                               len(self.rx_ycov),  # ue_yloc
                                               len(self.BeamSet), #tx_bdir*beam_width
                                               ])


    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        done = False
        while done:
            self.rx_bdir = self.BeamSet[action]
            wRF = ula.steervec(self.rx_bdir[1], self.rx_bdir[0], 0)#self.codebook[:,action[0]]#ula.steervec(self.N_rx, action[0], 0)
            #rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.h[:, :, 0])).dot(self.tx_beam) + (np.conj(wRF.T).dot(self.noise))[0]
            self.rssi_val = np.sqrt(self.rx_bdir[1] * self.tx_bdir[1]) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]

            #compute reward and goal state based on previous rssi value
            rwd, done = self.get_reward_goal(self.rssi_val)
            #done = self._gameover()
            self.rwd_sum = self.rwd_sum + rwd



        #self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        #self.obs = np.array([[self.rssi_val.real, self.rssi_val.imag]])#, self.tx_bdir]])
        self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
                                            self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        #self.obs = np.array([np.concatenate((np.array([self.tx_bdir/np.pi]), self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        #self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.rbdir_count = self.rbdir_count + 1


        return self.obs, rwd, done, {}

    def reset(self):
        #select random TX loc from RX coverage area
        #self.tx_loc = np.array([[random.choice(self.rx_xcov), random.choice(self.rx_ycov), 0]])
        tx_loc_xndx, tx_loc_yndx, tx_dir_ndx =self.obs_space.sample()
        self.tx_loc = np.array([[self.rx_xcov[tx_loc_xndx],self.rx_ycov[tx_loc_yndx], 0]])

        if(np.all(self.tx_loc == [0,0,0])):
            self.tx_loc = np.array([[40,40,0]])
        #select random tx beam from its codebook
        #self.tx_beam = random.choice(self.tx_codebook)


        self.channel = Channel(self.freq, self.tx_loc, self.rx_loc, self.sc_xyz, 'model', self.ch_model, 'nrx', self.N_rx,
                               'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)

        self.channel.generate_paths()
        self.npaths = self.channel.nlos_path + 1
        self.h = self.channel.get_h() #channel coefficient
        self.cap = self.get_capacity() #Compute capacity of channel for given location


        #project TX in the transmitter direction
        #self.tx_beam = ula.steervec(self.N_tx, self.channel.az_aod[0], self.channel.el_aod[0])
        self.tx_bdir = self.BeamSet[tx_dir_ndx]#(tx_bdir, tx_bmwidth)
        print(self.tx_bdir)
        self.tx_beam = ula.steervec(self.tx_bdir[1],self.tx_bdir[0], 0)

        rbdir_ndx,_,_ = self.action_space.sample() #select a random receive direction
        print(rbdir_ndx)
        self.rx_bdir = self.BeamSet[rbdir_ndx] #(rx_bdir, rx_bmwidth)
        wRF = ula.steervec(self.rx_bdir[1], self.rx_bdir[0] , 0)

        self.eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
        self.rssi_val = np.sqrt(self.tx_bdir[1]*self.rx_bdir[1])*np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]

        Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
        self.rate = np.log2(1 + SNR)  # in Gbit/s (self.B / self.nFFT) *
        self.best_rate = self.rate
        #self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.h[:, :, 0])).dot(self.tx_beam)+ (np.conj(wRF.T).dot(self.noise))[0]

        # state should be a factor of affective channel at transmitter + current RSSI value between TX and RX
        # A random state - comes from random fixed TX location, random TX beam from its codebook, random RX beam from its codebook
        #self.obs = np.array([[self.rssi_val.real, self.rssi_val.imag]])# self.tx_bdir]])
        self.norm_tx_xloc = np.array([(self.tx_loc[0][0]+np.max(self.rx_xcov))/(np.max(self.rx_xcov)-np.min(self.rx_xcov))])
        self.norm_tx_yloc = np.array([(self.tx_loc[0][1] + np.max(self.rx_ycov)) / (np.max(self.rx_ycov) - np.min(self.rx_ycov))])
        self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]), \
                                            self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        #self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]), self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        #print(r_bdir, self.tx_loc)
        #self.obs = np.array([np.concatenate((np.array([self.tx_bdir/np.pi]), self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        #self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.rbdir_count = 1
        self.rwd_sum =0
        return self.obs

    def test_reset(self, tx_loc, tbdir_ndx, sc):
        #select TX loc from RX coverage area
        self.tx_loc = tx_loc
        tx_dir_ndx = tbdir_ndx


        self.channel = Channel(self.freq, self.tx_loc, self.rx_loc, sc, 'model', self.ch_model, 'nrx', self.N_rx,
                               'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)

        self.channel.generate_paths()
        self.npaths = self.channel.nlos_path + 1
        self.h = self.channel.get_h() #channel coefficient
        self.cap = self.get_capacity() #Compute capacity of channel for given location

        #project TX in the transmitter direction
        #self.tx_beam = ula.steervec(self.N_tx, self.channel.az_aod[0], self.channel.el_aod[0])
        self.tx_bdir = self.BeamSet[tx_dir_ndx]#self.channel.az_aod[0]#
        self.tx_beam = ula.steervec(self.tx_bdir[1], self.tx_bdir[0], 0)

        rbdir_ndx = self.action_space.sample() #select a random receive direction
        self.rx_bdir = self.BeamSet[rbdir_ndx]
        wRF = ula.steervec(self.rx_bdir[1], self.rx_bdir[0] , 0)
        self.eff_ch = np.array(self.h[:,:,0]).dot(self.tx_beam)
        self.rssi_val = np.sqrt(self.tx_bdir[1]*self.rx_bdir[1])*np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]

        Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
        self.rate = np.log2(1 + SNR)  # in Gbit/s (self.B / self.nFFT) *
        self.best_rate = self.rate

        #self.obs = np.array([[self.rssi_val.real, self.rssi_val.imag]])
        #self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        self.norm_tx_xloc = np.array([(self.tx_loc[0][0]+np.max(self.rx_xcov))/(np.max(self.rx_xcov)-np.min(self.rx_xcov))])
        self.norm_tx_yloc = np.array([(self.tx_loc[0][1] + np.max(self.rx_ycov)) / (np.max(self.rx_ycov) - np.min(self.rx_ycov))])
        self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]), \
                                            self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        #self.obs = np.array([np.concatenate((np.array([self.tx_bdir/np.pi]), self.norm_tx_xloc, self.norm_tx_yloc),axis=0)])
        #self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.rbdir_count = 1
        self.rwd_sum = 0
        return self.obs

    def render(self, mode='human', close=False):
        return

    def get_capacity(self):
        #C= log2(1+P|h*n_r*n_t|^2 /N0
        #C = np.log2(1+ (db2lin(self.P_tx)*((np.square(np.linalg.norm(self.h))*self.N_tx*self.N_rx)+ np.linalg.norm(self.noise[0])**2))/(self.N0/self.B))
        C = np.log2(1 + (db2lin(self.P_tx) * ((np.square(np.linalg.norm(self.h)) * self.N_tx * self.N_rx)) / (self.N0 / self.B)))
        return C

    def get_exh_rate(self):
        best_rate = 0.0
        best_action_ndx = 0
        best_rssi_val = 0
        for ndx in range(len(self.BeamSet)):
            #eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
            wRF = ula.steervec(self.BeamSet[ndx][1], self.BeamSet[ndx][0], 0)
            rssi_val = np.sqrt(self.BeamSet[ndx][1] * self.tx_bdir[1]) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
            Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
            SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
            rate = np.log2(1 + SNR)

            if rate > best_rate:
                best_rate = rate
                best_action_ndx = ndx
                best_rssi_val = rssi_val
        return best_rate, best_action_ndx, best_rssi_val

    def meas_rate(self):
        rssi_values = []
        rate_values = []
        for ndx in range(len(self.BeamSet)):
            #eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
            wRF = ula.steervec(self.BeamSet[ndx][1], self.BeamSet[ndx][0], 0)
            rssi_val = np.sqrt(self.BeamSet[ndx][1] * self.tx_bdir[1]) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
            Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
            SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
            rate = np.log2(1 + SNR)
            rssi_values.append(rssi_val)
            rate_values.append(rate)

        return rssi_values, rate_values

    def get_reward_goal(self, rssi_val):
        #transmission energy
        Es = db2lin(self.P_tx) #* (1e-3 / self.B)
        SNR = Es * np.abs(rssi_val)**2 / (self.N0*self.B)
        rate = np.log2(1 + SNR)  # in Gbit/s (self.B / self.nFFT) *

        rwd=0.0
        done = False
        #if(rate > self.best_rate) and (self.rbdir_count < 8):
        #    self.best_rate = rate
        #    rwd = 1.0
            #done = True#math.exp(-1*self.rbdir_count)#1.0/self.rbdir_count#1.0*math.exp(-1*self.rbdir_count)#rate/self.cap, 1.0*math.exp(-1*self.rbdir_count)
        #if (self.rbdir_count == 3):
        #    rwd = 0.5
        #    done = True

        if (rate >= self.rate) and (self.rbdir_count < 3):
            rwd = 1.0#float(np.round(rate))
        if ((rate >= self.rate) and (self.rbdir_count ==3)):
            done = True
            #rwd =0.0

        self.rate = rate
        #print(rwd)
        return rwd, done

    def _gameover(self):
        if (self.rbdir_count == self.N_rx) or (self.best_rate == self.rate):#or (self.tx_bdir == self.rx_bdir) or (abs(self.tx_bdir-self.rx_bdir)==np.pi):
           return True
        else:
            return False

