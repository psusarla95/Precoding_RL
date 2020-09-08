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
from Source.misc_fun.utils import Generate_BeamDirs
# This is the 3D plotting toolkit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
CombRF_Env - RF base station Environment with beam refinement actions as well

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
- Action space - [0,1,2,.......8] -> [(0), ......(RBeamDir).......(pi)]

- Transmit Power= 30 dB, N_tx= 8, N_rx=8
'''

class CombRF_Env_v8(gym.Env):
    def __init__(self):
        self.N_tx = 1 #number of TX antennas
        self.N_rx = 8 #number of RX antennas
        self.P_tx = 0 #Power in dB
        #self.SF_time =20 #msec - for 60 KHz carrier frequency in 5G
        #self.alpha = 0 #angle of rotation of antenna w.r.t y-axis

        self.rx_loc = np.array([[0,0,25]]) #RX is at origin
        self.tx_loc = None
        self.freq = 30e9
        self.c = 3e8 #speed of light
        self.df = 60 * 1e3  # 75e3  # carrier spacing frequency
        self.nFFT = 1  # 2048  # no. of subspace carriers
        self.T_sym = 1 / self.df
        self.B = self.nFFT * self.df

        #self.obs_xyz = np.array([[150,0,0], [150, 150*np.tan(np.pi/16),0], [150, -150*np.tan(np.pi/16),0]])

        self.dbp = 0.0
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

        self.rx_stepsize = 100 #in m
        self.sc_xyz = np.array([[-100*np.cos(80*np.pi/180), 50*np.sin(80*np.pi/180), 22.5], [-100*np.cos(60*np.pi/180), -50**np.sin(60*np.pi/180), 22.5], [-50*np.cos(110*np.pi/180), 100*np.sin(110*np.pi/180), 22.5], [50*np.cos(85*np.pi/180), 100**np.sin(85*np.pi/180), 22.5]])  # np.array([[-100,50,11.5], [-100,-50,11.5], [-50,100,11.5],[50,100,11.5]])#np.array([[50,0,0], [-50,-100,0], [100,50,0],[50,-100,0]])#np.array([[0,100,0], [10,50,0], [40,60,0], [70,80,0], [100,50,0], [80,85,0], [20,30,0], [10,40,0], [80,20,0]])#np.array([[0,100,0]])#np.array([[0,100,0],[250,0,0],[-200,-150,0]]) #reflection points for now
        self.ch_model = 'uma-nlos'  # 'uma-nlos' #free-space path loss model
        self.init_ch_model = 'uma-nlos'

        self.deg = np.array([20,40,60,80])
        self.rx_xcov = np.array([100*np.cos(x*np.pi/180) for x in self.deg])#np.arange(100*np.cos(self.deg), 100*np.cos(self.deg)+1, self.rx_stepsize)#coverage along x axis
        self.rx_ycov = np.array([100*np.sin(x*np.pi/180) for x in self.deg])#np.arange(100*np.sin(self.deg), 100*np.sin(self.deg)+1, self.rx_stepsize) #coverage along y axis
        self.rx_zcov = np.arange(25,26, 10)#np.arange(21.5,22.5, 10)
        self.tx_beam = None

        self.aoa_min = 0
        self.aoa_max= 2*math.pi
        #self.beamwidth_vec = np.array([np.pi / self.N_rx, 2 * np.pi / self.N_rx])#, , 2 * np.pi / self.N_rx,4*np.pi/self.N_rx
        self.beamrefine_levels = np.array([0])
        self.BeamSet = Generate_BeamDirs(self.N_rx, self.beamrefine_levels)  # Set of all beam directions
        #self.BeamSet = np.delete(self.BeamSet, [0,len(self.BeamSet)-2,len(self.BeamSet)-1])
        #print(self.BeamSet)
        self.action_space = spaces.Discrete(len(self.BeamSet))
        self.action = None
        self.goal_steps = 8  # cardinality of Beamset
        self.obs_space = spaces.MultiDiscrete([len(self.rx_xcov),  # ue_xloc
                                               len(self.rx_ycov),  # ue_yloc
                                               len(self.rx_zcov),
                                               self.N_tx, #tx_bdir
                                               ])
        vec1, vec2 = self.get_rssi_range()
        self.min_rssi, self.max_rssi = np.abs(vec1), np.abs(vec2)

        # this logic is mainly for exh search over fast fading channel
        self.tx_locs = []
        for xloc in self.rx_xcov:
            for yloc in self.rx_ycov:
                for zloc in self.rx_zcov:
                    self.tx_locs.append(np.array([[xloc, yloc, zloc]]))

        self.dqnobs_counter = [0 for i in range(self.obs_space.nvec[3] * len(self.tx_locs))]
        self.dqneplen_counter = [0 for i in range(self.obs_space.nvec[3] * len(self.tx_locs))]

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, ch_randval=None):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if (ch_randval is not None):
            self.tx_num = self.get_txloc_ndx(self.tx_loc)
            self.channel.generate_paths(ch_randval, self.tx_num)
            self.npaths = self.channel.nlos_path + 1
            self.h = self.channel.get_h()  # channel coefficient
            self.tx_beam = ula.steervec(self.N_tx, self.tx_bdir, 0)
            self.eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)

        wRF = ula.steervec(self.N_rx, self.BeamSet[action], 0)#self.codebook[:,action[0]]#ula.steervec(self.N_rx, action[0], 0)
        #rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.h[:, :, 0])).dot(self.tx_beam) + (np.conj(wRF.T).dot(self.noise))[0]
        self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) + (np.conj(wRF.T).dot(self.noise))[0]

        #compute reward based on previous rssi value
        rwd, done = self.get_reward_goal(self.rssi_val)
        if (self.rbdir_count == self.goal_steps):
            self.tx_num = self.get_txloc_ndx(self.tx_loc)
            self.dqneplen_counter[self.tx_num * self.obs_space.nvec[3] + self.txdir_ndx] += 1

        self.rwd_sum = self.rwd_sum + rwd

        self.rx_bdir = self.BeamSet[action]

        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        # self.obs = np.array([[self.rssi_val.real, self.rssi_val.imag]])#, self.tx_bdir]])
        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                    self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([np.abs(self.rssi_val)]), \
        #                                     self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.norm_rssi = np.array([(np.abs(self.rssi_val) - self.min_rssi) / (self.max_rssi - self.min_rssi)])
        self.norm_rx_ndx = np.array([action / self.action_space.n])
        # self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx), axis=0)])
        # self.obs = np.array([self.norm_rssi])
        #self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.obs = np.array(
            [np.concatenate((self.norm_rx_ndx, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([self.tx_bdir/np.pi]), self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        # self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.rbdir_count = self.rbdir_count + 1
        return self.obs, rwd, done, {}

    def reset(self, ch_randval):
        #select random TX loc from RX coverage area
        #self.tx_loc = np.array([[random.choice(self.rx_xcov), random.choice(self.rx_ycov), 0]])
        tx_loc_xndx, tx_loc_yndx, tx_loc_zndx, tx_dir_ndx =self.obs_space.sample()
        self.tx_loc = np.array([[self.rx_xcov[tx_loc_xndx],self.rx_ycov[tx_loc_yndx], self.rx_zcov[tx_loc_zndx]]])

        #if(np.all(self.tx_loc == [0,0,22.5])):
        #self.tx_loc = np.array([[40,40,22.5]])

        self.dbp = 4*self.tx_loc[0,2]*self.rx_loc[0,2]*self.freq/self.c
        self.d_2d = np.linalg.norm(np.array([[self.tx_loc[0,0], self.tx_loc[0,1], 0]]) - np.array([[self.rx_loc[0,0], self.rx_loc[0,1], 0]]))

        if(self.dbp <= self.d_2d <= 5e3) and (self.ch_model == 'uma-los'):
            self.ch_model = self.init_ch_model + '-dbp'
        else:
            self.ch_model = self.init_ch_model

        self.channel = Channel(self.freq, self.tx_loc, self.rx_loc, self.sc_xyz, 'model', self.ch_model, 'nrx', self.N_rx,
                               'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)

        self.txdir_ndx = tx_dir_ndx
        self.tx_num = self.get_txloc_ndx(self.tx_loc)
        self.channel.generate_paths(ch_randval, self.tx_num)
        self.dqnobs_counter[self.tx_num*self.obs_space.nvec[3]+tx_dir_ndx] += 1
        self.npaths = self.channel.nlos_path + 1
        self.h = self.channel.get_h() #channel coefficient

        #project TX in the transmitter direction
        #self.tx_beam = ula.steervec(self.N_tx, self.channel.az_aod[0], self.channel.el_aod[0])
        self.tx_bdir = self.BeamSet[tx_dir_ndx]#self.channel.az_aod[0]#
        self.tx_beam = ula.steervec(self.N_tx, self.tx_bdir, 0)

        rbdir_ndx = self.action_space.sample() #select a random receive direction
        self.rx_bdir = self.BeamSet[rbdir_ndx]
        wRF = ula.steervec(self.N_rx, self.rx_bdir , 0)

        self.eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
        self.rssi_val = np.sqrt(self.N_rx*self.N_tx)*np.array(np.conj(wRF.T).dot(self.eff_ch)) + (np.conj(wRF.T).dot(self.noise))[0]

        Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
        self.rate = np.log2(1 + SNR)  # in Gbit/s (self.B / self.nFFT) *
        self.best_rate = self.rate

        # state should be a factor of affective channel at transmitter + current RSSI value between TX and RX
        # A random state - comes from random fixed TX location, random TX beam from its codebook, random RX beam from its codebook
        # self.obs = np.array([[self.rssi_val.real, self.rssi_val.imag]])
        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        self.norm_tx_xloc = np.array([(self.tx_loc[0][0]) / 100])  # np.max(self.rx_xcov)])#np.array([(self.tx_loc[0][0]+np.max(self.rx_xcov))/(np.max(self.rx_xcov))])#-np.min(self.rx_xcov))])
        self.norm_tx_yloc = np.array([(self.tx_loc[0][1]) / 100])  # max(np.max(self.rx_ycov),1)])#np.array([(self.tx_loc[0][1] + np.max(self.rx_ycov)) / (np.max(self.rx_ycov))])# - np.min(self.rx_ycov))])
        self.norm_tx_zloc = np.array([(self.tx_loc[0][2]) / 25])
        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]), \
        #                                    self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.norm_rssi = np.array([(np.abs(self.rssi_val) - self.min_rssi) / (self.max_rssi - self.min_rssi)])
        self.norm_tx_ndx = np.array([tx_dir_ndx/ self.obs_space.nvec[3]])
        self.norm_rx_ndx = np.array([rbdir_ndx/self.action_space.n])
        # self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx), axis=0)])
        # self.obs = np.array([self.norm_rssi])
        #self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.obs = np.array(
            [np.concatenate((self.norm_rx_ndx, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([np.abs(self.rssi_val)]), \
        #                                     self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([self.tx_bdir/np.pi]), self.norm_tx_xloc, self.norm_tx_yloc),axis=0)])
        # self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.rbdir_count = 1
        return self.obs

    def test_reset(self, tx_loc, tbdir_ndx, sc, ch_randval):
        self.tx_loc = tx_loc
        tx_dir_ndx = tbdir_ndx

        self.dbp = 4 * self.tx_loc[0, 2] * self.rx_loc[0, 2] * self.freq / self.c
        self.d_2d = np.linalg.norm(np.array([[self.tx_loc[0, 0], self.tx_loc[0, 1], 0]]) - np.array(
            [[self.rx_loc[0, 0], self.rx_loc[0, 1], 0]]))

        if(self.dbp <= self.d_2d <= 5e3) and (self.ch_model == 'uma-los'):
            self.ch_model = self.init_ch_model + '-dbp'
        else:
            self.ch_model = self.init_ch_model

        self.channel = Channel(self.freq, self.tx_loc, self.rx_loc, sc, 'model', self.ch_model, 'nrx', self.N_rx,
                               'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)

        self.txdir_ndx = tx_dir_ndx
        self.tx_num = self.get_txloc_ndx(self.tx_loc)
        self.channel.generate_paths(ch_randval, self.tx_num)
        self.dqnobs_counter[self.tx_num * self.obs_space.nvec[3] + tx_dir_ndx] += 1
        self.npaths = self.channel.nlos_path + 1
        self.h = self.channel.get_h() #channel coefficient

        #project TX in the transmitter direction
        #self.tx_beam = ula.steervec(self.N_tx, self.channel.az_aod[0], self.channel.el_aod[0])
        self.tx_bdir = self.BeamSet[tx_dir_ndx]#self.channel.az_aod[0]#
        self.tx_beam = ula.steervec(self.N_tx, self.BeamSet[tx_dir_ndx], 0)

        rbdir_ndx = self.action_space.sample() #select a random receive direction
        self.rx_bdir = self.BeamSet[rbdir_ndx]
        wRF = ula.steervec(self.N_rx, self.rx_bdir , 0)
        self.eff_ch = np.array(self.h[:,:,0]).dot(self.tx_beam)
        self.rssi_val = np.sqrt(self.N_rx*self.N_tx)*np.array(np.conj(wRF.T).dot(self.eff_ch)) + (np.conj(wRF.T).dot(self.noise))[0]

        Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
        self.rate = np.log2(1 + SNR)  # in Gbit/s (self.B / self.nFFT) *
        self.best_rate = self.rate

        # self.obs = np.array([[self.rssi_val.real, self.rssi_val.imag]])
        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        self.norm_tx_xloc = np.array([(self.tx_loc[0][0]) / 100])  # np.max(self.rx_xcov)])#np.array([(self.tx_loc[0][0]+np.max(self.rx_xcov))/(np.max(self.rx_xcov))])#-np.min(self.rx_xcov))])
        self.norm_tx_yloc = np.array([(self.tx_loc[0][1]) / 100])  # max(np.max(self.rx_ycov),1)])#np.array([(self.tx_loc[0][1] + np.max(self.rx_ycov)) / (np.max(self.rx_ycov))])# - np.min(self.rx_ycov))])
        self.norm_tx_zloc = np.array([(self.tx_loc[0][2]) / 25])
        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]), \
        #                                    self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.norm_rssi = np.array([(np.abs(self.rssi_val) - self.min_rssi) / (self.max_rssi - self.min_rssi)])
        self.norm_tx_ndx = np.array([tx_dir_ndx / self.obs_space.nvec[3]])
        # self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx), axis=0)])
        # self.obs = np.array([self.norm_rssi])
        #self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.norm_rx_ndx = np.array([rbdir_ndx / self.action_space.n])
        self.obs = np.array(
            [np.concatenate((self.norm_rx_ndx, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc,self.norm_tx_zloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([np.abs(self.rssi_val)]), \
        #                                     self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([self.tx_bdir/np.pi]), self.norm_tx_xloc, self.norm_tx_yloc),axis=0)])
        # self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.rbdir_count = 1
        return self.obs

    def render(self, mode='human', close=False):
        return

    def get_exh_rate(self):
        best_rate = 0.0
        best_action_ndx = 0
        best_rssi_val = 0
        for ndx in range(len(self.BeamSet)):
            #eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
            wRF = ula.steervec(self.N_rx, self.BeamSet[ndx], 0)
            rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
            Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
            SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
            rate = np.log2(1 + SNR)

            if rate > best_rate:
                best_rate = rate
                best_action_ndx = ndx
                best_rssi_val = rssi_val
        return best_rate, best_action_ndx, best_rssi_val

    def get_minmax_exhrate(self, ch_randval):
        max_rate = 0.0
        min_rate = 1e10
        max_action_ndx = 0
        min_action_ndx = 0
        max_rssi_val = 0
        min_rssi_val =0

        self.dbp = 4 * self.tx_loc[0, 2] * self.rx_loc[0, 2] * self.freq / self.c
        self.d_2d = np.linalg.norm(np.array([[self.tx_loc[0, 0], self.tx_loc[0, 1], 0]]) - np.array(
            [[self.rx_loc[0, 0], self.rx_loc[0, 1], 0]]))

        if(self.dbp <= self.d_2d <= 5e3) and (self.ch_model == 'uma-los'):
            self.ch_model = self.init_ch_model + '-dbp'
        else:
            self.ch_model = self.init_ch_model

        channel = Channel(self.freq, self.tx_loc, self.rx_loc, self.sc_xyz, 'model', self.ch_model, 'nrx', self.N_rx,
                               'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)

        tx_num = self.get_txloc_ndx(self.tx_loc)
        channel.generate_paths(ch_randval, tx_num)
        npaths = self.channel.nlos_path + 1
        h = self.channel.get_h()  # channel coefficient
        #self.cap = self.get_capacity()  # Compute capacity of channel for given location

        for rbdir_ndx in range(self.action_space.n):
            wRF = ula.steervec(self.N_rx, self.BeamSet[rbdir_ndx], 0)
            #for tbdir_ndx in range(self.N_tx):
            #tx_beam = ula.steervec(self.N_tx, self.BeamSet[tbdir_ndx], 0)
            #eff_ch = np.array(h[:, :, 0]).dot(tx_beam)

            rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) + (np.conj(wRF.T).dot(self.noise))[0]
            Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
            SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
            rate = np.log2(1 + SNR)

            if rate > max_rate:
                max_rate = rate
                max_action_ndx = rbdir_ndx
                max_rssi_val = rssi_val

            if rate < min_rate:
                min_rate = rate
                min_action_ndx = rbdir_ndx
                min_rssi_val = rssi_val
        return min_rate, max_rate, min_action_ndx, max_action_ndx, min_rssi_val, max_rssi_val

    def meas_rate(self):
        rssi_values = []
        rate_values = []
        for ndx in range(len(self.BeamSet)):
            #eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
            wRF = ula.steervec(self.N_rx, self.BeamSet[ndx], 0)
            rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
            Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
            SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
            rate = np.log2(1 + SNR)
            rssi_values.append(rssi_val)
            rate_values.append(rate)

        return rssi_values, rate_values

    def get_rssi_range(self):
        loc1 = np.array([[0,0,22.5]])
        sc = np.array([[-100*np.cos(80*np.pi/180), 50*np.sin(80*np.pi/180), 22.5], [-100*np.cos(60*np.pi/180), -50**np.sin(60*np.pi/180), 22.5], [-50*np.cos(110*np.pi/180), 100*np.sin(110*np.pi/180), 22.5], [50*np.cos(85*np.pi/180), 100**np.sin(85*np.pi/180), 22.5]])#np.array([[-100,50,21.5], [-100,-50,21.5], [-50,100,21.5],[50,100,21.5]])#, [-50,100,11.5],[50,100,11.5]]np.array([[50,0,0], [-50,-100,0], [100,50,0],[50,-100,0]])#np.array([])
        ch_model = 'uma-nlos'#'uma-nlos'

        channel = Channel(self.freq, loc1, self.rx_loc, sc, 'model', ch_model, 'nrx', self.N_rx,
                               'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)

        tx_num = self.get_txloc_ndx(loc1)
        channel.generate_paths(np.exp(1j * 2 * np.pi * 0.6), tx_num) #0.3 is some defined default value

        h = channel.get_h()

        max_rssi = 0
        for tx_dir_ndx in range(self.N_tx):
            tx_dir = self.BeamSet[tx_dir_ndx]
            tx_beam = ula.steervec(self.N_tx, tx_dir, 0)
            eff_ch = np.array(h[:, :, 0]).dot(tx_beam)
            for rx_dir_ndx in range(len(self.BeamSet)):
                rx_dir = self.BeamSet[rx_dir_ndx]
                wRF = ula.steervec(self.N_rx, rx_dir, 0)
                rssi_val = np.sqrt(self.N_tx*self.N_rx) * np.array(np.conj(wRF.T).dot(eff_ch)) + (np.conj(wRF.T).dot(self.noise))[0]
                if(np.abs(max_rssi)**2 < np.abs(rssi_val)**2):
                    max_rssi = rssi_val

        loc2 = np.array([[500, 500, 22.5]])
        sc = np.array([[-100*np.cos(80*np.pi/180), 50*np.sin(80*np.pi/180), 22.5], [-100*np.cos(60*np.pi/180), -50**np.sin(60*np.pi/180), 22.5], [-50*np.cos(110*np.pi/180), 100*np.sin(110*np.pi/180), 22.5], [50*np.cos(85*np.pi/180), 100**np.sin(85*np.pi/180), 22.5]])#np.array([[-100,50,21.5], [-100,-50,21.5], [-50,100,21.5],[50,100,21.5]])#, [-50,100,11.5],[50,100,11.5]]np.array([[50,0,0], [-50,-100,0], [100,50,0],[50,-100,0]])#np.array([])
        ch_model = 'uma-nlos'#'uma-nlos'

        channel = Channel(self.freq, loc2, self.rx_loc, sc, 'model', ch_model, 'nrx', self.N_rx,
                          'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)

        tx_num = self.get_txloc_ndx(loc2)
        channel.generate_paths(np.exp(1j * 2 * np.pi * 0.6), tx_num)
        h = channel.get_h()

        min_rssi = 1e20
        for tx_dir_ndx in range(self.N_tx):
            tx_dir = self.BeamSet[tx_dir_ndx]
            tx_beam = ula.steervec(self.N_tx, tx_dir, 0)
            eff_ch = np.array(h[:, :, 0]).dot(tx_beam)
            for rx_dir_ndx in range(len(self.BeamSet)):
                rx_dir = self.BeamSet[rx_dir_ndx]
                wRF = ula.steervec(self.N_rx, rx_dir, 0)
                rssi_val = np.sqrt(self.N_rx*self.N_tx) * np.array(np.conj(wRF.T).dot(eff_ch)) + (np.conj(wRF.T).dot(self.noise))[0]
                if (np.abs(min_rssi) ** 2 > np.abs(rssi_val) ** 2):
                    min_rssi = rssi_val

        return min_rssi, max_rssi

    def get_reward_goal(self, rssi_val):
        #transmission energy
        Es = db2lin(self.P_tx) #* (1e-3 / self.B)
        SNR = Es * np.abs(rssi_val)**2 / (self.N0*self.B)
        rate = np.log2(1 + SNR)  # in Gbit/s (self.B / self.nFFT) *
        rwd = 0.0
        done = False

        if (self.rbdir_count == self.goal_steps)or ((rate == self.rate) and self.rbdir_count >1):  # ((rate >= self.rate) and (self.rbdir_count ==2)) or
            done = True
        if  (rate > self.best_rate):  #(rate > self.rate) and and (self.rbdir_count < 2)
            rwd = 1.0#float(np.round(rate))#1.0  # float(np.round(rate))
            self.best_rate = rate
        if (rate < self.rate)or ((rate > self.rate) and (rate < self.best_rate)): # and (rate < self.best_rate)
            rwd = -1.0#*float(np.round(rate))#-1.0
        #rwd = rate

        self.rate = rate
        # print(rwd)
        return rwd, done

    def get_txloc_ndx(self, loc):
        break_flag = False
        loc_ndx = 0
        for xloc in self.rx_xcov:
            for yloc in self.rx_ycov:
                for zloc in self.rx_zcov:
                    tx_loc = np.array([[xloc, yloc, zloc]])
                    if (np.all(tx_loc == loc)):
                        break_flag = True
                        break
                    if break_flag:
                        break
                    loc_ndx = loc_ndx + 1
                if break_flag:
                    break

        return loc_ndx
