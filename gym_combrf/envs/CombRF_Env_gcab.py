import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np
import random
import math
from Source.MIMO import MIMO
from Source.antenna import ula, upa
from Source.misc_fun import channel_mmW, upa_channel_mmW
from Source.misc_fun.geometry import *
from Source.misc_fun.conversion import *
from Source.misc_fun.codebook import DFT_Codebook
from Source.misc_fun.utils import Generate_Beams, Generate_UPABeams
# This is the 3D plotting toolkit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
CombRF_Env_cab - RF base station Environment for Contextual Multi-armed bandit problem using e-greedy policy

for a given location, algorithm learn the best beam pair

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

class CombRF_Env_gcab(gym.Env):
    def __init__(self):
        self.Ntx_y = 8  # tx antennas along y-direction
        self.Ntx_z = 1  # tx antennas along z-direction
        self.Nrx_y = 8  # rx antennas along y-direction
        self.Nrx_z = 1  # rx antennas along z-direction

        self.N_tx = self.Ntx_y * self.Ntx_z  # number of TX antennas
        self.N_rx = self.Nrx_y * self.Nrx_z  # number of RX antennas
        self.P_tx = 0 #Power in dB
        self.ant_arr = 'ula'
        #self.SF_time =20 #msec - for 60 KHz carrier frequency in 5G
        #self.alpha = 0 #angle of rotation of antenna w.r.t y-axis

        self.rx_loc = np.array([[0,0,25]]) #RX is at origin
        self.tx_loc = None
        self.freq = 30e9
        self.c = 3e8  # speed of light
        self.df = 60 * 1e3  # 75e3  # carrier spacing frequency
        self.nFFT = 1  # 2048  # no. of subspace carriers
        self.T_sym = 1 / self.df
        self.B = self.nFFT * self.df
        self.sc_xyz = np.array([])#np.array([[650, 300, 21.5], [0, -550, 21.5]])  # np.array([[-100,50,21.5], [-100,-50,21.5], [-50,100,21.5],[50,100,21.5]])#np.array([[-100,50,11.5], [-100,-50,11.5], [-50,100,11.5],[50,100,11.5]])#np.array([[50,0,0], [-50,-100,0], [100,50,0],[50,-100,0]])#np.array([[0,100,0], [10,50,0], [40,60,0], [70,80,0], [100,50,0], [80,85,0], [20,30,0], [10,40,0], [80,20,0]])#np.array([[0,100,0]])#np.array([[0,100,0],[250,0,0],[-200,-150,0]]) #reflection points for now
        # self.obs_xyz = np.array([[150,0,0], [150, 150*np.tan(np.pi/16),0], [150, -150*np.tan(np.pi/16),0]])
        self.ch_model = 'uma-los'  # 'uma-nlos' #free-space path loss model
        self.init_ch_model = 'uma-los'
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

        self.rx_stepsize = 20  # in m
        self.rx_xcov = np.array([-300])  # np.array([-700,100])#, 650, 100])#np.arange(-200, -1, self.rx_stepsize)#*np.cos(58*np.pi/180)coverage along x axis
        self.rx_ycov = np.array([-300])  # np.arange(-200, -1, self.rx_stepsize)#np.array([300, 550])#, -400, 550])#np.arange(-200, -1, self.rx_stepsize) #coverage along y axis
        self.rx_zcov = np.array([51.5])  # np.arange(21.5,22.5, 10)
        self.tx_beam = None

        self.aoa_min = 0
        self.aoa_max= 2*math.pi
        self.beamwidth_vec = np.array([np.pi / self.N_rx])
        self.action_space = spaces.Discrete(self.N_rx)
        self.action = None
        if self.ant_arr == 'ula':
            self.BeamSet = Generate_Beams(self.N_rx, self.beamwidth_vec)   # Set of all beam directions
        else:
            self.BeamSet = Generate_UPABeams(self.Nrx_y, self.Nrx_z, np.array([np.pi/self.Nrx_y]))  # Set of all beam directions

        self.episode_time = 10  # Fading length considered
        self.obs_space = spaces.MultiDiscrete([len(self.rx_xcov),  # ue_xloc
                                               len(self.rx_ycov),  # ue_yloc
                                               len(self.rx_zcov),
                                               self.N_tx,  # tx_bdir
                                               ])

        # this logic is mainly for exh search over fast fading channel
        self.tx_locs = []
        for xloc in self.rx_xcov:
            for yloc in self.rx_ycov:
                for zloc in self.rx_zcov:
                    if (xloc == 0) and (yloc == 0):
                        self.tx_locs.append(np.array([[50, 50, zloc]]))
                    else:
                        self.tx_locs.append(np.array([[xloc, yloc, zloc]]))


    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, ch_randval=None):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if (ch_randval is not None):
            tx_num = self.get_txloc_ndx(self.tx_loc)
            self.channel.generate_paths(ch_randval, tx_num)
            self.npaths = self.channel.nlos_path + 1
            self.h = self.channel.get_h()  # channel coefficient
            #if self.ant_arr == 'ula':
            #    self.tx_beam = ula.steervec(self.N_tx, self.tx_bdir, 0)
            #if self.ant_arr == 'upa':
            #    self.tx_beam = upa.var_steervec(self.N_tx, self.N_tx, self.tx_bdir)
            self.eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
        # tx_dir_ndx, rx_dir_ndx = int(action / self.N_tx), int(action % self.N_rx)
        if self.ant_arr == 'ula':
            wRF = ula.var_steervec(self.N_rx, self.BeamSet[action],0)  # self.codebook[:,action[0]]#ula.steervec(self.N_rx, action[0], 0)
        if self.ant_arr == 'upa':
            wRF = upa.var_steervec(self.Nrx_y, self.Nrx_z, self.BeamSet[action])
        self.rx_bdir = self.BeamSet[action]
        # fRF = ula.steervec(self.N_tx, self.BeamSet[tx_dir_ndx], 0)
        # self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.h[:, :, 0])).dot(fRF) #+ (np.conj(wRF.T).dot(self.noise))[0]
        # self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
        self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch))
        Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        self.SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
        self.rate = np.log2(1 + self.SNR)
        self.ep_rates.append(self.rate)

        self.rbdir_count = self.rbdir_count + 1

        #rwd, done = self.get_reward_goal(self.rssi_val)
        # assign the reward 1.0 for selected action
        #rwd = self.get_reward(self.rssi_val)
        done = self._gameover()

        return done

    def reset(self, loc_ndx, tx_dir_ndx, ch_randval):
        #select random TX loc from RX coverage area
        #self.tx_loc = np.array([[random.choice(self.rx_xcov), random.choice(self.rx_ycov), 0]])
        #tx_loc_xndx, tx_loc_yndx, tx_dir_ndx =self.obs_space.sample()
        rx_dir_ndx = self.action_space.sample()
        #tx_dir_ndx, rx_dir_ndx = int(temp / self.N_tx), int(temp % self.N_rx)
        self.tx_loc = self.tx_locs[loc_ndx]#np.array([[self.rx_xcov[tx_loc_xndx],self.rx_ycov[tx_loc_yndx], 0]])

        if (self.tx_loc[0][0]==0) and (self.tx_loc[0][1]==0):
            self.tx_loc = np.array([[50,50,self.tx_loc[0][2]]])

        self.dbp = 4 * self.tx_loc[0, 2] * self.rx_loc[0, 2] * self.freq / self.c
        self.d_2d = np.linalg.norm(np.array([[self.tx_loc[0, 0], self.tx_loc[0, 1], 0]]) - np.array(
            [[self.rx_loc[0, 0], self.rx_loc[0, 1], 0]]))

        if (self.dbp <= self.d_2d <= 5e3) and (self.ch_model == 'uma-los'):
            self.ch_model = self.init_ch_model + '-dbp'
        else:
            self.ch_model = self.init_ch_model

        if self.ant_arr == 'ula':
            self.channel = channel_mmW.Channel(self.freq, self.tx_loc, self.rx_loc, self.sc_xyz, 'model', self.ch_model, 'nrx', self.N_rx,
                                   'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)
            self.channel.generate_paths(ch_randval, loc_ndx)
            self.npaths = self.channel.nlos_path + 1
            self.h = self.channel.get_h()  # channel coefficient
            self.tx_bdir = self.BeamSet[tx_dir_ndx]  # self.channel.az_aod[0]#
            self.tx_beam = ula.var_steervec(self.N_tx, self.BeamSet[tx_dir_ndx], 0)

        if self.ant_arr == 'upa':
            self.channel = upa_channel_mmW.Channel(self.freq, self.tx_loc, self.rx_loc, self.sc_xyz, 'model', self.ch_model, 'nrx_y', self.Nrx_y, 'nrx_z', self.Nrx_z,'ntx_y', self.Ntx_y, 'ntx_y', self.Ntx_z, 'nFFT', self.nFFT, 'df', self.df)
            self.channel.generate_paths(ch_randval, loc_ndx)
            self.npaths = self.channel.nlos_path + 1
            self.h = self.channel.get_h() #channel coefficient
            self.tx_bdir = self.BeamSet[tx_dir_ndx]  # self.channel.az_aod[0]#
            self.tx_beam = upa.var_steervec(self.Ntx_y, self.Ntx_z, self.BeamSet[tx_dir_ndx])
        #project TX in the transmitter direction
        #self.tx_beam = ula.steervec(self.N_tx, self.channel.az_aod[0], self.channel.el_aod[0])


        #rbdir_ndx = self.action_space.sample() #select a random receive direction
        #self.rx_bdir = self.BeamSet[rx_dir_ndx]
        #if self.ant_arr == 'ula':
        #    wRF = ula.steervec(self.N_rx, self.rx_bdir,0)  # self.codebook[:,action[0]]#ula.steervec(self.N_rx, action[0], 0)
        #if self.ant_arr == 'upa':
        #    wRF = upa.var_steervec(self.N_rx, self.N_rx, self.rx_bdir)
        #wRF = ula.steervec(self.N_rx, self.rx_bdir , 0)

        self.eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
        self.rbdir_count = 0
        self.best_rate = 0.0
        self.rate = 0.0

        #rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch))  # + (np.conj(wRF.T).dot(self.noise))[0]
        #Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        #SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
        #self.rate = np.log2(1 + SNR)
        self.ep_rates = []
        return self.rbdir_count

    def render(self, mode='human', close=False):
        return

    def get_exh_rate(self):
        best_rate = 0.0
        best_action_ndx = 0
        best_rssi_val = 0
        for ndx in range(self.N_rx):
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

    def meas_rate(self):
        rssi_values = []
        rate_values = []
        for ndx in range(self.N_rx):
            #eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
            wRF = ula.steervec(self.N_rx, self.BeamSet[ndx], 0)
            rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
            Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
            SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
            rate = np.log2(1 + SNR)
            rssi_values.append(rssi_val)
            rate_values.append(rate)

        return rssi_values, rate_values

    def get_rate(self):
        return self.rate

    def get_minmax_exhrate(self, loc_ndx, tx_dir_ndx, ch_randval):
        max_rate = 0.0
        min_rate = 1e10
        max_action_ndx = 0
        min_action_ndx = 0
        max_rssi_val = 0
        min_rssi_val = 0

        tx_loc = self.tx_locs[loc_ndx]
        self.dbp = 4 * tx_loc[0, 2] * self.rx_loc[0, 2] * self.freq / self.c
        self.d_2d = np.linalg.norm(np.array([[tx_loc[0, 0], tx_loc[0, 1], 0]]) - np.array(
            [[self.rx_loc[0, 0], self.rx_loc[0, 1], 0]]))

        if (self.dbp <= self.d_2d <= 5e3) and (self.ch_model == 'uma-los'):
            self.ch_model = self.init_ch_model + '-dbp'
        else:
            self.ch_model = self.init_ch_model

        if self.ant_arr == 'ula':
            channel = channel_mmW.Channel(self.freq, tx_loc, self.rx_loc, self.sc_xyz, 'model', self.ch_model,
                                          'nrx', self.N_rx,
                                          'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)
            channel.generate_paths(ch_randval, loc_ndx)
            # snpaths = channel.nlos_path + 1
            h = channel.get_h()  # channel coefficient
            tx_bdir = self.BeamSet[tx_dir_ndx]  # self.channel.az_aod[0]#
            tx_beam = ula.var_steervec(self.N_tx, tx_bdir, 0)

        if self.ant_arr == 'upa':
            channel = upa_channel_mmW.Channel(self.freq, tx_loc, self.rx_loc, self.sc_xyz, 'model', self.ch_model,
                                              'nrx_y', self.Nrx_y, 'nrx_z', self.Nrx_z, 'ntx_y', self.Ntx_y, 'ntx_y',
                                              self.Ntx_z, 'nFFT', self.nFFT, 'df', self.df)
            channel.generate_paths(ch_randval, loc_ndx)
            # self.npaths = self.channel.nlos_path + 1
            h = channel.get_h()  # channel coefficient
            tx_bdir = self.BeamSet[tx_dir_ndx]  # self.channel.az_aod[0]#
            tx_beam = upa.var_steervec(self.Ntx_y, self.Ntx_z, tx_bdir)

        # tx_num = self.get_txloc_ndx(self.tx_loc)
        # channel.generate_paths(ch_randval, tx_num)
        # npaths = self.channel.nlos_path + 1
        # h = self.channel.get_h()  # channel coefficient
        # self.cap = self.get_capacity()  # Compute capacity of channel for given location
        eff_ch = np.array(h[:, :, 0]).dot(tx_beam)

        for rbdir_ndx in range(self.action_space.n):
            if (self.ant_arr == 'ula'):
                wRF = ula.var_steervec(self.N_rx, self.BeamSet[rbdir_ndx], 0)
            if (self.ant_arr == 'upa'):
                wRF = upa.var_steervec(self.Nrx_y, self.Nrx_z, self.BeamSet[rbdir_ndx])
            # for tbdir_ndx in range(self.N_tx):
            # tx_beam = ula.steervec(self.N_tx, self.BeamSet[tbdir_ndx], 0)
            # eff_ch = np.array(h[:, :, 0]).dot(tx_beam)

            rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(
                np.conj(wRF.T).dot(eff_ch))  # + (np.conj(wRF.T).dot(self.noise))[0]
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


    def compute_data_rate(self, rxbdir_ndx):
        wRF = ula.steervec(self.N_rx, self.BeamSet[rxbdir_ndx], 0)
        rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch))  # + (np.conj(wRF.T).dot(self.noise))[0]

        Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
        rate = np.log2(1 + SNR)  # in Gbit/s (self.B / self.nFFT) *
        return rate

    def get_reward_goal(self, rssi_val):
        # transmission energy
        Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
        rate = np.log2(1 + SNR)  # in Gbit/s (self.B / self.nFFT) *
        # self.SNR_list.append(self.SNR)
        rwd = 0.0  # (float(np.around(rate-self.best_rate,decimals=2)))
        done = False

        if(self.rbdir_count ==1):
            self.rate = rate
            self.best_rate = rate
        if (self.rbdir_count == self.action_space.n) or ((rate == self.rate) and self.rbdir_count > 1):  # ((rate >= self.rate) and (self.rbdir_count ==2)) or
            done = True
        if (rate >= self.best_rate) and (self.rbdir_count > 1):  # (rate > self.rate) and and (self.rbdir_count < 2)
            rwd = 1.0  # float(np.round(rate))#1.0  # float(np.round(rate))
            self.best_rate = rate
        if (rate < self.rate) or ((rate > self.rate) and (rate < self.best_rate)):  # and (rate < self.best_rate)
            rwd = -1.0  # *float(np.round(rate))#-1.0

        self.rate = rate
        return rwd, done

    def _gameover(self):
        if (self.rbdir_count == self.N_rx):#or (self.tx_bdir == self.rx_bdir) or (abs(self.tx_bdir-self.rx_bdir)==np.pi):
           return True
        else:
            return False

    def get_txloc_ndx(self, loc):
        break_flag = False
        loc_ndx = 0
        for xloc in self.rx_xcov:
            for yloc in self.rx_ycov:
                for zloc in self.rx_zcov:
                    if (xloc == 0) and (yloc == 0):
                        tx_loc = np.array([[50, 50, zloc]])
                    else:
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