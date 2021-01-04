import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np
import random
import math
from Source.MIMO import MIMO
#from Source.misc_fun.channel_mmW import *
from Source.misc_fun import channel_mmW, upa_channel_mmW
from Source.misc_fun.conversion import *
from Source.misc_fun.geometry import *
from Source.antenna import ula, upa
from Source.misc_fun.codebook import DFT_Codebook
from Source.misc_fun.utils import Generate_Beams, Generate_UPABeams
# This is the 3D plotting toolkit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
CombRF_Env - RF base station Environment for o-dqn with starting action and episode length as the actions

based on the chosen episode length, a cyclic set of actions w.r.t starting action are applied under the episode accordingly

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

class CombRF_Env_v14(gym.Env):
    def __init__(self):
        self.Ntx_y = 8  #tx antennas along y-direction
        self.Ntx_z = 1  #tx antennas along z-direction
        self.Nrx_y = 8  #rx antennas along y-direction
        self.Nrx_z = 1  #rx antennas along z-direction

        self.N_tx = self.Ntx_y*self.Ntx_z #number of TX antennas
        self.N_rx = self.Nrx_y*self.Nrx_z #number of RX antennas
        self.P_tx = 0 #Power in dB
        self.ant_arr = 'ula'
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
        self.sc_xyz = np.array([])#np.array([[650,300,61.5], [0,-550,41.5]])#np.array([[-100,50,21.5], [-100,-50,21.5], [-50,100,21.5],[50,100,21.5]])#np.array([[-100,50,11.5], [-100,-50,11.5], [-50,100,11.5],[50,100,11.5]])#np.array([[50,0,0], [-50,-100,0], [100,50,0],[50,-100,0]])#np.array([[0,100,0], [10,50,0], [40,60,0], [70,80,0], [100,50,0], [80,85,0], [20,30,0], [10,40,0], [80,20,0]])#np.array([[0,100,0]])#np.array([[0,100,0],[250,0,0],[-200,-150,0]]) #reflection points for now
        #self.obs_xyz = np.array([[150,0,0], [150, 150*np.tan(np.pi/16),0], [150, -150*np.tan(np.pi/16),0]])
        self.ch_model ='uma-los'#'uma-nlos' #free-space path loss model
        self.init_ch_model = 'uma-los'
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
        self.best_rate =0.0
        self.best_action = -1
        self.cap = None #capacity of the channel for given conditions
        self.rbdir_count = 0
        self.rwd_sum = 0

        self.rx_stepsize = 20 #in m
        self.rx_xcov = np.array([-300,-100, 100])#np.array([-700,100])#, 650, 100])#np.arange(-200, -1, self.rx_stepsize)#*np.cos(58*np.pi/180)coverage along x axis
        self.rx_ycov = np.array([-300,-100, 100])#np.arange(-200, -1, self.rx_stepsize)#np.array([300, 550])#, -400, 550])#np.arange(-200, -1, self.rx_stepsize) #coverage along y axis
        self.rx_zcov = np.array([51.5, 71.5])#np.arange(21.5,22.5, 10)
        self.tx_beam = None

        self.aoa_min = 0
        self.aoa_max= 2*math.pi
        self.beamwidth_vec = np.array([np.pi / self.N_rx])#, , 2 * np.pi / self.N_rx, 2 * np.pi / self.N_rx,4*np.pi/self.N_rx
        if self.ant_arr == 'ula':
            self.BeamSet = Generate_Beams(self.N_rx, self.beamwidth_vec)  # Set of all beam directions  # Set of all beam directions
        else:
            self.BeamSet = Generate_UPABeams(self.N_rx, self.N_rx, np.array([np.pi / self.N_rx]))  # Set of all beam directions
        #self.BeamSet = Generate_UPABeams(self.Nrx_y, self.Nrx_z, self.beamwidth_vec)  # Set of all beam directions
        #self.action_space = spaces.Discrete(int(sum([self.N_rx/(2**i) for i in range(len(self.beamwidth_vec))])))
        self.action_space = spaces.Discrete(self.N_tx * self.N_rx*self.N_tx * self.N_rx)
        self.action = None
        self.goal_steps = self.N_tx*self.N_rx # cardinality of Beamset
        self.obs_space = spaces.MultiDiscrete([len(self.rx_xcov),  # ue_xloc
                                               len(self.rx_ycov),  # ue_yloc
                                               len(self.rx_zcov),
                                               #self.N_tx, #tx_bdir
                                               ])
        #vec1, vec2 = self.get_rssi_range()
        #self.min_rssi, self.max_rssi = np.abs(vec1), np.abs(vec2)

        # this logic is mainly for exh search over fast fading channel
        self.tx_locs = []
        for xloc in self.rx_xcov:
            for yloc in self.rx_ycov:
                for zloc in self.rx_zcov:
                    if (xloc==0) and (yloc==0):
                        self.tx_locs.append(np.array([[50, 50, zloc]]))
                    else:
                        self.tx_locs.append(np.array([[xloc, yloc, zloc]]))

        self.dqnobs_counter = [0 for i in range(len(self.tx_locs))]
        self.dqneplen_counter = [0 for i in range(len(self.tx_locs))]
        self.dqneplen_list = [[] for i in range(len(self.tx_locs))]
        self.dqnepaction_list = [[] for i in range(len(self.tx_locs))]
        self.dqnepsilon_list = [[] for i in range(len(self.tx_locs))]
        self.dqnactionflag_list = [[] for i in range(len(self.tx_locs))]
        self.dqnactionrwd_list = [[] for i in range(len(self.tx_locs))]
        self.dqntemprwd_list = [[] for i in range(len(self.tx_locs))]
        self.action_list =[]
        self.epsilon_list =[]
        self.reward_list = []
        self.temprwd_list = []
        self.dqnbestbeam_ndxlist = [-1 for i in range(len(self.tx_locs))]
        self.dqnbesttxbeam_ndxlist = [-1 for i in range(len(self.tx_locs))]
        self.dqnbestrate_list = [0.0 for i in range(len(self.tx_locs))]
        self.SNR_list =[]

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_coverage(self, rx_xcov, rx_ycov, rx_zcov):
        self.rx_xcov = rx_xcov
        self.rx_ycov = rx_ycov
        self.rx_zcov = rx_zcov

        self.obs_space = spaces.MultiDiscrete([len(self.rx_xcov),  # ue_xloc
                                               len(self.rx_ycov),  # ue_yloc
                                               len(self.rx_zcov),
                                               #self.N_tx,  # tx_bdir
                                               ])
        # vec1, vec2 = self.get_rssi_range()
        # self.min_rssi, self.max_rssi = np.abs(vec1), np.abs(vec2)

        # this logic is mainly for exh search over fast fading channel
        self.tx_locs = []
        for xloc in self.rx_xcov:
            for yloc in self.rx_ycov:
                for zloc in self.rx_zcov:
                    if (xloc == 0) and (yloc == 0):
                        self.tx_locs.append(np.array([[50, 50, zloc]]))
                    else:
                        self.tx_locs.append(np.array([[xloc, yloc, zloc]]))

        self.dqnobs_counter = [0 for i in range(len(self.tx_locs))]
        self.dqneplen_counter = [0 for i in range(len(self.tx_locs))]
        self.dqneplen_list = [[] for i in range(len(self.tx_locs))]
        self.dqnepaction_list = [[] for i in range(len(self.tx_locs))]
        self.dqnepsilon_list = [[] for i in range(len(self.tx_locs))]
        self.dqnactionflag_list = [[] for i in range(len(self.tx_locs))]
        self.dqnactionrwd_list = [[] for i in range(len(self.tx_locs))]
        self.dqntemprwd_list = [[] for i in range(len(self.tx_locs))]
        self.action_list = []
        self.epsilon_list = []
        self.reward_list = []
        self.temprwd_list = []
        self.dqnbestbeam_ndxlist = [-1 for i in range(len(self.tx_locs))]
        #self.dqnbesttxbeam_ndxlist = [-1 for i in range(len(self.tx_locs))]
        self.dqnbestrate_list = [0.0 for i in range(len(self.tx_locs))]
        self.SNR_list = []
        return

    def step(self, action, ch_randval=None):
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        init_action, eps_len = int(action / (self.N_rx*self.N_tx)), int(action % (self.N_rx*self.N_tx))


        tx_dir_ndx, rx_dir_ndx = int(action / self.N_rx), int(action % self.N_rx)
        self.txdir_ndx = tx_dir_ndx
        self.rbdir_ndx = rx_dir_ndx
        self.tx_bdir = self.BeamSet[tx_dir_ndx]  # self.channel.az_aod[0]#
        self.rx_bdir = self.BeamSet[rx_dir_ndx]

        if (ch_randval is not None):
            self.tx_num = self.get_txloc_ndx(self.tx_loc)
            self.channel.generate_paths(ch_randval, self.tx_num)
            self.ch_model = self.channel.model_name

        #self.npaths = self.channel.nlos_path + 1
        self.h = self.channel.get_h()  # channel coefficient
        if (self.ant_arr == 'ula'):
            self.tx_beam = ula.var_steervec(self.N_tx, self.tx_bdir, 0)
        if (self.ant_arr == 'upa'):
            self.tx_beam = upa.var_steervec(self.Ntx_y, self.Ntx_z, self.tx_bdir)
        self.eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)

        if self.ant_arr == 'ula':
            wRF = ula.var_steervec(self.N_rx, self.BeamSet[self.rbdir_ndx],0)  # self.codebook[:,action[0]]#ula.steervec(self.N_rx, action[0], 0)
        if self.ant_arr == 'upa':
            wRF = upa.var_steervec(self.Nrx_y, self.Nrx_z, self.BeamSet[self.rbdir_ndx])

        #rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.h[:, :, 0])).dot(self.tx_beam) + (np.conj(wRF.T).dot(self.noise))[0]
        self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
        self.rbdir_count = self.rbdir_count + 1

        if(self.rbdir_count == 1):
            self.dqn_action = action

        self.rx_bdir = self.BeamSet[self.rbdir_ndx]
        Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        self.SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
        self.rate = np.log2(1 + self.SNR)  # in Gbit/s (self.B / self.nFFT) *

        #self.ep_rates.append(self.rate)
        self.ep_rates[action] = self.rate
        self.ep_rssi[action] = self.rssi_val
        self.ep_actions.append(action)
        #init_flag = False
        #if (self.rate > self.best_rate):
        #    init_flag = True
        #compute reward based on previous rssi value
        #rwd, done, init_flag = self.get_reward_goal(self.rssi_val)#, beam_flag)
        #rwd, init_flag = self.get_rewards()
        #self.dqntemprwd_list[self.tx_num * self.action_space.n + action] = temp_rwd

        #self.tx_num = self.get_txloc_ndx(self.tx_loc)
        #if (init_flag): #and (self.dqnbestrate_list[self.tx_num * self.obs_space.nvec[3] + self.txdir_ndx] < self.best_rate):
        #    self.dqnbestbeam_ndxlist[self.tx_num * self.obs_space.nvec[3] + self.txdir_ndx] = action
            #self.best_action = action
            #self.bestbeam_ndx = action
        #if(self.rbdir_count == self.goal_steps):
        #    self.tx_num = self.get_txloc_ndx(self.tx_loc)
        #    self.dqneplen_counter[self.tx_num * self.obs_space.nvec[3] + self.txdir_ndx] += 1

        #self.rwd_sum = self.rwd_sum + rwd



        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        # self.obs = np.array([[self.rssi_val.real, self.rssi_val.imag]])#, self.tx_bdir]])
        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                    self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([np.abs(self.rssi_val)]), \
        #                                     self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        #self.norm_rssi = np.array([(np.abs(self.rssi_val) - self.min_rssi) / (self.max_rssi - self.min_rssi)])
        self.norm_rx_ndx = np.array([action / self.action_space.n])
        # self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx), axis=0)])
        # self.obs = np.array([self.norm_rssi])
        #self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        #self.obs = np.array(
        #    [np.concatenate((self.norm_rx_ndx, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        #self.obs = np.array(
        #    [np.concatenate((self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        #self.obs = np.array(
        #    [np.concatenate((self.txbeam_obs, self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([self.tx_bdir/np.pi]), self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        # self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])

        return #self.obs, rwd, done, {}

    def reset(self, loc_ndx, ch_randval,eps):
        #select random TX loc from RX coverage area
        #self.tx_loc = np.array([[random.choice(self.rx_xcov), random.choice(self.rx_ycov), 0]])
        #tx_loc_xndx, tx_loc_yndx, tx_loc_zndx, tx_dir_ndx =obs_sample#self.obs_space.sample()
        #self.tx_loc = np.array([[self.rx_xcov[tx_loc_xndx],self.rx_ycov[tx_loc_yndx], self.rx_zcov[tx_loc_zndx]]])
        #if (self.rx_xcov[tx_loc_xndx] == 0) and (self.rx_ycov[tx_loc_yndx] == 0):
        #    self.tx_loc = np.array([[50, 50, self.rx_zcov[tx_loc_zndx]]])
        #tx_num, tx_dir_ndx = obs_sample
        self.tx_loc = self.tx_locs[loc_ndx]
        if(self.tx_loc[0][0] == 0) and (self.tx_loc[0][1] == 0):
            self.tx_loc = np.array([[50, 50, self.tx_loc[0][3]]])
        #if(np.all(self.tx_loc == [0,0,22.5])):
        #self.tx_loc = np.array([[40,40,22.5]])

        #self.dbp = 4*self.tx_loc[0,2]*self.rx_loc[0,2]*self.freq/self.c
        #self.d_2d = np.linalg.norm(np.array([[self.tx_loc[0,0], self.tx_loc[0,1], 0]]) - np.array([[self.rx_loc[0,0], self.rx_loc[0,1], 0]]))

        #if(self.dbp <= self.d_2d <= 5e3) and (self.ch_model == 'uma-los'):
        #    self.ch_model = self.init_ch_model + '-dbp'
        #else:
        #    self.ch_model = self.init_ch_model
        if self.ant_arr == 'ula':
            self.channel = channel_mmW.Channel(self.freq, self.tx_loc, self.rx_loc, self.sc_xyz, 'model',
                                               self.init_ch_model, 'nrx', self.N_rx,
                                               'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)
            self.channel.generate_paths(ch_randval, loc_ndx)
            self.ch_model = self.channel.model_name
            # self.npaths = self.channel.nlos_path + 1
            self.h = self.channel.get_h()  # channel coefficient

        if self.ant_arr == 'upa':
            self.channel = upa_channel_mmW.Channel(self.freq, self.tx_loc, self.rx_loc, self.sc_xyz, 'model', self.init_ch_model, 'nrx_y', self.Nrx_y,'nrx_z', self.Nrx_z,
                                   'ntx_y', self.Ntx_y,'ntx_z', self.Ntx_z, 'nFFT', self.nFFT, 'df', self.df)
            self.channel.generate_paths(ch_randval, loc_ndx)
            self.ch_model = self.channel.model_name
            #self.dqnobs_counter[self.tx_num*self.obs_space.nvec[3]+tx_dir_ndx] += 1
            #self.npaths = self.channel.nlos_path + 1
            self.h = self.channel.get_h() #channel coefficient
            #self.cap = self.get_capacity() #Compute capacity of channel for given location

        action = self.dqnbestbeam_ndxlist[loc_ndx]
        self.tx_num = self.get_txloc_ndx(self.tx_loc)
        if action >=0 :
            tx_dir_ndx, rx_dir_ndx = int(action / self.N_rx), int(action % self.N_rx)
            self.txdir_ndx = tx_dir_ndx
            self.rbdir_ndx = rx_dir_ndx  # self.action_space.sample() #select a random receive direction
            self.change_bestrxaction = self.rbdir_ndx
            self.change_besttxaction = self.txdir_ndx

            if self.ant_arr == 'ula':
                self.tx_bdir = self.BeamSet[self.txdir_ndx]
                # if (self.beam_param == 'beam-width'):
                self.tx_beam = ula.var_steervec(self.N_tx, self.BeamSet[self.txdir_ndx], 0)
                # else:
                #    self.tx_beam = ula.steervec(self.N_tx, self.BeamSet[tx_dir_ndx], 0)
                self.eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)

                self.rx_bdir = self.BeamSet[self.rbdir_ndx]
                # print("reset: ", rbdir_ndx)
                #if (self.beam_param == 'beam-width'):
                wRF = ula.var_steervec(self.N_rx, self.rx_bdir, 0)
                #else:
                #    wRF = ula.steervec(self.N_rx, self.rx_bdir, 0)

                self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch))  # + (np.conj(wRF.T).dot(self.noise))[0]
                Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
                # self.SNR_list = []
                self.SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
                self.best_rate = np.log2(1 + self.SNR)  # in Gbit/s (self.B / self.nFFT) *
                self.rate = self.best_rate
                    #self.init_rate = self.best_rate
                    #self.best_rssi = self.rssi_val
                #else:
                #    wRF = np.ones(self.N_rx, dtype=np.complex)
                #    wRF.real = [-1.0 / np.sqrt(self.N_rx) for i in range(len(wRF))]
                #    wRF.imag = [-1.0 / np.sqrt(self.N_rx) for i in range(len(wRF))]
                #    self.best_rate = -np.inf
                #    self.rate = -np.inf
                    #self.init_rate = -np.inf
                #    self.rssi_val = np.complex(0, 0)

            if self.ant_arr == 'upa':
                #project TX in the transmitter direction
                #self.tx_beam = ula.steervec(self.N_tx, self.channel.az_aod[0], self.channel.el_aod[0])
                #self.channel.az_aod[0]#
                #self.tx_xbdir, self.tx_ybdir = (self.tx_bdir)
                self.tx_bdir = self.BeamSet[self.txdir_ndx]  # self.channel.az_aod[0]#
                self.tx_beam = upa.var_steervec(self.Ntx_y,self.Ntx_z, self.tx_bdir)
                self.eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)

                self.rx_bdir = self.BeamSet[self.rbdir_ndx]
                # print("reset: ", rbdir_ndx)
                wRF = upa.var_steervec(self.Nrx_x, self.Nrx_y, self.rx_bdir)

                self.rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(
                    np.conj(wRF.T).dot(self.eff_ch))  # + (np.conj(wRF.T).dot(self.noise))[0]
                Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
                # self.SNR_list = []
                self.SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
                self.best_rate = np.log2(1 + self.SNR)  # in Gbit/s (self.B / self.nFFT) *
                self.rate = self.best_rate
                self.init_rate = self.best_rate
                # self.best_rssi = self.rssi_val
        else:
            self.txdir_ndx = -1
            self.rbdir_ndx = -1  # self.action_space.sample() #select a random receive direction
            self.change_bestrxaction = -1
            self.change_besttxaction = -1
            fRF = np.ones(self.N_tx, dtype=np.complex)
            fRF.real = [-1.0 / np.sqrt(self.N_tx) for i in range(len(fRF))]
            fRF.imag = [-1.0 / np.sqrt(self.N_tx) for i in range(len(fRF))]
            self.eff_ch = np.array(self.h[:, :, 0]).dot(fRF)

            wRF = np.ones(self.N_rx, dtype=np.complex)
            wRF.real = [-1.0 / np.sqrt(self.N_rx) for i in range(len(wRF))]
            wRF.imag = [-1.0 / np.sqrt(self.N_rx) for i in range(len(wRF))]
            self.best_rate = -np.inf
            self.rate = -np.inf
            #self.init_rate = -np.inf
            self.rssi_val = np.complex(0, 0)



        #self.rssi_val = np.sqrt(self.N_rx*self.N_tx)*np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
        #Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        #self.SNR_list = []
        #self.SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
        #self.best_rate = np.log2(1 + self.SNR)  # in Gbit/s (self.B / self.nFFT) *

        #self.prev_bestaction = self.dqnbestbeam_ndxlist[self.tx_num * self.obs_space.nvec[3] + self.txdir_ndx]
        #if (self.rate >=0.0):
        #    self.dqnbestbeam_ndxlist[self.tx_num * self.obs_space.nvec[3] + self.txdir_ndx] = rbdir_ndx
        #self.SNR = 0
        #self.rbdir_ndx = 0
        #self.rate = 0.0
        #self.best_rate = 0.0#self.rate
        #self.best_action = self.rbdir_ndx
        #self.init_rate = self.best_rate
        #self.mini_rate = self.rate
        #self.beam_flag = 0.0
        #self.SNR_list.append(self.SNR)
        # state should be a factor of affective channel at transmitter + current RSSI value between TX and RX
        # A random state - comes from random fixed TX location, random TX beam from its codebook, random RX beam from its codebook
        # self.obs = np.array([[self.rssi_val.real, self.rssi_val.imag]])
        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        if (len(self.tx_locs) == 1):
            self.norm_tx_xloc = np.array([(self.tx_loc[0][0]) / (np.max(self.rx_xcov))])
            self.norm_tx_yloc = np.array([(self.tx_loc[0][1]) / (np.max(self.rx_ycov))])
            self.norm_tx_zloc = np.array([(self.tx_loc[0][2]) / (np.max(self.rx_zcov))])
        else:
            self.norm_tx_xloc = np.array([(self.tx_loc[0][0]-np.min(self.rx_xcov))/(np.max(self.rx_xcov)-np.min(self.rx_xcov))])#np.array([(self.tx_loc[0][0]) / 1000])  # np.max(self.rx_xcov)])#np.array([(self.tx_loc[0][0]+np.max(self.rx_xcov))/(np.max(self.rx_xcov))])#-np.min(self.rx_xcov))])
            self.norm_tx_yloc = np.array([(self.tx_loc[0][1]-np.min(self.rx_ycov))/(np.max(self.rx_ycov)-np.min(self.rx_ycov))])#np.array([(self.tx_loc[0][1]) / 1000])  # max(np.max(self.rx_ycov),1)])#np.array([(self.tx_loc[0][1] + np.max(self.rx_ycov)) / (np.max(self.rx_ycov))])# - np.min(self.rx_ycov))])
            self.norm_tx_zloc = np.array([(self.tx_loc[0][2]-np.min(self.rx_zcov))/(np.max(self.rx_zcov)-np.min(self.rx_zcov))])#-np.min(self.rx_zcov)-np.min(self.rx_zcov)np.array([(self.tx_loc[0][2]) / 22.5])
        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]), \
        #                                    self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        #self.norm_rssi = np.array([(np.abs(self.rssi_val) - self.min_rssi) / (self.max_rssi - self.min_rssi)])
        self.norm_tx_ndx = np.array([self.txdir_ndx/ self.N_tx])
        self.norm_rx_ndx = np.array([self.rbdir_ndx/self.N_rx])
        # self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx), axis=0)])
        # self.obs = np.array([self.norm_rssi])
        #self.obs = np.array([np.concatenate((self.norm_rssi, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        #self.obs = np.array(
        #    [np.concatenate((self.norm_rx_ndx, self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        #self.obs = np.array([np.concatenate((self.norm_tx_ndx, self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        #self.txbeam_obs = []
        #min_val = -1.0/np.sqrt(self.N_tx)
        #max_val = 1.0/np.sqrt(self.N_tx)
        #for elem in self.tx_beam:
        #    self.txbeam_obs.append((elem.real-min_val)/(max_val-min_val))
        #    self.txbeam_obs.append((elem.imag-min_val)/(max_val-min_val))
        #self.txbeam_obs = np.array(self.txbeam_obs)

        #self.obs = np.array([np.concatenate((self.txbeam_obs, self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc, self.norm_tx_zloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([np.abs(self.rssi_val)]), \
        #                                     self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        # self.obs = np.array([np.concatenate((np.array([self.tx_bdir/np.pi]), self.norm_tx_xloc, self.norm_tx_yloc),axis=0)])
        # self.obs = np.array([np.concatenate((self.norm_tx_xloc, self.norm_tx_yloc), axis=0)])
        self.ep_actions = []
        self.action_list =[]
        self.ep_rates = [-np.inf for x in range(self.action_space.n)]
        self.ep_rssi = [np.complex(0,0) for x in range(self.action_space.n)]
        self.rbdir_count = 0
        self.dqn_action = None
        self.nbgh_action = False
        return self.obs

    def test_reset(self, tx_loc, tbdir_ndx,rbdir_ndx, sc, ch_randval):
        self.tx_loc = tx_loc
        tx_dir_ndx = tbdir_ndx

        self.dbp = 4 * self.tx_loc[0, 2] * self.rx_loc[0, 2] * self.freq / self.c
        self.d_2d = np.linalg.norm(np.array([[self.tx_loc[0, 0], self.tx_loc[0, 1], 0]]) - np.array(
            [[self.rx_loc[0, 0], self.rx_loc[0, 1], 0]]))

        if(self.dbp <= self.d_2d <= 5e3) and (self.ch_model == 'uma-los'):
            self.ch_model = self.init_ch_model + '-dbp'
        else:
            self.ch_model = self.init_ch_model

        self.channel = Channel(self.freq, self.tx_loc, self.rx_loc, sc, 'model', self.ch_model, 'nrx_y', self.Nrx_y, 'nrx_z', self.Nrx_z,
                               'ntx_y', self.Ntx_y, 'ntx_y', self.Ntx_z, 'nFFT', self.nFFT, 'df', self.df)

        self.txdir_ndx = tx_dir_ndx
        self.tx_num = self.get_txloc_ndx(self.tx_loc)
        self.channel.generate_paths(ch_randval, self.tx_num)
        #self.dqnobs_counter[self.tx_num * self.obs_space.nvec[3] + tx_dir_ndx] += 1
        self.npaths = self.channel.nlos_path + 1
        self.h = self.channel.get_h() #channel coefficient
        #self.cap = self.get_capacity() #Compute capacity of channel for given location

        #project TX in the transmitter direction
        #self.tx_beam = ula.steervec(self.N_tx, self.channel.az_aod[0], self.channel.el_aod[0])
        self.tx_bdir = self.BeamSet[tx_dir_ndx]#self.channel.az_aod[0]#
        self.tx_beam = upa.var_steervec(self.Ntx_y, self.Ntx_z, self.BeamSet[tx_dir_ndx])

        #rbdir_ndx = self.action_space.sample() #select a random receive direction
        self.rx_bdir = self.BeamSet[rbdir_ndx]
        wRF = upa.var_steervec(self.Nrx_y, self.Nrx_z, self.rx_bdir)
        self.eff_ch = np.array(self.h[:,:,0]).dot(self.tx_beam)
        self.rssi_val = np.sqrt(self.N_rx*self.N_tx)*np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]

        Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
        self.SNR_list = []
        self.SNR = Es * np.abs(self.rssi_val) ** 2 / (self.N0 * self.B)
        self.rate = np.log2(1 + self.SNR)  # in Gbit/s (self.B / self.nFFT) *
        self.best_rate = self.rate
        self.init_rate = self.rate
        self.mini_rate = self.rate
        self.beam_flag = 0.0
        self.SNR_list.append(self.SNR)
        # self.obs = np.array([[self.rssi_val.real, self.rssi_val.imag]])
        # self.obs = np.array([np.concatenate((np.array([self.rssi_val.real]), np.array([self.rssi_val.imag]),
        #                                     self.eff_ch.real.ravel(), self.eff_ch.imag.ravel()), axis=0)])
        self.norm_tx_xloc = np.array([(self.tx_loc[0][0]) / 1000])  # np.max(self.rx_xcov)])#np.array([(self.tx_loc[0][0]+np.max(self.rx_xcov))/(np.max(self.rx_xcov))])#-np.min(self.rx_xcov))])
        self.norm_tx_yloc = np.array([(self.tx_loc[0][1]) / 1000])  # max(np.max(self.rx_ycov),1)])#np.array([(self.tx_loc[0][1] + np.max(self.rx_ycov)) / (np.max(self.rx_ycov))])# - np.min(self.rx_ycov))])
        self.norm_tx_zloc = np.array([(self.tx_loc[0][2]) / 22.5])
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
        self.rbdir_count = 0
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
        for ndx in range(self.N_rx):
            #eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
            wRF = upa.var_steervec(self.Nrx_y, self.Nrx_z, self.BeamSet[ndx])
            rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
            Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
            SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
            rate = np.log2(1 + SNR)

            if rate > best_rate:
                best_rate = rate
                best_action_ndx = ndx
                best_rssi_val = rssi_val
        return best_rate, best_action_ndx, best_rssi_val

    def get_minmax_exhrate(self, loc_ndx, ch_randval):
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

        #if (self.dbp <= self.d_2d <= 5e3) and (self.ch_model == 'uma-los'):
        #    self.ch_model = self.init_ch_model + '-dbp'
        #else:
        #    self.ch_model = self.init_ch_model

        if self.ant_arr == 'ula':
            channel = channel_mmW.Channel(self.freq, tx_loc, self.rx_loc, self.sc_xyz, 'model', self.init_ch_model, 'nrx',
                                          self.N_rx,
                                          'ntx', self.N_tx, 'nFFT', self.nFFT, 'df', self.df)
            channel.generate_paths(ch_randval, loc_ndx)
            self.ch_model = channel.model_name

        if self.ant_arr == 'upa':
            channel = upa_channel_mmW.Channel(self.freq, tx_loc, self.rx_loc, self.sc_xyz, 'model', self.init_ch_model,
                                              'nrx_y', self.Nrx_y, 'nrx_z', self.Nrx_z, 'ntx_y', self.Ntx_y, 'ntx_y',
                                              self.Ntx_z, 'nFFT', self.nFFT, 'df', self.df)
            channel.generate_paths(ch_randval, loc_ndx)
            self.ch_model = channel.model_name
        # tx_num = self.get_txloc_ndx(self.tx_loc)
        # channel.generate_paths(ch_randval, tx_num)
        # npaths = self.channel.nlos_path + 1
        # h = self.channel.get_h()  # channel coefficient
        # self.cap = self.get_capacity()  # Compute capacity of channel for given location
        for txdir_ndx in range(self.N_tx):
            if self.ant_arr == 'ula':
                # snpaths = channel.nlos_path + 1
                h = channel.get_h()  # channel coefficient
                tx_bdir = self.BeamSet[txdir_ndx]  # self.channel.az_aod[0]#
                tx_beam = ula.var_steervec(self.N_tx, tx_bdir, 0)

            if self.ant_arr == 'upa':
                # self.npaths = self.channel.nlos_path + 1
                h = channel.get_h()  # channel coefficient
                tx_bdir = self.BeamSet[txdir_ndx]  # self.channel.az_aod[0]#
                tx_beam = upa.var_steervec(self.Ntx_y, self.Ntx_z, tx_bdir)

            eff_ch = np.array(h[:, :, 0]).dot(tx_beam)
            for rbdir_ndx in range(self.N_rx):
                if (self.ant_arr == 'ula'):
                    wRF = ula.var_steervec(self.N_rx, self.BeamSet[rbdir_ndx], 0)
                if (self.ant_arr == 'upa'):
                    wRF = upa.var_steervec(self.Nrx_x, self.Nrx_y, self.BeamSet[rbdir_ndx])
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


    def meas_rate(self):
        rssi_values = []
        rate_values = []
        for ndx in range(self.N_rx):
            #eff_ch = np.array(self.h[:, :, 0]).dot(self.tx_beam)
            wRF = upa.steervec(self.N_rx, self.N_rx, self.BeamSet[ndx], 0)
            rssi_val = np.sqrt(self.N_rx * self.N_tx) * np.array(np.conj(wRF.T).dot(self.eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
            Es = db2lin(self.P_tx)  # * (1e-3 / self.B)
            SNR = Es * np.abs(rssi_val) ** 2 / (self.N0 * self.B)
            rate = np.log2(1 + SNR)
            rssi_values.append(rssi_val)
            rate_values.append(rate)

        return rssi_values, rate_values

    def get_rssi_range(self):
        loc1 = np.array([[0,0,22.5]])
        sc = np.array([[650,300,61.5], [0,-550,41.5]])#np.array([[-100,50,21.5], [-100,-50,21.5], [-50,100,21.5],[50,100,21.5]])#, [-50,100,11.5],[50,100,11.5]]np.array([[50,0,0], [-50,-100,0], [100,50,0],[50,-100,0]])#np.array([])
        ch_model = 'uma-nlos'

        channel = Channel(self.freq, loc1, self.rx_loc, sc, 'model', ch_model, 'nrx_y', self.Nrx_y,'nrx_z', self.Nrx_z,
                               'ntx_y', self.Ntx_y,'ntx_z', self.Ntx_z, 'nFFT', self.nFFT, 'df', self.df)

        tx_num = self.get_txloc_ndx(loc1)
        channel.generate_paths(np.exp(1j * 2 * np.pi * 0.6), tx_num) #0.3 is some defined default value

        h = channel.get_h()

        max_rssi = 0
        for tx_dir_ndx in range(self.obs_space.nvec[3]):
            tx_dir = self.BeamSet[tx_dir_ndx]
            tx_beam = upa.var_steervec(self.Ntx_y, self.Ntx_z, tx_dir)
            eff_ch = np.array(h[:, :, 0]).dot(tx_beam)
            for rx_dir_ndx in range(self.action_space.n):
                rx_dir = self.BeamSet[rx_dir_ndx]
                wRF = upa.var_steervec(self.Nrx_y, self.Nrx_z,  rx_dir)
                rssi_val = np.sqrt(self.N_tx*self.N_rx) * np.array(np.conj(wRF.T).dot(eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
                if(np.abs(max_rssi)**2 < np.abs(rssi_val)**2):
                    max_rssi = rssi_val

        loc2 = np.array([[600, 600, 22.5]])
        sc = np.array([[650,300,61.5], [0,-550,41.5]])#np.array([[-100,50,21.5], [-100,-50,21.5], [-50,100,21.5],[50,100,21.5]])#np.array([[-100,50,11.5], [-100,-50,11.5], [-50,100,11.5],[50,100,11.5]])#np.array([[50,0,0], [-50,-100,0], [100,50,0],[50,-100,0]])#np.array([])
        ch_model = 'uma-nlos'#'uma-nlos'

        channel = Channel(self.freq, loc2, self.rx_loc, sc, 'model', ch_model, 'nrx_y', self.Nrx_y,'nrx_z', self.Nrx_z,
                          'ntx_y', self.Ntx_y, 'ntx_z', self.Ntx_z,'nFFT', self.nFFT, 'df', self.df)

        tx_num = self.get_txloc_ndx(loc2)
        channel.generate_paths(np.exp(1j * 2 * np.pi * 0.6), tx_num)
        h = channel.get_h()

        min_rssi = 1e20
        for tx_dir_ndx in range(self.obs_space.nvec[3]):
            tx_dir = self.BeamSet[tx_dir_ndx]
            tx_beam = upa.var_steervec(self.Ntx_y, self.Ntx_y, tx_dir)
            eff_ch = np.array(h[:, :, 0]).dot(tx_beam)
            for rx_dir_ndx in range(self.action_space.n):
                rx_dir = self.BeamSet[rx_dir_ndx]
                wRF = upa.var_steervec(self.Nrx_y, self.Nrx_z, rx_dir)
                rssi_val = np.sqrt(self.N_rx*self.N_tx) * np.array(np.conj(wRF.T).dot(eff_ch)) #+ (np.conj(wRF.T).dot(self.noise))[0]
                if (np.abs(min_rssi) ** 2 > np.abs(rssi_val) ** 2):
                    min_rssi = rssi_val

        return min_rssi, max_rssi

    def get_reward_goal(self, rssi_val):
        #transmission energy
        Es = db2lin(self.P_tx) #* (1e-3 / self.B)
        self.SNR = Es * np.abs(rssi_val)**2 / (self.N0*self.B)
        rate = np.log2(1 + self.SNR)  # in Gbit/s (self.B / self.nFFT) *
        self.SNR_list.append(self.SNR)
        rwd = 0.0#(float(np.around(rate-self.best_rate,decimals=2)))
        done = False
        init_flag = False

        #if ((self.rbdir_count) == self.goal_steps): #or ((rate == self.rate) and self.rbdir_count >1):  # ((rate >= self.rate) and (self.rbdir_count ==2)) or
        #    done = True

        #if (rate >= (self.best_rate)) and (self.rbdir_count == 1):
        #    rwd = 1.0
        #if (rate < (self.best_rate)) and (self.rbdir_count == 1):
        #if  (rate >= (self.best_rate)) and (self.rbdir_count==1):  #(rate > self.rate) and and (self.rbdir_count < 2)
        #    rwd = 2.0#float(np.around(rate, decimals=4))#1.0  # float(np.round(rate))
        #    self.best_rate = rate
        #    init_flag = True
        #if(self.rbdir_count == 1):
        #    self.init_rate = rate

        if (rate < (self.best_rate-0.7)) and (self.rbdir_count == 1): #init_rate < best_rate
            rwd = -1.0#*(self.time)#*float(np.around(rate, decimals=4))

        #if (rate < self.init_rate):
        #    rwd = -1.0
        #if (rate > self.init_rate):
        #    rwd= 1.0

        if ((self.best_rate-rate) <= 0.7):#(rate >= self.best_rate):#(((self.best_rate -0.2)<=rate) and (rate <= self.best_rate)):
            rwd = 1.0#*(self.time)

        if (rate > self.best_rate):
        #    self.best_rate = rate
            init_flag = True
            #self.beam_flag = True
        #if (rate < self.rate)or ((rate > self.rate) and (rate < self.best_rate)): # and (rate < self.best_rate)
        #    rwd = 0.0#*float(np.round(rate))#-1.0
        #if (rate >= (self.best_rate)) and (self.rbdir_count == 1):
        #    rwd = 1.0


        #if (rate > self.init_rate):
        #else:
        #    rwd = np.float(np.around(rate-self.best_rate, decimals=4))

        self.rate = rate
        # print(rwd)
        return rwd, done, init_flag

    def get_reward(self):

        best_eprate = max(max(self.ep_rates), self.best_rate)

        #if(self.ep_rates[self.dqn_action] == best_eprate) and not(self.best_rate == -np.inf):
        #    rwd = 1.0
        #else:
        #    rwd = -1.0
        ep_rwds = []
        for i in range(len(self.ep_actions)):
            if (self.ep_rates[self.ep_actions[i]] == best_eprate) and (i==0) and not (self.best_rate == -np.inf):
                rwd = 1.0
            elif (self.ep_rates[self.ep_actions[i]] == best_eprate) and (i>0):
                rwd = 1.0
            else:
                rwd = -1.0
            ep_rwds.append(rwd)

        best_action = self.ep_rates.index(max(self.ep_rates))
        if(self.ep_rates[best_action] > self.best_rate):
            self.dqnbestbeam_ndxlist[self.tx_num] = best_action

        #nbgh_list = [self.ep_rates[i] for i in range(len(self.ep_rates)) if not (i == self.dqn_action)]
        #if (max(nbgh_list) == best_eprate):
        #    self.nbgh_action = True
        #rwd_vals = [1.0 if ((x == best_eprate) and (x >= self.best_rate)) else 0.0 for x in self.ep_rates]
        #print("ep_rates: {} , reward values: {}".format(self.ep_rates, rwd_vals))
        #return rwd_vals[action_ndx]
        return ep_rwds#rwd

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

    def get_obs_sample(self, episode_num):
        ndx = int(episode_num % (len(self.tx_locs)*self.obs_space.nvec[3]))
        tx_dir_ndx = int(ndx % self.obs_space.nvec[3])
        tx_num = int(ndx / self.obs_space.nvec[3])
        return tx_num, tx_dir_ndx
