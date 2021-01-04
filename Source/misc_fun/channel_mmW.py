#from scipy.constants import speed_of_light
from Source.antenna import ula
from Source.misc_fun.conversion import *
from Source.misc_fun.geometry import *
import numpy as np
from scipy.stats import rice

class Channel:
    
    ntx: int  # number of transmit antennas
    nrx: int  # number of receive antennas
    nFFT: int  # number of OFDM subcarriers

    def __init__(self, fc, tx_xyz, rx_xyz, sc_xyz, *args):
        """
        :param fc: carrier frequency
        :param args: variable list of argument
        """
        self.c = 3e8
        self.lam = self.c / fc
        self.fc = fc
        self.tx = tx_xyz
        self.rx = rx_xyz
        self.sc = sc_xyz
        self.dbp = 4 * self.tx[0, 2] * self.rx[0, 2] * self.fc / self.c
        self.d_2d = np.linalg.norm(np.array([[self.tx[0,0], self.tx[0,1], 0]]) - np.array([[self.rx[0,0], self.rx[0,1], 0]]))

        self.nlos_path = self.sc.shape[0]
        self.pathloss = np.zeros(self.nlos_path + 1)
        self.coeff = np.zeros(self.nlos_path + 1, dtype=complex)
        self.tau = np.zeros(self.nlos_path + 1)
        self.az_aoa = np.zeros(self.nlos_path + 1)
        self.el_aoa = np.zeros(self.nlos_path + 1)
        self.az_aod = np.zeros(self.nlos_path + 1)
        self.el_aod = np.zeros(self.nlos_path + 1)

        varargin = args
        nargin = len(args)
        self.model_name = ''


        for n in range(0, nargin, 2):
            if varargin[n] == 'model':
                if varargin[n + 1] == 'umi-sc-los':
                    self.model = {'alpha': 2., 'beta': 31.4, 'gamma': 2.1, 'sigma': 2.9,'k': 0.0}
                elif varargin[n + 1] == 'umi-os-los':
                    self.model = {'alpha': 2.6, 'beta': 24., 'gamma': 1.6, 'sigma': 4.,'k': 0.0}
                elif varargin[n + 1] == 'uma-av-los':
                    self.model = {'alpha':2.2 , 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0,'k': 0.0} #sigma-2.8, 4.1
                elif varargin[n + 1] == 'uma-los':
                    self.model = {'alpha':2.2 , 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0,'k': 0.0} #sigma-2.8, 4.1
                elif varargin[n + 1] == 'uma-los-dbp':
                    self.model = {'alpha': 4.0, 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0,'k': -9*np.log10(self.dbp**2 + (self.tx[0,2]-self.rx[0,2])**2)}  # sigma-2.8, 4.1
                elif varargin[n + 1] == 'uma-nlos':
                    self.model = {'alpha': 3.0, 'beta': 32.4, 'gamma': 2.0, 'sigma': 0.0,'k': 0.0}  #7.8beta- 17.6, sigma-9.9 2.8
                elif varargin[n + 1] == 'uma-av-nlos':
                    self.model = {'alpha': (4.6-0.7*np.log10(self.tx[0,2])), 'beta': -17.5, 'gamma': 2.0, 'sigma': 0.0,'k': 20*np.log10(40*np.pi/3)}  #7.8beta- 17.6, sigma-9.9 2.8
                elif varargin[n + 1] == 'fsp':
                    self.model = {'alpha': 2., 'beta': 32.4478, 'gamma': 2, 'sigma': 4.,'k': 0.0}#32.4478
                elif varargin[n + 1] == 'fsp-nlos':
                    self.model = {'alpha': 2.2, 'beta': 32.4478, 'gamma': 2, 'sigma': 4.,'k': 0.0}#32.4478
                self.model_name = varargin[n + 1]
            elif varargin[n] == 'nrx':
                self.nrx = varargin[n+1]
            elif varargin[n] == 'ntx':
                self.ntx = varargin[n+1]
            elif varargin[n] == 'nFFT':
                self.nFFT = varargin[n+1]
            elif varargin[n] == 'df':
                self.df = varargin[n + 1]

    def ploss(self, d, model, tx_num):
        np.random.seed(tx_num)
        rho = 10 * model['alpha'] * np.log10(d) + model['beta'] \
              + 10 * model['gamma'] * np.log10(self.fc / 1e9) \
              + model['sigma'] * np.random.randn(d.shape[0],d.shape[1]) + model['k']
        #20 * np.log10(self.Dist) + 20 * np.log10(self.freq) - 147.55
        return rho

    def set_model(self, tx_num, sc_ndx=None):
        #print(self.model_name, self.d_2d, self.tx[0,2])
        #uma-los with UAV-AV
        if (self.model_name == 'uma-los') and (self.d_2d <=4e3) and (22.5 < self.tx[0,2] <=300):
            self.model_name = 'uma-av-los'
            self.model = {'alpha': 2.2, 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0, 'k': 0.0}  #sigma: 4.64*np.exp(-0.0066*self.tx[0,2])

        #uma-los within breakpoitn distance
        elif(self.model_name == 'uma-los') and (10< self.d_2d <= self.dbp) and (1.5 < self.tx[0,2] <=22.5):
            self.model_name = 'uma-los'
            self.model = {'alpha': 2.2, 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0, 'k': 0.0}  #sigma:4.0

        #uma-los after breakpoint distance
        elif (self.model_name == 'uma-los') and (self.dbp< self.d_2d <= 5e3) and (1.5 < self.tx[0,2] <=22.5):
            self.model_name = 'uma-los-dbp'
            self.model = {'alpha': 4.0, 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0,
                      'k': -9 * np.log10(self.dbp ** 2 + (self.tx[0, 2] - self.rx[0, 2]) ** 2)} #sigma:4.0

        #uma-nlos with UAV-AV
        elif (self.model_name == 'uma-nlos') and (self.d_2d <= 4e3) and (10 < self.tx[0,2] <=100):
            self.model_name = 'uma-av-nlos'
            self.model = {'alpha': (4.6 - 0.7 * np.log10(self.tx[0, 2])), 'beta': -17.5, 'gamma': 2.0, 'sigma': 0.0,
                      'k': 20 * np.log10(40 * np.pi / 3)}  # 7.8beta- 17.6, sigma-9.9 2.8
            #print("came here")
        #uma-nlos with terrestial UAV
        elif (self.model_name == 'uma-nlos') and (1.5 < self.tx[0,2] <=22.5):
            if (10< self.d_2d <= self.dbp):
                los_model = {'alpha': 2.2, 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0, 'k': 0.0} #sigma:4.0
            else:
                los_model = {'alpha': 4.0, 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0,
                      'k': -9 * np.log10(self.dbp ** 2 + (self.tx[0, 2] - self.rx[0, 2]) ** 2)} #sigma:4.0
            d = np.linalg.norm(self.tx - self.rx)
            los_pathloss = self.ploss(d.reshape(1,1), los_model, tx_num)

            nlos_model = {'alpha': 3.908, 'beta': 13.54, 'gamma': 2.0, 'sigma': 0.0,
                      'k': -0.6*(self.tx[0, 2]-1.5)} #sigma:6.0
            d = np.linalg.norm(self.tx - self.sc[sc_ndx]) + np.linalg.norm(self.rx - self.sc[sc_ndx])
            nlos_pathloss = self.ploss(d.reshape(1, 1), nlos_model, tx_num)

            self.model_name = 'uma-nlos'
            if (los_pathloss > nlos_pathloss):
                self.model = los_model
            else:
                self.model = nlos_model
        else:
            print("Unrecognized pathloss model")
        return

    def generate_paths(self, ch_randval, tx_num):
        #choosing the right model
        self.model_name = 'uma-los'
        self.set_model(tx_num)
        #print("los model selection done")
        d = np.linalg.norm(self.tx - self.rx)
        self.pathloss[0] = self.ploss(d.reshape(1, 1), self.model, tx_num)
        #if (self.model_name == 'uma-los-dbp'):
        #     dbp = 4 * self.tx[0, 2] * self.rx[0, 2] * self.fc / self.c
        #    self.pathloss[0] -= 9*np.log10(dbp**2 + (self.tx[0,2]-self.rx[0,2])**2)

        self.coeff[0] = np.sqrt(db2pow(-self.pathloss[0]))*np.exp(1j * 2 * np.pi * 0.6)#ch_randval#np.random.rand()np.exp(1j * 2 * 0.6)#
        self.tau[0] = d/self.c
        (self.az_aoa[0], self.el_aoa[0], temp) = cart2sph(self.tx[0, 0] - self.rx[0, 0], self.tx[0, 1] -
                                                          self.rx[0, 1], self.tx[0, 2] - self.rx[0, 2])
        (self.az_aod[0], self.el_aod[0], temp) = cart2sph(self.rx[0, 0] - self.tx[0, 0], self.rx[0, 1] -
                                                          self.tx[0, 1], self.rx[0, 2] - self.tx[0, 2])

        for n in range(0, self.nlos_path):
            '''
            d = np.linalg.norm(self.tx - self.sc[n]) + np.linalg.norm(self.rx - self.sc[n])
            self.pathloss[n + 1] = self.ploss(d.reshape(1, 1))
            self.coeff[n + 1] = np.sqrt(db2pow(-self.pathloss[n])) * np.exp(1j * 2 * np.pi * np.random.rand())
            self.tau[n + 1] = d / speed_of_light
            # AOD
            (self.az_aod[n + 1], self.el_aod[n + 1], temp) = cart2sph(self.sc[n, 0] - self.rx[0, 0],
                                                                      self.sc[n, 1] - self.rx[0, 1],
                                                                      self.sc[n, 2] - self.rx[0, 2])
            # AOA
            (self.az_aoa[n + 1], self.el_aoa[n + 1], temp) = cart2sph(self.sc[n, 0] - self.tx[0, 0],
                                                                      self.sc[n, 1] - self.tx[0, 1],
                                                                      self.sc[n, 2] - self.tx[0, 2])
            '''
            self.model_name = 'uma-nlos'
            self.dbp = (4 * self.tx[0, 2] * self.sc[n, 2] * self.fc / self.c) + (4 * self.rx[0, 2] * self.sc[n, 2] * self.fc / self.c)
            self.d_2d = np.linalg.norm(np.array([[self.tx[0, 0], self.tx[0, 1], 0]]) - np.array([[self.sc[n, 0], self.sc[n, 1], 0]])) + \
                        np.linalg.norm(np.array([[self.rx[0, 0], self.rx[0, 1], 0]]) - np.array([[self.sc[n, 0], self.sc[n, 1], 0]]))
           #print("nlos started")
            self.set_model(tx_num,n)
            #print("nlospath: {}, modelname: {}, sc_ndx: {}, dbp: {}, d_2d: {}, tx ht: {}".format(self.nlos_path, self.model_name, n,
            #                                                                        self.dbp, self.d_2d, self.tx[0,2]))
            #print("came here")
            d = np.linalg.norm(self.tx - self.sc[n]) + np.linalg.norm(self.rx - self.sc[n])
            self.pathloss[n + 1] = self.ploss(d.reshape(1, 1), self.model,tx_num)
            self.coeff[n + 1] = np.sqrt(db2pow(-self.pathloss[n+1]))*np.exp(1j * 2 * np.pi * 0.6)#ch_randval#* np.exp(1j * 2 * np.pi * 0.3)#np.random.rand()
            self.tau[n + 1] = d / self.c
            # AOD
            (self.az_aod[n + 1], self.el_aod[n + 1], temp) = cart2sph(self.sc[n, 0] - self.tx[0, 0],
                                                                      self.sc[n, 1] - self.tx[0, 1],
                                                                      self.sc[n, 2] - self.tx[0, 2])
            # AOA
            (self.az_aoa[n + 1], self.el_aoa[n + 1], temp) = cart2sph(self.sc[n, 0] - self.rx[0, 0],
                                                                      self.sc[n, 1] - self.rx[0, 1],
                                                                      self.sc[n, 2] - self.rx[0, 2])

    def get_h(self):
        h = np.zeros([self.nrx, self.ntx, self.nFFT], dtype=complex)
        npaths = self.nlos_path + 1
        for n in range(0, npaths):
            gd = 1 / np.sqrt(self.nFFT) * np.exp(-1j * 2 * np.pi * np.arange(0, self.nFFT) * self.tau[n] * self.df)
            gr = ula.steervec(self.nrx, self.az_aoa[n], self.el_aoa[n])
            gt = ula.steervec(self.ntx, self.az_aod[n], self.el_aod[n])
            ha = self.coeff[n] * np.outer(gr, gt.conj())

            for nc in range(0, self.nFFT):
                h[:, :, nc] += ha * gd[nc]
        return h

    def get_h_tx(self):
        h = np.zeros([1, self.ntx, self.nFFT], dtype=complex)
        npaths = self.nlos_path + 1

        for n in range(0, npaths):
            gd = 1 / np.sqrt(self.nFFT) * np.exp(-1j * 2 * np.pi * np.arange(0, self.nFFT) * self.tau[n] * self.df)
            gr = 1
            gt = ula.steervec(self.ntx, self.az_aod[n], self.el_aod[n])
            ha = self.coeff[n] * np.outer(gr, gt.conj())
            for nc in range(0, self.nFFT):
                h[:, :, nc] += ha * gd[nc]
        return h


    def get_h_rx(self):
        h = np.zeros([self.nrx, 1, self.nFFT], dtype=complex)
        npaths = self.nlos_path + 1
        for n in range(0, npaths):
            gd = 1 / np.sqrt(self.nFFT) * np.exp(-1j * 2 * np.pi * np.arange(0, self.nFFT) * self.tau[n] * self.df)
            gr = ula.steervec(self.nrx, self.az_aoa[n], self.el_aoa[n])
            gt = ula.steervec(1, self.az_aoa[n], self.el_aoa[n])
            ha = self.coeff[n] * np.outer(gr, gt.conj())
            for nc in range(0, self.nFFT):
                h[:, :, nc] += ha * gd[nc]
        return h