#from scipy.constants import speed_of_light
from Source.antenna import upa
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
                    self.model = {'alpha': 2., 'beta': 31.4, 'gamma': 2.1, 'sigma': 2.9}
                elif varargin[n + 1] == 'umi-os-los':
                    self.model = {'alpha': 2.6, 'beta': 24., 'gamma': 1.6, 'sigma': 4.}
                elif varargin[n + 1] == 'uma-los':
                    self.model = {'alpha':2.2 , 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0} #sigma-2.8, 4.1
                elif varargin[n + 1] == 'uma-los-dbp':
                    self.model = {'alpha': 4.0, 'beta': 28.0, 'gamma': 2.0, 'sigma': 0.0}  # sigma-2.8, 4.1
                elif varargin[n + 1] == 'uma-nlos':
                    self.model = {'alpha': 3.0, 'beta': 32.4, 'gamma': 2.0, 'sigma': 0.0}  #7.8beta- 17.6, sigma-9.9 2.8
                elif varargin[n + 1] == 'fsp':
                    self.model = {'alpha': 2., 'beta': 32.4478, 'gamma': 2, 'sigma': 4.}#32.4478
                elif varargin[n + 1] == 'fsp-nlos':
                    self.model = {'alpha': 2.2, 'beta': 32.4478, 'gamma': 2, 'sigma': 4.}#32.4478
                self.model_name = varargin[n + 1]
            elif varargin[n] == 'nrx_y':
                self.nrx_y = varargin[n+1]
            elif varargin[n] == 'nrx_z':
                self.nrx_z = varargin[n+1]
            elif varargin[n] == 'ntx_y':
                self.ntx_y = varargin[n+1]
            elif varargin[n] == 'ntx_z':
                self.ntx_z = varargin[n+1]
            elif varargin[n] == 'nFFT':
                self.nFFT = varargin[n+1]
            elif varargin[n] == 'df':
                self.df = varargin[n + 1]

    def ploss(self, d, tx_num):
        np.random.seed(tx_num)
        rho = 10 * self.model['alpha'] * np.log10(d) + self.model['beta'] \
              + 10 * self.model['gamma'] * np.log10(self.fc / 1e9) + self.model['sigma'] * np.random.randn(d.shape[0],
                                                                                                          d.shape[1])

        #20 * np.log10(self.Dist) + 20 * np.log10(self.freq) - 147.55
        return rho

    def generate_paths(self, ch_randval, tx_num):

        d = np.linalg.norm(self.tx - self.rx)
        self.pathloss[0] = self.ploss(d.reshape(1, 1), tx_num)
        if (self.model_name == 'uma-los-dbp'):
            dbp = 4*self.tx[0,2]*self.rx[0,2]*self.fc/self.c
            self.pathloss[0] -= 9*np.log10(dbp**2 + (self.tx[0,2]-self.rx[0,2])**2)

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
            d = np.linalg.norm(self.tx - self.sc[n]) + np.linalg.norm(self.rx - self.sc[n])
            self.pathloss[n + 1] = self.ploss(d.reshape(1, 1), tx_num)
            self.coeff[n + 1] = np.sqrt(db2pow(-self.pathloss[n]))*np.exp(1j * 2 * np.pi * 0.6)#ch_randval#* np.exp(1j * 2 * np.pi * 0.3)#np.random.rand()
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
        h = np.zeros([self.nrx_y*self.nrx_z, self.ntx_y*self.ntx_z, self.nFFT], dtype=complex)
        npaths = self.nlos_path + 1
        for n in range(0, npaths):
            gd = 1 / np.sqrt(self.nFFT) * np.exp(-1j * 2 * np.pi * np.arange(0, self.nFFT) * self.tau[n] * self.df)
            gr = upa.var_steervec(self.nrx_y, self.nrx_z, (self.az_aoa[n], self.nrx_y, self.el_aoa[n], self.nrx_z))
            gt = upa.var_steervec(self.ntx_y, self.ntx_z, (self.az_aod[n], self.ntx_y, self.el_aod[n], self.ntx_z))
            ha = self.coeff[n] * np.outer(gr, gt.conj())
            #print(ha.shape)
            #print(gd)
            for nc in range(0, self.nFFT):
                h[:, :, nc] += ha * gd[nc]
        return h

    def get_h_tx(self):
        h = np.zeros([1, self.ntx_y*self.ntx_z, self.nFFT], dtype=complex)
        npaths = self.nlos_path + 1

        for n in range(0, npaths):
            gd = 1 / np.sqrt(self.nFFT) * np.exp(-1j * 2 * np.pi * np.arange(0, self.nFFT) * self.tau[n] * self.df)
            gr = 1
            gt = upa.var_steervec(self.ntx_y, self.ntx_z, (self.az_aod[n], self.ntx_y, self.el_aod[n], self.ntx_z))
            ha = self.coeff[n] * np.outer(gr, gt.conj())
            for nc in range(0, self.nFFT):
                h[:, :, nc] += ha * gd[nc]
        return h


    def get_h_rx(self):
        h = np.zeros([self.nrx_y*self.nrx_z, 1, self.nFFT], dtype=complex)
        npaths = self.nlos_path + 1
        for n in range(0, npaths):
            gd = 1 / np.sqrt(self.nFFT) * np.exp(-1j * 2 * np.pi * np.arange(0, self.nFFT) * self.tau[n] * self.df)
            gr = upa.var_steervec(self.nrx_y, self.nrx_z, (self.az_aoa[n], self.nrx_y, self.el_aoa[n], self.nrx_z))
            gt = upa.var_steervec(1, 1, (self.az_aoa[n], 1, self.el_aoa[n], 1))
            ha = self.coeff[n] * np.outer(gr, gt.conj())
            for nc in range(0, self.nFFT):
                h[:, :, nc] += ha * gd[nc]
        return h