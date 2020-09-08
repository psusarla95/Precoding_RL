import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
from Source.antenna import ula
from Source.misc_fun.channel_mmW import Channel
from Source.misc_fun.conversion import db2lin

'''

###################
Custom Space Mapping
####################

Example:
actions = {
    ['RBS'] :[0,2,1],
    ['TBS'] :[0,2,1]
    }
Custom_Space_Mapping(actions) =
    { 0:[0,0],1:[0,1],2:[0,2],3:[1,0],4:[1,1], 5:[1,2], 6:[2,0], 7:[2,1], 8: [2,2]}

'''

def Custom_Space_Mapping(actions):

    parameter_count = len(actions.keys())
    parameter_list = []
    for key in actions.keys():
        par_range = actions[key]#[actions.keys[i]]
        parameter_list.append(list(range(par_range[0],par_range[1]+1,par_range[2])))


    #creates a list of all possible tuples from given lists of action values
    action_val_tuples = [list(x) for x in np.array(np.meshgrid(*parameter_list)).T.reshape(-1,len(parameter_list))]
    action_key_list = list(np.arange(len(action_val_tuples)))

    action_values = dict(zip(action_key_list,action_val_tuples))

    return action_values


def plot(values, moving_avg_period, test_values):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    #plt.plot(test_values)
    plt.pause(0.001)

    #plt.plot(test_values)
    #plt.pause(0.001)
    print("Episode", len(values), "\t", moving_avg_period, "episode moving avg:", moving_avg[-1], end="\r")
    #if is_ipython: display.clear_output(wait=True)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))#torch.tensor([-4.0 for i in range(period-1)])
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))#torch.tensor([-4.0 for i in range(len(values))])#torch.zeros(len(values))
        return moving_avg.numpy()


def plotbeam(ang, n):
    w = ula.steervec(n, ang, 0)#np.array(array_factor(d,ang, n))
    #print(w.shape)
    wh = w.T.conj()
    r = np.arange(0, 1, 0.001)
    theta = 2* np.pi * r
    #wh= wh.reshape(,)
    #print(wh, w)
    gr = np.abs(np.array([wh.dot(ula.steervec(n, x, 0)) for x in theta]))#ula.steervec(n, x, 0)
    #print("gr:{0}".format(gr))
    #ax = plt.subplot(111, projection='polar')
    ##print(theta.shape, gr.shape)
    #ax.plot(theta, gr)
    #plt.show()
    return theta, gr

def var_plotbeam(beam_val, n):
    ang, active_n = beam_val
    w = ula.var_steervec(n, beam_val, 0)#np.array(array_factor(d,ang, n))
    #print(w.shape)
    wh = w.T.conj()
    r = np.arange(0, 1, 0.001)
    theta = 2* np.pi * r
    #wh= wh.reshape(,)
    #print(wh, w)
    gr = np.abs(np.array([wh.dot(ula.var_steervec(n,(x,active_n), 0)) for x in theta]))#ula.steervec(n, x, 0)
    #print("gr:{0}".format(gr))
    #ax = plt.subplot(111, projection='polar')
    ##print(theta.shape, gr.shape)
    #ax.plot(theta, gr)
    #plt.show()
    return theta, gr

'''
def plotbeam(ang, n):
    w = ula.steervec(n, ang, 0)#np.array(array_factor(d,ang, n))
    #print(w.shape)
    wh = w.T.conj()
    r = np.arange(0, 1, 0.001)
    theta = 2* np.pi * r
    #wh= wh.reshape(,)
    #print(wh, w)
    gr = np.abs(np.array([wh.dot(ula.steervec(n, x, 0)) for x in theta]))#ula.steervec(n, x, 0)
    #print("gr:{0}".format(gr))
    #ax = plt.subplot(111, projection='polar')
    ##print(theta.shape, gr.shape)
    #ax.plot(theta, gr)
    #plt.show()
    return theta, gr
'''

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

def Generate_Beams(N,width_vec):
    BeamSet = []
    min_ang = 0
    max_ang = np.pi
    #print("width_vec: ", width_vec)
    for bm_wdth in width_vec:
        beam_angles = np.arange(min_ang + bm_wdth, max_ang + bm_wdth, bm_wdth)
        for i in range(len(beam_angles)):
            n_ant = min(int(np.pi/bm_wdth), N)
            BeamSet.append((np.arctan2(np.sin(beam_angles[i]), np.cos(beam_angles[i])), n_ant)) #(beamdir, active_antennas)
    dt = np.dtype('float,int')
    return np.array(BeamSet, dtype=dt)

def Generate_BeamDirs(N,refine_levels):
    BeamSet = []
    min_ang = 0
    max_ang = np.pi
    bm_wdth = np.pi/N
    #print("width_vec: ", width_vec)
    beam_angles = np.arange(min_ang + bm_wdth, max_ang + bm_wdth, bm_wdth)
    for bm_angle in beam_angles:
        for level in refine_levels:
            if level == 0:
                BeamSet.append(np.around(np.arctan2(np.sin(bm_angle), np.cos(bm_angle)), decimals=2))
            else:
                angle1 = (bm_angle + bm_wdth/(2**level))% np.pi
                angle2 = (bm_angle - bm_wdth/(2**level))% np.pi
                radian_angle1 = np.around(np.arctan2(np.sin(angle1), np.cos(angle1)), decimals=2)
                radian_angle2 = np.around(np.arctan2(np.sin(angle2), np.cos(angle2)), decimals=2)
                if (radian_angle1 not in BeamSet):
                    BeamSet.append(radian_angle1)
                if (radian_angle2 not in BeamSet):
                    BeamSet.append(radian_angle2)
    return np.array(BeamSet)


def All_Exhaustive_RateMeas(tx_locs, N_tx, N_rx):

    # print(len(tx_locs))
    tx_beams = Generate_BeamDir(N_tx)
    rx_beams = Generate_BeamDir(N_rx)

    freq =  30e9
    rx_loc = np.array([[0,0,0]])
    sc_xyz = np.array([[0,150,0],[250,50,0],[-200,-150,0]])
    ch_model = 'uma-nlos'
    nFFT = 1
    df = 60*1e3
    P_tx = 30 #dB
    B = nFFT * df
    N0dBm = -174  # mW/Hz
    N0 = db2lin(N0dBm) * (10 ** -3)  # in WHz-1
    gau = np.zeros((N_rx, 1), dtype=np.complex)
    for i in range(gau.shape[0]):
        gau[i] = complex(np.random.randn(), np.random.randn())
    noise = np.sqrt(N0 / 2) * gau

    verts =[]
    for loc in tx_locs:
        if (np.all(loc == [0, 0, 0])):
            loc = np.array([[40, 40, 0]])

        channel = Channel(freq, loc, rx_loc, sc_xyz, 'model', ch_model, 'nrx', N_rx,'ntx', N_tx, 'nFFT', nFFT, 'df', df)
        channel.generate_paths()
        h = channel.get_h()

        beam_pairs = []
        rate_val = []
        for tx_bdir in tx_beams:
            for rx_bdir in rx_beams:

                bpair = "(" + str(tx_bdir*(180/np.pi)) + "," + str(rx_bdir*(180/np.pi)) + ")"
                beam_pairs.append(bpair)

                fRF = ula.steervec(N_tx, tx_bdir, 0)
                wRF = ula.steervec(N_rx, rx_bdir, 0)
                rssi_val = np.sqrt(N_rx * N_tx) * np.array(np.conj(wRF.T).dot(h[:, :, 0])).dot(fRF) + \
                           (np.conj(wRF.T).dot(noise))[0]
                Es = db2lin(P_tx)  # * (1e-3 / self.B)
                SNR = Es * np.abs(rssi_val) ** 2 / (N0 * B)
                rate = np.log2(1 + SNR)
                rate_val.append(rate)

        beam_pairs = np.array(beam_pairs)
        rate_val = np.array(rate_val)

        verts.append([*zip(np.arange(1,len(beam_pairs)+1), rate_val)])

    return verts