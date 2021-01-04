import numpy as np


def steervec(n_x, n_y, beam_angles):
    theta, phi = beam_angles
    #P = np.zeros([3, ntx])
    #P[0, :] = np.arange(0, ntx)
    #P[1, :] = np.zeros(ntx)
    #P[2, :] = np.zeros(ntx)

    #theta is the elevation angle, phi is the azimuthal angle
    #kv = np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)])
    #kv = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)])
    #kv.reshape(3, 1)
    #v = 1 / np.sqrt(ntx) * np.exp(1j * 2 * np.pi * 0.5 * (P.transpose()).dot(kv))

    P_x =np.exp(1j*2*np.pi*0.5*(np.arange(0,n_x)*np.cos(theta)*np.sin(phi)))
    P_y= np.exp(1j*2*np.pi*0.5*(np.arange(0,n_y)*np.sin(theta)*np.sin(phi)))
    P_z = np.exp(1j * 2 * np.pi * 0.5 * (np.arange(0, n_y) * np.cos(phi)))
    v = (1/np.sqrt(n_x*n_y))*np.kron(P_x,P_y)
    return v


def var_steervec(n_x, n_y, beam_vals):
    #print(beam_vals, type(beam_vals))
    theta, active_antx, phi, active_anty = beam_vals
   # P = np.zeros([3, ntx])
   # P[0, :] = np.zeros(ntx)#np.arange(0, ntx)
   # P[1, :] = np.zeros(ntx)
   # P[2, :] = np.zeros(ntx)

    #kv = np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)])
    #kv.reshape(3, 1)

    P_x =np.exp(1j*2*np.pi*0.5*(np.arange(0,n_x)*np.cos(theta)*np.sin(phi)))
    P_y= np.exp(1j*2*np.pi*0.5*(np.arange(0,n_y)*np.sin(theta)*np.sin(phi)))
    P_z = np.exp(1j * 2 * np.pi * 0.5 * (np.arange(0, n_y) * np.cos(phi)))
    v = (1/np.sqrt(n_x*n_y))*np.kron(P_x,P_y)
    #v = 1 / np.sqrt(ntx) * np.exp(1j * 2 * np.pi * 0.5 * (P.transpose()).dot(kv))
    #v = np.zeros(ntx, dtype=np.complex)
    #print(P[0:])
    #for i in range(min(ntx,active_ant)):
    #    P[0, i] = (i % active_ant)
        #print(P[i,:])
        #for j in range(0, ant_factor):
    #    v[i] = 1 / np.sqrt(ntx) * np.exp(1j * 2 * np.pi * 0.5 * (P[:,i].transpose()).dot(kv))
    return v
