import numpy as np
import cmath
import math

def db2pow(x):
    y = 10 ** (x / 10)
    return (y)

# conversion from dB to linear
def db2lin(val_db):
    val_lin = 10 ** (val_db / 10)
    return val_lin

# cosine over degree
def cosd(val):
    return cmath.cos(val * math.pi / 180)

# sine over degree
def sind(val):
    return cmath.sin(val * math.pi / 180)

# asin in degree
def asind(val):
    #return 180/pi*math.asin(val)
    #return np.degrees(np.sinh(val))
    c1 = cmath.asin(val)
    c2 = complex(math.degrees(c1.real), math.degrees(c1.imag))
    return c2

#deg2rad for complex number
def deg2rad(val):
    l = [val.real*cmath.pi/180, val.imag*cmath.pi/180]
    c1 = complex(np.around(l[0], decimals=4), np.around(l[1], decimals=4))
    return c1

# acosine in degree
def acosd(val):
    return np.degrees(np.sinh(val))


def search_max(x, kmax):
    m = []
    value = []
    for k in range(0, kmax):
        v = np.argmax(x)
        value = np.append(value, x[v])
        x[v] = -np.Inf
        m = np.append(m, int(v))

    return np.array(m), np.array(value)
