import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from ref_file2.ioq5_model import IPMSMSimulator

sim = IPMSMSimulator('FEMM_Lambda_d_matrix.csv', 'FEMM_Lambda_q_matrix.csv', 'FEMM_Lambda_0_matrix.csv', pole_pairs=4, rs=0.0618)
ia = np.sin(0)
ib = np.sin(0 + 2.0 * np.pi / 3.0)
ic = np.sin(0 - 2.0 * np.pi / 3.0)
print("ia, ib, ic:", ia, ib, ic)
idq0 = np.array([0.0, -34, 0.0])
Ldq0 = sim.get_flux(idq0)
print("Ldq0:", Ldq0*1000)

b=30**np.pi/180
park_3x3 = np.array([
            [np.cos(b), np.sin(b), 0], 
            [-np.sin(b), np.cos(b), 0],
            [0, 0, 1]
        ])
print("park_3x3 Transpose:", park_3x3.T)
print("park_3x3 inverse:", np.linalg.inv(park_3x3))

clac = np.array([[1, -1/2, -1/2], 
                         [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                         [1/2, 1/2, 1/2]])
clac2 = 2/3 * np.array([[1, -1/2, -1/2], 
                         [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                         [1/2, 1/2, 1/2]])
print("clac inverse:")
print(np.linalg.inv(clac2)) 
print("clac Transpose:")
print(clac.T)

