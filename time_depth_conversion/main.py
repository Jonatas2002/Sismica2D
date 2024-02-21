from model2section import Model2section
from model2section import read_binary_matrix

import numpy as np

dh = 25.0

nx = [681, 601, 801]
nz = [141, 161, 181]

marmousi = read_binary_matrix(nz[0],nx[0],"models/marmousi_141x681_25m.bin")
eageSalt = read_binary_matrix(nz[1],nx[1],"models/eageSalt_161x601_25m.bin")
overthurst = read_binary_matrix(nz[2],nx[2],"models/overthrust_181x801_25m.bin")

dt = 1e-3    # time spacing [s]
time = 4.0   # total time [s]
fmax = 45    # Max frequency [Hz] 

Model2section(marmousi, dh, time, dt, fmax)
Model2section(eageSalt, dh, time, dt, fmax)
Model2section(overthurst, dh, time, dt, fmax)

