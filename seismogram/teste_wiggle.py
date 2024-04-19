import numpy as np
import matplotlib.pyplot as plt
from seismic import wiggle

# Leitura do arquivo
nx = 282
nz = 1501
dt = 2e-3
dx = 25
t = np.arange(nz)*dt

cmp_gather = np.reshape(np.fromfile("seismogram/arquivo binario/open_data_seg_poland_vibroseis_2ms_1501x282_shot_1.bin", dtype=np.float32), (nx, nz))

# -----------------------------------------------------------------------------
trace1 = cmp_gather[0,:]
trace80 = cmp_gather[79,:]
trace100 = cmp_gather[99,:]
trace161 = cmp_gather[160,:]
# -----------------------------------------------------------------------------

perc = np.percentile(cmp_gather, 90)
# Vizualizando o CMP Gather

fig, ax = plt.subplots(figsize = (5,7))
ax.set_title('CMP Gather',  fontsize = 15)
ax.imshow(cmp_gather.T, aspect='auto', cmap='Grays', extent=[-(nx * dx)/2, (nx * dx)/2, nz * dt, 0], vmin=-perc, vmax=perc)
ax.plot(trace100 - 1025, t, 'red', '---')

ax.set_xlabel('x = offset[m]',  fontsize = 15)
ax.set_ylabel('t = TWT [s]',  fontsize = 15)
#ax.set_xticks(np.arange(-3525,3526,1175))

fig.tight_layout()
plt.show()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# fig, ax = plt.subplots(ncols = 4, nrows = 1, num = "TRAÇO 1", figsize = (12, 10))
# ax[0].set_title('TRAÇO 1')
# ax[0].plot(trace1, t)
# ax[0].set_ylabel("Time [s]")
# ax[0].set_xlabel(r"$x(t)$")

# ax[1].set_title('TRAÇO 80')
# ax[1].plot(trace80, t)
# ax[1].set_ylabel("Time [s]")
# ax[1].set_xlabel(r"$x(t)$")

# ax[2].set_title('Time Domain')
# ax[2].plot(trace80, t)
# ax[2].set_ylabel("Time [s]")
# ax[2].set_xlabel(r"$x(t)$")

# ax[3].set_title('Time Domain')
# ax[3].plot(trace161, t)
# ax[3].set_ylabel("Time [s]")
# ax[3].set_xlabel(r"$x(t)$")

# fig.tight_layout()
# plt.show()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
plt.title("Plot Wiggle")
plt.suptitle("Synthetic Seismic Wiggle", fontsize=16)
tw = wiggle(cmp_gather.T, t, xx=None, color='k', sf=1.5, verbose=False)

plt.xlabel('Traço Sismico')
plt.ylabel('Tempo(s)')
plt.savefig("SyntheticSeismicWiggle.png")
plt.show()

plt.figure(figsize=(15, 10))
plt.title("Plot Wiggle")
plt.suptitle("Synthetic Seismic Wiggle", fontsize=16)
tw1 = wiggle(cmp_gather[79:81,:].T, t, xx=None, color='k', sf=0.1, verbose=False)
tw2 = wiggle(cmp_gather[100:102,:].T, t, xx=None, color='r', sf=0.1, verbose=False)

plt.xlabel('Traço Sismico')
plt.ylabel('Tempo(s)')
plt.savefig("SyntheticSeismicWiggle.png")
plt.show()


