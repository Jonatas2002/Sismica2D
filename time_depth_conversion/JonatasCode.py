import numpy as np
import matplotlib.pyplot as plt
from seismic import time_for_depth
from seismic import depth_for_time

# Leitura do arquivo
nx = 681
nz = 141
seismic1 = np.reshape(np.fromfile("models/marmousi_141x681_25m.bin", dtype=np.float32), (nx, nz))

# Definindo parâmetros
dh = 25.0
dt = 1e-3    # Espaçamento temporal [s]
time = 4.0   # Tempo total [s]
fmax = 45    # Frequência máxima [Hz] 


xloc = np.linspace(0, nx - 1, 9, dtype=int)
xlab = np.array(xloc * dh, dtype=int)

zloc = np.linspace(0, nz - 1, 9, dtype=int)
zlab = np.array(zloc * dh, dtype=int)

# Tempo
tempos = []
for i in range(nz): 
    tempo = depth_for_time(nz, seismic1[i, :], seismic1[i, :])
    tempos.append(tempo)

Seismic_time = np.array(tempos).T


# Plotando o modelo de velocidade
plt.figure(figsize=(15, 6))
plt.subplot(211)
plt.imshow(seismic1.T, aspect='auto', cmap='jet', extent=[0, nx * dh, nz * dh, 0])
#plt.imshow(seismic1.T, aspect='auto', cmap='jet')

#plt.xticks(xloc * dh, xlab)
#plt.yticks(zloc * dh, zlab)
plt.colorbar(label='Velocity [m/s]')
plt.xlabel('Distance [m]')
plt.ylabel('Depth [m]')
plt.title('Velocity Model')

plt.subplot(212)
plt.imshow(Seismic_time, aspect='auto', cmap='jet', extent=[0, nx * dh, np.max(Seismic_time), 0])
#plt.imshow(Seismic_time, aspect='auto', cmap='jet')

plt.colorbar(label='Velocity [m/s]')
plt.xlabel('Distance [m]')
plt.ylabel('time [s]')
plt.title('Velocity Model')

plt.show()

