# Importar as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from seismic import wiggle
from seismic import Ricker
from seismic import reflectivity
from seismic import FFT
from seismic import time_for_depth
from seismic import depth_for_time


'''Exemplo de conversão de tempo para profundidade em modelagem 1D'''

# Exemplos de modelagem no tempo

# Parametros geral
Ts = 1   # tempo em segundos
dt = 0.002  # taxa de amostragem
nt = int((Ts/dt) + 1) # numero de amostra
t = np.linspace(0, Ts, nt, endpoint=False)   #base de tempo
tlag= 0.5 # Deslocamento no tempo em segundo

fs = 30  #frequencia do sinal ricker

# Função Wavelet Ricker
R = Ricker(fs, t-tlag)
mascara, freqR, fft_absR = FFT(nt, R, dt)

# PLOT DOS GRAFICOS
plt.figure()
plt.suptitle("visualização gráfica da wavelet ricker ", fontsize=16)

# Plot wavelet ricker
plt.title('Função Wavelet Ricker', fontsize=12)
plt.plot(t, R, 'b',  label="Ricker \nfs = {} Hz".format(fs))
plt.grid()
plt.xlabel('tempo (s)', fontsize=10) # Legenda do eixo x
plt.ylabel('Amplitude', fontsize=10)  # Legenda do eixo y
plt.legend(loc='upper right', fontsize=11)


##############################################################################
plt.tight_layout()
plt.show()

# Perfil Velocidade
velocidade = np.zeros(len(t))
v1 = velocidade[0:int(nt/5)] = 1500   # Velocidade da Agua
v2 = velocidade[int(nt/5):int(2*nt/5)] = 4000  # Velocidade no Arenito
v3 = velocidade[int(2*nt/5):int(3*nt/5)] = 2000 # Velocidade no Argilito
v4 = velocidade[int(3*nt/5):int(4*nt/5)] = 4500  # Velocidade no Sal
v5 = velocidade[int(4*nt/5):int(5*nt/5)] = 5000  # Velocidade no Carbonato

# Perfil Densidade
densidade = np.zeros(nt)
d1 = densidade[0:int(nt/5)] = 1000  # Densidade da Agua
d2 = densidade[int(nt/5):int(2*nt/5)] = 2600  # Densidade do Arenito
d3 = densidade[int(2*nt/5):int(3*nt/5)] = 1800  # Densidade do Argilito
d4 = densidade[int(3*nt/5):int(4*nt/5)] = 2200   # Densidade do Sal
d5 = densidade[int(4*nt/5):int(5*nt/5)] = 2500  # Densidade do Carbonato


# Calculo da Impedância e Refletividade
z = velocidade*densidade  # Calculo da Impedância
z
# Calculo da Refletividade
refletividade =  reflectivity(velocidade,densidade)

# Convolvendo a refletividade com cada uma das wavelets e gerendo traços sismicos sinteticos
trace1D = np.convolve(R, refletividade, mode='same')   # Convolução da Refletividade com a wavelet ricker

plt.figure(figsize=(18, 7))
plt.suptitle("Visualização de modelos de velocidade, densidade, impedância, refletividade das camadas, e traços sísmicos na base temporal", fontsize=12)

# Plot Perfil de Velocidade
plt.subplot(1,5,1)
plt.plot(velocidade,t)
plt.title('Perfil Velocidade de Camadas')
plt.xlabel('Velocidade (m/s)')
plt.ylabel('Tempo (s)')
plt.ylim(max(t), min(t))

# Plot Perfil de Densidade
plt.subplot(1,5,2)
plt.plot(densidade,t)
plt.title('Perfil Densidade de Camadas')
plt.xlabel('Densidade (kg/m³)')
plt.ylabel('Tempo (s)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.ylim(max(t), min(t))

# Plot da impedância
plt.subplot(1,5,3)
plt.plot(z,t)
plt.title('Impedância Acustica')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Impedância Acustica')
#plt.ylabel('Tempo (s)')
plt.ylim(max(t), min(t))

# Plot Refletividade de Camadas
plt.subplot(1,5,4)
plt.plot(refletividade,t)
plt.title('Refletividade de Camadas')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Refletividade (kg/m³)')
#plt.ylabel('Tempo (s)')
plt.ylim(max(t), min(t))

plt.subplot(1,5,5)
plt.plot(trace1D,t,'b', label='Ricker {} Hz'.format(fs))
plt.title('Traço Sismico 1 (Ricker)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Traço Sismico')
#plt.ylabel('Tempo (s)')
plt.legend(loc='upper right', fontsize=11)
plt.ylim(max(t), min(t))

#plt.imshow(np.array([trace1D]*nt).T, aspect='auto',
#           extent=(np.min(trace1D),np.max(trace1D),
#           np.max(t), np.min(t)), cmap='Greys')

plt.tight_layout()
#plt.show()

#####################################################
#####################################################

# Convertendo tempo para profundidade
depth = time_for_depth(nt, t, velocidade)

plt.figure(figsize=(18, 7))
plt.suptitle("Visualização de modelos de velocidade, densidade, impedância, refletividade das camadas, e traços sísmicos na base da profundidade", fontsize=12)

# Plot Perfil de Velocidade
plt.subplot(1,5,1)
plt.plot(velocidade,depth)
plt.title('Perfil Velocidade de Camadas')
plt.xlabel('Velocidade (m/s)')
plt.ylabel('Tempo (s)')
plt.ylim(max(depth), min(depth))

# Plot Perfil de Densidade
plt.subplot(1,5,2)
plt.plot(densidade,depth)
plt.title('Perfil Densidade de Camadas')
plt.xlabel('Densidade (kg/m³)')
plt.ylabel('Tempo (s)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.ylim(max(depth), min(depth))


plt.subplot(1,5,3)
plt.plot(z,depth)
plt.title('Impedância Acustica')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Impedância Acustica')
#plt.ylabel('Tempo (s)')
plt.ylim(max(depth), min(depth))

# Plot Refletividade de Camadas
plt.subplot(1,5,4)
plt.plot(refletividade,depth)
plt.title('Refletividade de Camadas')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Refletividade (kg/m³)')
#plt.ylabel('Tempo (s)')
plt.ylim(max(depth), min(depth))

plt.subplot(1,5,5)
plt.plot(trace1D, depth,'b', label='Ricker {} Hz'.format(fs))
plt.title('Traço Sismico 1 (Ricker)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Traço Sismico')
#plt.ylabel('Tempo (s)')
plt.legend(loc='upper right', fontsize=11)
plt.ylim(max(depth), min(depth))

#plt.imshow(np.array([trace1D]*nt).T, aspect='auto',
#           extent=(np.min(trace1D),np.max(trace1D),
#           np.max(depth), np.min(depth)), cmap='Greys')

plt.tight_layout()
#plt.show()

####################################################
#####################################################

# Convertendo profundidade para tempo
time = depth_for_time(nt, depth, velocidade)

plt.figure(figsize=(18, 7))
plt.suptitle("Visualização de modelos convertidos de volta para a base do tempo", fontsize=12)

# Plot Perfil de Velocidade
plt.subplot(1,5,1)
plt.plot(velocidade,time)
plt.title('Perfil Velocidade de Camadas')
plt.xlabel('Velocidade (m/s)')
plt.ylabel('Tempo (s)')
plt.ylim(max(time), min(time))

# Plot Perfil de Densidade
plt.subplot(1,5,2)
plt.plot(densidade,time)
plt.title('Perfil Densidade de Camadas')
plt.xlabel('Densidade (kg/m³)')
plt.ylabel('Tempo (s)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.ylim(max(time), min(time))


plt.subplot(1,5,3)
plt.plot(z,time)
plt.title('Impedância Acustica')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Impedância Acustica')
#plt.ylabel('Tempo (s)')
plt.ylim(max(time), min(time))

# Plot Refletividade de Camadas
plt.subplot(1,5,4)
plt.plot(refletividade,time)
plt.title('Refletividade de Camadas')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Refletividade (kg/m³)')
#plt.ylabel('Tempo (s)')
plt.ylim(max(time), min(time))

plt.subplot(1,5,5)
plt.plot(trace1D, time,'b', label='Ricker {} Hz'.format(fs))
plt.title('Traço Sismico 1 (Ricker)')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Traço Sismico')
#plt.ylabel('Tempo (s)')
plt.legend(loc='upper right', fontsize=11)
plt.ylim(max(time), min(time))

#plt.imshow(np.array([trace1D]*nt).T, aspect='auto',
#           extent=(np.min(trace1D),np.max(trace1D),
#           np.max(time), np.min(time)), cmap='Greys')

plt.tight_layout()
plt.show()

#####################################################
#####################################################

''' Exemplo de conversão de tempo para profundidade em modelagem 2D '''

"""## Modelo Convolucional 2D """

# Velocidade e Densidade das camadas
vel = np.array([1500, 4000, 2000, 4500, 5000])  # Velocidade
den = np.array([1000, 2600, 1800, 2200, 2500])  # Densidade

# Construção da Matriz de Velocidade e Densidade
nx = 250
velocidade = np.zeros((nt, nx)) + vel[0]
densidade = np.zeros((nt, nx)) + den[0]

horizon1 = np.zeros(nx,dtype='int')
horizon2 = np.zeros(nx,dtype='int')
horizon3 = np.zeros(nx,dtype='int')
horizon31 = np.zeros(nx,dtype='int')  # Falha do Horizonte 3
horizon4 = np.zeros(nx,dtype='int')
horizon41 = np.zeros(nx,dtype='int')  # Falha no Horizonte 4
horizon5 = np.zeros(nx,dtype='int')
horizon51 = np.zeros(nx,dtype='int')


z0, z1, z2, z3, z4, z5, z6, z7 = (nt // 3), (nt // 2.5), (nt // 1.45), (nt // 1), (nt // 1.53), (nt // 1.05), (nt // 1), (nt // 0.92)
height, height1, height2, height3, height4 = 60, 90, 70, 90, 170
L, L1, L2, L3, L4 = (2 * nx), (2 * nx), (1.8 * nx), (1.9 * nx), (8 * nx)

for i in range(nx):
    horizon1[i] = int(z0 + height * np.cos(2*np.pi*(i) / L + np.pi))
    horizon2[i] = int(z1 + height1 * np.cos(2*np.pi*(i) / L1 + np.pi))
    horizon3[i] = int(z2 + height2 * np.sin(2*np.pi*(i) / L2 + np.pi))
    horizon31[i] = int(z4 + height2 * np.sin(2*np.pi*(i) / L2 + np.pi))

    horizon4[i] = int(z3 + height3 * np.sin(2*np.pi*(i) / L3 + np.pi))
    horizon41[i] = int(z5 + height3 * np.sin(2*np.pi*(i) / L3 + np.pi))

    horizon5[i] = int(z6 + height4 * np.sin(2*np.pi*(i) / L4 + np.pi))
    horizon51[i] = int(z7 + height4 * np.sin(2*np.pi*(i) / L4 + np.pi))

for i in range(nx):
    for j in range(nt):
        if (j >= horizon1[i]):
            velocidade[j,i] = vel[1]
            densidade[j,i] = den[1]
        if (j >= horizon2[i]):
            velocidade[j,i] = vel[2]
            densidade[j,i] = den[2]
        if (j >= horizon3[i]):
            velocidade[j,i] = vel[3]
            densidade[j,i] = den[3]
        elif (j >= horizon31[i]):
            velocidade[j,90:120] = vel[3]
            densidade[j,90:120] = den[3]
        if (j >= horizon4[i]):
            velocidade[j,i] = vel[4]
            densidade[j,i] = den[4]
        if (j >= horizon41[i]):
            velocidade[j,90:120] = vel[4]
            densidade[j,90:120] = den[4]
        if (j >= horizon5[i]):
            velocidade[j,i] = 1600
            densidade[j,i] = 1100
        if (j >= horizon51[i]):
            velocidade[j,90:120] = 1600
            densidade[j,90:120] = 1100
            
z = velocidade*densidade  # Calculo da Impedância

# Criando a matriz de refletividade
refletividades = []
for i in range(nx):
    refletividade = reflectivity(velocidade[:, i], densidade[:, i])
    refletividades.append(refletividade)

refletividade = np.array(refletividades).T

# Traço Sismico
traces = []

for i in range(nx):
    trace = np.convolve(R, refletividade[:, i], 'same')
    traces.append(trace)

TRACE = np.array(traces).T

# Profundidade
profundidades = []
for i in range(nx):
    profundidade = time_for_depth(nt, t, velocidade[:, i])
    profundidades.append(profundidade)

prof = np.array(profundidades).T

# Plot dos graficos
plt.figure(figsize=(15,14))
plt.suptitle("visualização gráfica  da refletividade das camadas 2D", fontsize=16)

plt.subplot(211)
plt.imshow(refletividade, aspect='auto',
           extent=(np.min(refletividade),np.max(refletividade),
           np.max(t), np.min(t)), cmap='gray')
plt.title('Refletividade de Camadas')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Refletividade')
plt.ylabel('Tempo (s)')
plt.ylim(max(t), min(t))


plt.subplot(212)
plt.imshow(refletividade, aspect='auto',
           extent=(np.min(refletividade),np.max(refletividade),
           np.max(prof[:, 100]), np.min(prof[:, 100])), cmap='gray')
plt.title('Refletividade de Camadas')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Refletividade')
plt.ylabel('Profundidade (m)')
plt.ylim(np.max(prof[:, 100]), np.min(prof[:, 100]))
plt.tight_layout()
plt.show()


# Plot wiggle na base do tempo 
plt.figure(figsize=(15, 7))
plt.subplot(211)
plt.title("Plot Wiggle")
plt.suptitle("Synthetic Seismic Wiggle", fontsize=16)
wiggle(TRACE, t, xx=None, color='k', sf=0.15, verbose=False)
plt.xlabel('Traço Sismico')
plt.ylabel('Tempo(s)')

# Plot wiggle na base da Profundidade
plt.subplot(212)
plt.title("Plot Wiggle")
wiggle(TRACE, prof[:, 0], xx=None, color='k', sf=0.15, verbose=False)
plt.xlabel('Traço Sismico')
plt.ylabel('Profundidade(m)')

plt.tight_layout()
plt.savefig("SyntheticSeismicWiggle.png")
plt.show()

