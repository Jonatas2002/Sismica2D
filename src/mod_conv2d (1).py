# Importar as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from plot_wiggle import wiggle

# -----------------------------------------------------------
# -----------------------------------------------------------

# Definindo as funções

# Wavelet Ricker
def Ricker(fs,t):
    R = (1 - 2 * np.pi**2 * fs**2 * t**2 ) * (np.exp(-np.pi**2 * fs**2 * t**2))
    return R

# Transformada de Fourier
def fft_wavelet(n, s, dt):
    freq = np.fft.fftfreq(n, dt)  # Cálculo da frequência
    mascara = freq > 0
    fft_calculo = np.fft.fft(s)  # Cálculo da transformada do sinal final
    fft_abs = 2.0 * np.abs(fft_calculo / n)

    return mascara, freq, fft_abs

# Calculo da Refletividade
def reflectivity(velocidade, densidade):
    z = densidade * velocidade
    refl = np.zeros(len(z))

    for i in range(len(z)-1):
        z2 = z[i+1]
        z1 = z[i]
        refl[i] = (z2 - z1) / (z2 + z1)

    return refl

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

# Parametros geral
T = 1   # tempo em segundos
dt = 0.002  # taxa de amostragem
n = int((T/dt) + 1) # numero de amostra
t = np.linspace(0, T, n, endpoint=False)   #base de tempo
tlag= 0.5 # Deslocamento no tempo em segundo

fs = 30  #frequencia do sinal ricker

# Função Wavelet Ricker
R = Ricker(fs, t-tlag)
mascara, freqR, fft_absR = fft_wavelet(n, R, dt)

# PLOT DOS GRAFICOS
plt.figure(figsize=(12, 3))
plt.suptitle("visualização gráfica da wavelet ricker , acompanhada do espectro de frequência a direita", fontsize=16)

# Plot wavelet ricker
plt.subplot(1,2,1)
plt.title('Função Wavelet Ricker', fontsize=12)
plt.plot(t, R, 'b',  label="Ricker \nfs = {} Hz".format(fs))
plt.grid()
plt.xlabel('tempo (s)', fontsize=10) # Legenda do eixo x
plt.ylabel('Amplitude', fontsize=10)  # Legenda do eixo y
plt.legend(loc='upper right', fontsize=11)

plt.subplot(1,2,2)
plt.title('Espectro de Frequencia -  Wavelet Ricker', fontsize=12)
plt.plot(freqR[mascara], fft_absR[mascara], 'b' , label="Ricker \nfs = {} Hz".format(fs))
plt.grid()
plt.xlabel('Frequência (Hz)', fontsize=10)  # legenda do eixo x
plt.ylabel('|X(f)|', fontsize=10)  # legenda do eixo y
plt.legend(loc='upper right', fontsize=11)

##############################################################################
plt.tight_layout()
#plt.show()

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

"""## Modelo Convolucional 2D """

# velocidades das camadas
vel1 = 1500 # Agua
vel2 = 4000
vel3 = 2000
vel4 = 4500
vel5 = 5000

# Densidade das camadas
den1 = 1000 # Agua
den2 = 2600
den3 = 1800
den4 = 2200
den5 = 2500

# Construção da Matriz de Velocidade e Densidade
nx = 250
velocidade = np.zeros((n, nx)) + vel1
densidade = np.zeros((n, nx)) + den1

horizon1 = np.zeros(nx,dtype='int')
horizon2 = np.zeros(nx,dtype='int')
horizon3 = np.zeros(nx,dtype='int')
horizon31 = np.zeros(nx,dtype='int')  # Falha do Horizonte 3
horizon4 = np.zeros(nx,dtype='int')
horizon41 = np.zeros(nx,dtype='int')  # Falha no Horizonte 4
horizon5 = np.zeros(nx,dtype='int')


z0, z1, z2, z3, z4, z5, z6 = (n // 3), (n // 2.5), (n // 1.45), (n // 1), (n // 1.53), (n // 1.05), (n // 1)
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

for i in range(nx):
    for j in range(n):
        if (j >= horizon1[i]):
            velocidade[j,i] = vel2
            densidade[j,i] = den2
        if (j >= horizon2[i]):
            velocidade[j,i] = vel3
            densidade[j,i] = den3
        if (j >= horizon3[i]):
            velocidade[j,i] = vel4
            densidade[j,i] = den4
        elif (j >= horizon31[i]):
            velocidade[j,90:120] = vel4
            densidade[j,90:120] = den4
        if (j >= horizon4[i]):
            velocidade[j,i] = vel5
            densidade[j,i] = den5
        if (j >= horizon41[i]):
            velocidade[j,90:120] = vel5
            densidade[j,90:120] = den5
        if (j >= horizon5[i]):
            velocidade[j,i] = 100
            densidade[j,i] = 100            
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

# Simular uma falha
# definir a posição onde irá ocorrer as falhas
# deslocar as propriedades verticalmente na posição definida                        

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
            
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

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

# Plot dos graficos
plt.figure(figsize=(15,15))
plt.suptitle("visualização gráfica  da refletividade das camadas 2D", fontsize=16)
plt.imshow(refletividade, aspect='auto',
           extent=(np.min(refletividade),np.max(refletividade),
           np.max(t), np.min(t)), cmap='gray')
plt.title('Refletividade de Camadas')
#plt.yticks([])  # Remova as marcações do eixo y
plt.xlabel('Refletividade')
plt.ylabel('Tempo (s)')
plt.ylim(max(t), min(t))
#plt.show()

# Inserir o dado do traço sismico e a base de tempo ou profundidade
plt.figure(figsize=(15, 15))
plt.title("Plot Wiggle")
plt.suptitle("visualização gráfica dos traços sismicos lado a lado  utilizando o wiggle", fontsize=16)
wiggle(TRACE, t, xx=None, color='k', sf=0.15, verbose=False)
plt.xlabel('Traço Sismico')
plt.ylabel('Tempo(s)')

plt.savefig("SyntheticSeismicWiggle2.png")
plt.show()