import numpy as np
import matplotlib.pyplot as plt

def insert_zeros(trace, tt=None):
    """Insert zero locations in data trace and tt vector based on linear fit"""

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def wiggle_input_check(data, tt, xx, sf, verbose):
    ''' Helper function for wiggle() and traces() to check input

    '''

    # Input check for verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    # Input check for data
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array")

    # Input check for tt
    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose:
            print("tt is automatically generated.")
            print(tt)
    else:
        if type(tt).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(tt.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")

    # Input check for xx
    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose:
            print("xx is automatically generated.")
            print(xx)
    else:
        if type(xx).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(xx.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")
        if verbose:
            print(xx)

    # Input check for streth factor (sf)
    if not isinstance(sf, (int, float)):
        raise TypeError("Strech factor(sf) must be a number")

    # Compute trace horizontal spacing
    ts = np.min(np.diff(xx))

    # Rescale data by trace_spacing and strech_factor
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std * ts * sf

    return data, tt, xx, ts


def wiggle(data, tt=None, xx=None, color='k', sf=0.15, verbose=False):
    '''Wiggle plot of a sesimic data section

    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)

    Use the column major order for array as in Fortran to optimal performance.

    The following color abbreviations are supported:

    ==========  ========
    character   color
    ==========  ========
    'b'         blue
    'g'         green
    'r'         red
    'c'         cyan
    'm'         magenta
    'y'         yellow
    'k'         black
    'w'         white
    ==========  ========


    '''

    # Input check
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)

    # Plot data using matplotlib.pyplot
    Ntr = data.shape[1]

    ax = plt.gca()
    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx[ntr]

        if verbose:
            print(offset)

        trace_zi, tt_zi = insert_zeros(trace, tt)
        ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                         where=trace_zi >= 0,
                         facecolor=color)
        ax.plot(trace_zi + offset, tt_zi, color)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()
    
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

# Conversão tempo para profundidade
def time_for_depth(nt, time, model):
    depth = np.zeros(nt)
    for i in range(nt):
        if i > 0:
            depth[i] = depth[i-1] + (model[i-1] * (time[i] - time[i-1])) / 2

    return depth

# Conversão profundidade para tempo
def depth_for_time(nt, depth, model):
    time = np.zeros(nt)

    for i in range(nt):
        if i > 0:
            time[i] = np.array(time[i-1] + ( 2 * (depth[i] - depth[i - 1]))/ model[i-1])

    return time