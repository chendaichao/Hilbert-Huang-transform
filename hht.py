import math
import numpy as np
import torch 
from scipy import interpolate
from scipy.signal import argrelmin, argrelmax
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_device(torch_device):
    '''
        Set the device for computing HHT.
    '''
    device = torch_device

def _hilbert(x):
    '''
        Perform Hilbert transform along the last axis of x.
        
        Parameters:
        -------------
        x : torch.tensor, of double
            Signal data.
        
        Returns:
        -------------
        analytic : torch.tensor, with size (..., 2)
            The analytic signal. Notice that the last axis of dimension 2 represents 
            the real and image parts of complex numbers.
    '''
    if (type(x) is not torch.tensor):
        x = torch.as_tensor(x)
    x = x.double().to(device)
    N = x.shape[-1]
    Xf = torch.rfft(x, signal_ndim = 1, onesided=False)
    H = torch.zeros(N, dtype = torch.double, device = device)
    if (N % 2 == 0):
        H[0] = H[N//2] = 1
        H[1:N//2] = 2
    else:
        H[0] = 1
        H[1:(N+1)//2] = 2
    H = H.view(torch.Size( [1]*(x.ndim-1) + [N, 1]) ) 
    return torch.ifft(Xf*H, signal_ndim = 1)
    
def get_envelope_frequency(x, delta_t = None, fs = None, ret_analytic = False, freq_unit = 'Hz'):
    '''
        Compute the envelope and instantaneous freqency function of the given signal, using Hilbert transform.
        The transformation is done along the last axis.
        
        Parameters:
        -------------
        x : torch.tensor
            Signal data. must be real.
        delta_t, fs : real. Optional, but exacly one of them must be specified. 
            delta_t = the time interval between two consecutive samples (in seconds). 
            fs = sampling frequencies in Hz = 1. / delta_t
        ret_analytic : bool, optional ( Default: False )
            Whether to return the analytic signal.
        freq_unit : string, 'Hz' or 'rad'. ( Default: 'Hz' )
            Whether to measure the frequency in Hz or radius.
        
        Returns:
        -------------
        (envelope, freq) (or (..., analytic) if ret_analytic is True): tuple
            envelope : torch.tensor
                       The envelope function, with its shape 
                       same as `x` except for the size of last axis being N - 1,
                       where N = `x.shape[-1]`
            freq     : torch.tensor
                       The instantaneous freqency function, with its shape 
                       same as `x` except for the size of last axis being N - 1,
                       where N = `x.shape[-1]`
            analytic : torch.tensor
                       The analytic (complex) signal, with its shape same as `x`.               
    '''
    assert (delta_t is not None) or (fs is not None), "One of `fs` and `delta_t` must be specified."
    if (delta_t is None):
        delta_t = 1. / fs
    
    x = torch.as_tensor(x).to(device)
    
    analytic = _hilbert(x)
    envelope = analytic.norm(dim=-1)[..., :-1]
    sub = analytic[..., 1:, :] - analytic[..., :-1, :]
    add = analytic[..., 1:, :] + analytic[..., :-1, :]
    freq = 2./delta_t * (add[..., 0] * sub[..., 1] - add[..., 1] * sub[..., 0]) / (add**2).sum(dim=-1)
    if (freq_unit == 'Hz'):
        freq /= (2 * math.pi)
    return (envelope, freq) if not ret_analytic else (envelope, freq, analytic)

def find_IMF(x, tolerance = 0.002, num_extrema = 3):
    '''
        Finding an intrinsic mode function using the sifting process.
        
        Parameters:
        -------------
        x : torch.tensor, 1D
            Signal data. must be real.
        tolerance : real, optional. ( Default: 0.002 )
            The threshold used in the stopping criteria for the sifting process.
            When the relative squared mean difference between two consecutive 
            sifting results is smaller than `tolerance`, the sifting process will stop.     
        num_extrema : int, optional ( Default: 2 )
            If (#maxima in `x`) or (#minima in `x`) is <= `num_extrema`,  `x` will be 
            considered as a residual.
        
        Returns:
        -------------
        imf, or None if `x` is considered as a residual signal.
        imf : torch.tensor, 1D
              An intrinsic mode function.
    '''
    x = torch.as_tensor(x).double().to(device)
    N = x.shape[0]
    
    iter = 0
    t = list(range(N))
    while (True):
        
        maxima = argrelmax(x.cpu().numpy(), mode="warp")[0]
        minima = argrelmin(x.cpu().numpy(), mode="warp")[0]
        if (len(maxima) <= num_extrema) or (len(minima) <= num_extrema):
            return None
        
        x_maxima, x_minima = x[maxima], x[minima]
        tck_up, tck_down = (interpolate.splrep(maxima, x_maxima.cpu(), k=3), 
                            interpolate.splrep(minima, x_minima.cpu(), k=3))
        envelope_up, envelope_down = interpolate.splev(t, tck_up, ext=3), interpolate.splev(t, tck_down, ext=3)
        envelope_up, envelope_down = (torch.from_numpy(envelope_up).to(device), 
                                      torch.from_numpy(envelope_down).to(device))
        
        mean = (envelope_up + envelope_down) / 2
        h = x - mean
        if ( (mean**2).sum() / (x**2).sum() < tolerance ):
            break
        x = h 
        
    return h

def emd(x, eps = 0.01, max_num_imf = 20, tolerance = 0.002, num_extrema = 3, ret_residual = False):
    '''
        Perform empirical mode decomposition.
        
        Parameters:
        -------------
        x : torch.tensor, 1D
            Signal data. must be real.
        eps : real, optional. Default: 0.01
            When the maximum amplitude of the residual signal < (eps)*(that of the original signal `x`),
            stop the decomposition.
        max_num_imf : int, optional. Default: 20
            The maximum number of IMFs to be extracted from `x`.
        tolerance, num_extrema : optional.
            See help(find_IMF).
        ret_residual : bool, optional. Default: False
            Whether to return the residual signal as well.
        
        Returns:
        -------------
        imfs, or (imfs, residual) if `ret_residual` is True:
        
        imfs : list of torch.tensor
            A list consisting of the extrated IMFs, each of which has the same shape as `x`. 
        residual : 1-D torch.tensor
            The residual signal, with the same shape as `x`.
    '''
    imfs = []
    x = torch.as_tensor(x).double().to(device)
    scale = torch.abs(x).max()
    for _ in range(max_num_imf):
        if (torch.abs(x).max() < eps * scale):
            break
        imf = find_IMF(x, tolerance, num_extrema)
        if (imf is None):
            break
        imfs.append(imf)
        x = x - imf
    
    return (imfs, x) if ret_residual else imfs

def plot_IMFs(x, imfs, delta_t = None, fs = None, index = None, align_yticks = True, time_range = None, save_fig = None):
    '''
        Visualize the IMFs extracted from the original signal by empirical model decomposition.
        
        Parameters:
        ------------
        x : array_like, 1D
            Signal data. must be real.
        imfs : list of 1-D torch.tensor
            IMFs obtained from `emd`.
        delta_t, fs : real. Optional, but exacly one of them must be specified. 
            delta_t = the time interval between two consecutive samples (in seconds). 
            fs = sampling frequencies in Hz = 1. / delta_t
        index : array of integers. Optional.
            The indices of the IMF component to be shown.
        align_yticks : bool. Optional. Default: True.
            Whether to align the y-axises of all IMFs.
        time_range : (L, R)-tuple. Optional.
            The range of time axis to be shown.
        save_fig : string or None. Optional. Default: None
            Save the image as `save_fig` if it is not None.
        
        Returns: None
    '''
    assert (delta_t is not None) or (fs is not None), "One of `fs` and `delta_t` must be specified."
    if (delta_t is None):
        delta_t = 1. / fs
        
    x = torch.as_tensor(x).double().to(device)
    num_imfs = len(imfs)
    t = torch.arange(x.shape[0], dtype = torch.double) * delta_t
    if (time_range):
        L, R = time_range
        L, R = min(int(L/delta_t), len(t)-1), min(int(R/delta_t)+1, len(t))
        t = t[L:R]
    if (index is None):
        index = list(range(num_imfs))
    else:
        num_imfs = len(index)
    
    plt.figure(figsize=(10, num_imfs+2))
    plt.subplots_adjust(hspace = 0.3, left = 0.3)
    
    # plot signals
    ax = plt.subplot(num_imfs+2, 1, 1)
    scale = max(torch.abs(imfs[:, L:R] if time_range else imfs).max(), (x[L:R] if time_range else x).max())
    scale = scale.cpu()
    ax.plot(t, x[L:R].cpu() if time_range else x.cpu())
    ax.set_ylabel("signal")
    ax.set_ylim(-scale, scale)
    ax.set_xticks([])
    
    # plot IMFs
    for i_ in range(len(index)):
        i = index[i_]
        imf = imfs[i]
        if not align_yticks:
            scale = torch.abs(imf[L:R] if time_range else imf).max().cpu()
        ax = plt.subplot(num_imfs+2, 1, i_+2)
        ax.plot(t, imf[L:R].cpu() if time_range else imf.cpu())
        ax.set_ylim(-scale, scale)
        ax.set_ylabel("IMF %d" % (i))
        ax.set_xticks([])
    
    # plot residual
    x = x - imfs.sum(axis = 0)
    if not align_yticks:
        scale = np.abs(x[L:R] if time_range else x).max().cpu()
    ax = plt.subplot(num_imfs+2, 1, num_imfs+2)
    ax.plot(t, x[L:R].cpu() if time_range else x.cpu())
    ax.set_ylabel("residual")
    ax.set_ylim(-scale, scale)
    ax.set_xlabel("time")
    if (save_fig):
        plt.savefig(save_fig)
    plt.show()
    
def hilbert_huang(x, delta_t = None, fs = None, eps = 0.01, 
                  max_num_imf = 20, tolerance = 0.002, num_extrema = 3, freq_unit = 'Hz'):
    '''
        Perform Hilbert-Huang transform on the signal `x`, and return the amplitude and 
        instantaneous frequency function of each impirical mode.
        
        Parameters:
        -----------
        x : array_like, 1D
            Signal data. must be real.
        delta_t, fs : real. Optional, but exacly one of them must be specified. 
            delta_t = the time interval between two consecutive samples (in seconds). 
            fs = sampling frequencies in Hz = 1. / delta_t
        eps, max_num_imf: optional
            See `help(emd)`.
        tolerance, num_extrema : optional
            See `help(find_IMF)`.
        freq_unit : string, 'Hz' or 'rad'. ( Default: 'Hz' )
            Whether to measure the frequency in Hz or radius.
            
        Returns:
        -----------
        (imfs, imfs_env, imfs_freq) : list of numpy.ndarray
            ( N = len(x), nIMF = #IMFs ) 
            imfs : 2-D torch.tensor of shape (nIMF+1, N)
                IMFs and the residual obtained from `emd`.
            imfs_env : 2-D torxh.tensor of shape (nIMF, N-1)
                The envelope functions of all IMFs.
            imfs_freq : 2-D torxh.tensor of shape (nIMF, N-1)
                The instantaneous frequency functions of all IMFs.
    '''
    assert (delta_t is not None) or (fs is not None), "One of `fs` and `delta_t` must be specified."
    if (delta_t is None):
        delta_t = 1. / fs
    imfs = torch.stack(emd(x, eps, max_num_imf, tolerance, num_extrema))
    imfs_env, imfs_freq = get_envelope_frequency(imfs, delta_t, freq_unit = freq_unit)
    return imfs, imfs_env, imfs_freq

def hilbert_spectrum(imfs_env, imfs_freq, delta_t = None, fs = None, freq_unit = 'Hz', freq_lim = None, 
                     time_range = None, display = True, save_spectrum = None, save_marginal = None):
    '''
        Compute the Hilbert spectrum H(t, f) (which quantify the changes of frequencies of all IMFs over time)
        and the marginal spectrum h(f) = \int_t H(t, f) dt. 
        
        Parameters:
        ------------
        imfs_env : 2-D torch.tensor
                The envelope functions of all IMFs. Each row represents an envelope function.
        imfs_freq : 2-D torch.tensor
                The instantaneous frequency functions of all IMFs. Each row represents an
                instantaneous frequency function.
        delta_t, fs : real. Optional, but exacly one of them must be specified. 
            delta_t = the time interval between two consecutive samples (in seconds). 
            fs = sampling frequencies in Hz = 1. / delta_t
        freq_unit : string, 'Hz' or 'rad'. Optional. ( Default: 'Hz' )
            Whether to measure the frequency in Hz or radius.
        freq_lim : real, Optional.
            Specifying the maximum instantaneous frequency. If not given, it will be
            automatically selected.
        time_range : (L, R)-tuple. Optional.
            Specifying the range of time domain. If not given, it will be the time domain
            of the whole signal.
        display : bool, optional. ( Default : True )
            Whether to illustrate the Hilbert spectrum and the marginal spectrum.
        save_spectrum, save_marginal : string or None. Optional. Default: None
            Save the illustration of the spectrum / marginal-spectrum as 
            `save_spectrum` / `save_marginal` if it is not None.
        
        Notes:
        --------
        If the total time length is T (s), then the frequency resolution will be 1/T (Hz).
        The temporal resolution will be exactly `delta_t`.
        
        Returns: 
        ----------
        (spectrum, time_axis, freq_axis) : 3-tuple
        
        spectrum : torch.sparse.FloatTensor
            A sparse pytorch tensor, representing the Hilbert spectrum H(t, f).
        time_axis : numpy.array, 1D, of size(`spectrum.shape[0]`)
            The label for the time axis of the spectrum.
        freq_axis : numpy.array, 1D, of size(`spectrum.shape[1]`)
            The label for the frequency axis (in `freq_unit`) of the spectrum. 
         
    '''
    assert (delta_t is not None) or (fs is not None), "One of `fs` and `delta_t` must be specified."
    if (delta_t is None):
        delta_t = 1. / fs
        
    imfs_env = imfs_env.double()
    
    N = imfs_freq.shape[1]  # total number of sampling points
    T = N * delta_t         # total time length
    freq_res = 1. / T       # frequency resolution
    if (freq_unit == "rad"):
        freq_res *= 2 * math.pi
    
    if (freq_lim is None):
        # use binary search to find an frequency range containing at least 99.9% of the total energy
        totalE = (imfs_env**2).sum()
        low, high = 0., 1/(2*delta_t)
        if (freq_unit == "rad"):
            high *= 2 * math.pi
        while (low + freq_res < high):
            mid = (low + high)/2
            if (( (imfs_freq<=mid).double() * imfs_env**2).sum() <= 0.999 * totalE):
                low = mid
            else:
                high = mid
        freq_lim = high
    M = int(freq_lim / freq_res + 1)
    
    if (time_range):
        L, R = time_range
        L, R = min(int(L/delta_t), N-1), min(int(R/delta_t)+1, N)
        imfs_env, imfs_freq = imfs_env[:, L:R], imfs_freq[:, L:R]
        N = R-L
    
    spectrum = torch.sparse.DoubleTensor(N, M+1).to(device)
    num_imf = imfs_freq.shape[0]
    time_idx = torch.arange(N, dtype=torch.long).to(device)
    
    for i in range(num_imf):
        
        freq_idx = (imfs_freq[i] / freq_res).long()
        freq_idx[ freq_idx < 0 ] = 0
        freq_idx[ freq_idx > M ] = M
        
        idx = torch.stack([time_idx, freq_idx])
        value = imfs_env[i, :]**2
        
        spectrum.add_(torch.sparse.DoubleTensor(idx, value, torch.Size([N, M+1])).to(device))
    
    spectrum = spectrum.coalesce()
    time_axis = torch.arange(L, R, dtype=torch.double, device=device) * delta_t \
                                if time_range else torch.arange(N, dtype = torch.double, device = device) * delta_t
    freq_axis = torch.arange(M+1, dtype=torch.double, device=device) * freq_res    
    
    if (display):
        
        spectrum_, time_axis_, freq_axis_ = _shrink(spectrum, time_axis, freq_axis)
        spectrum_ = spectrum_.cpu().to_dense().numpy()
        eps = spectrum_.max() * (1e-5)
        coutour = plt.pcolormesh(time_axis_.cpu(), freq_axis_.cpu(), 10*np.log10(spectrum_.T + eps), cmap = plt.cm.YlGnBu_r)
        plt.colorbar(coutour, label = "energy (dB)")
        plt.xlabel("time")
        plt.ylabel("frequency (%s)" % (freq_unit))
        plt.title("Hilbert spectrum")
        if (save_spectrum):
            plt.savefig(save_spectrum)
        plt.show()
    
        marginal = torch.sparse.sum(spectrum, dim=0).cpu().to_dense().view(-1)
        #plt.plot(freq_axis, 10*np.log10(marginal + 1e-10))
        plt.plot(freq_axis.cpu(), marginal)
        plt.xlabel("frequency (%s)" % (freq_unit))
        plt.ylabel("energy")
        plt.title("Marginal spectrum")
        if (save_marginal):
            plt.savefig(save_marginal)
        plt.show()
    
    return spectrum, time_axis, freq_axis

def _shrink(spectrum, time_axis, freq_axis, numt = 1000, numf = 1000):
    
    spectrum = spectrum.coalesce()
    indices = spectrum.indices()
    values = spectrum.values()
    
    if (freq_axis.shape[0] >= 2 * numf):
        stride_f = freq_axis.shape[0] // numf
        freq_axis = freq_axis[::stride_f]
        indices[1, :] = (indices[1, :] // stride_f).long()
        
    if (time_axis.shape[0] >= 2 * numt):
        stride_t = time_axis.shape[0] // numt
        time_axis = time_axis[::stride_t] 
        indices[0, :] = (indices[0, :] // stride_t).long()
    
    spectrum = torch.sparse.DoubleTensor(indices, values, torch.Size([time_axis.shape[0], freq_axis.shape[0]]))
    return spectrum, time_axis, freq_axis