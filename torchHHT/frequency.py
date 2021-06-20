import torch, math
from .interpolation1d import _Interpolate
from matplotlib import pyplot as plt

# -------- Hilbert-transform for demodulation ---------------

def hilbert(x):
    '''
        Perform Hilbert transform along the last axis of x.
        
        Parameters:
        -------------
        x (Tensor) : The signal data. 
                     The Hilbert transform is performed along last dimension of `x`.
        
        Returns:
        -------------
        analytic (Tensor): A complex tensor with the same shape of `x`,
                           representing its analytic signal. 
            
    '''
    x = torch.as_tensor(x).double()
    
    N = x.shape[-1]
    Xf = torch.fft.fft(x)
    if (N % 2 == 0):
        Xf[..., 1 : N//2] *= 2
        Xf[..., N//2+1 : ] = 0
    else:
        Xf[..., 1 : (N+1)//2] *= 2
        Xf[..., (N+1)//2 : ] = 0
    return torch.fft.ifft(Xf)

def get_envelope_frequency(x, fs, ret_analytic = False, **kwargs):
    '''
        Compute the envelope and instantaneous freqency function of the given signal, using Hilbert transform.
        The transformation is done along the last axis.
        
        Parameters:
        -------------
        x (Tensor) : 
            Signal data. The last dimension of `x` is considered as the temporal dimension.
        fs (real) : 
            Sampling frequencies in Hz.
        ret_analytic (bool, optional) :
            Whether to return the analytic signal.
            ( Default: False )
        
        Returns:
        -------------
        (envelope, freq)             when `ret_analytic` is False
        (envelope, freq, analytic)   when `ret_analytic` is True
        
            envelope (Tensor) : 
                       The envelope function, with its shape same as `x`.
            freq     (Tensor) : 
                       The instantaneous freqency function measured in Hz, with its shape 
                       same as `x`.
            analytic (Tensor) : 
                       The analytic (complex) signal, with its shape same as `x`.
    '''
    x = torch.as_tensor(x).double()
    
    analytic = hilbert(x)
    envelope = analytic.abs()
    sub = torch.cat( (analytic[..., 1:] - analytic[..., :-1], 
                        (analytic[..., -1]-analytic[..., -2]).unsqueeze(-1)
                        ) , axis = -1 )
    add = torch.cat( (analytic[..., 1:] + analytic[..., :-1], 
                        2 * analytic[..., -1].unsqueeze(-1)
                        ) , axis = -1 )
    freq = 2 * fs * ((sub / add).imag)
    freq[freq.isinf()] = 0
    del sub, add
    freq /= (2 * math.pi)
    return (envelope, freq) if not ret_analytic else (envelope, freq, analytic)
