import torch, math
from .frequency import get_envelope_frequency
from .interpolation1d import _Interpolate

def find_IMF(x, 
             num_sifting : int = 10, 
             thres_num_extrema : int = 2):
    '''
        Extracting an intrinsic mode function using the sifting process.
        
        Parameters:
        -------------
        x : Tensor, of shape (..., # sampling points )
            Signal data. 
        num_sifting : int, optional.
            The number of sifting times. 
            ( Default : 10 )
        thres_num_extrema : int, optional 
            If (#maxima in `x`) or (#minima in `x`) <= `thres_num_extrema`,  `x` will be 
            considered as a signal residual and thus an all-zero function will be the resulting IMF.
            ( Default: 2 )
            
        Returns:
        -------------
        imf : Tensor, of shape (..., # sampling points)
              The extrated intrinsic mode functions for each signal.
              It will be on the same device as `x`.
    '''
    assert num_sifting > 0, "The number of sifting times should be at least one."
    
    x = torch.as_tensor(x).double()
    device = x.device
    N = x.shape[-1]
    batch_dim = x.shape[:-1] # the batch dimensions
    x = x.view(-1, N)
    batch_num = x.shape[0] # the number of batches
    is_residual = torch.zeros(batch_num, dtype = torch.bool, device = device)
    
    evaluate_points = (torch.arange(N, device = device).view(1, -1) + \
                       (2 * N) * torch.arange(batch_num, device = device).view(-1, 1)).view(-1)

    for _ in range(num_sifting):

        # constructing the envelope by interpolation using cubic Hermite spline
        tmp, tmpleft, tmpright = x[..., 1:-1], x[..., :-2], x[..., 2:]
        
        # ---- the upper envelope ----
        maxima_bool = torch.cat( ( (x[..., 0] >= x[..., 1]).view(-1, 1), 
                                    (tmp >= tmpright) & (tmp >= tmpleft), 
                                    (x[..., -1] >= x[..., -2]).view(-1, 1), 
                                    torch.ones((batch_num, 1), dtype = torch.bool, device = device)
                                    ), dim = 1 )
        is_residual.logical_or_( maxima_bool.sum(dim = -1) - 1 <= thres_num_extrema)
        maxima = maxima_bool.nonzero(as_tuple = False).double()
        zero_grad_pos = (maxima[:, 1] < N).logical_not()
        x_maxima = torch.zeros(maxima.shape[0], device = x.device, dtype = x.dtype)
        x_maxima[zero_grad_pos.logical_not()] = x[maxima_bool[:, :N]]
        del maxima_bool
        maxima[zero_grad_pos, 1] = N + (N-1)/2
        maxima = maxima[:, 1] + maxima[:, 0] * 2 * N 
        maxima = torch.cat( (torch.tensor(-(N+1)/2, device = device).view(1),  maxima) )
        x_maxima = torch.cat( (torch.tensor(0, device = device).view(1),  x_maxima) )
        zero_grad_pos = torch.cat( (torch.tensor(0, device = device).view(1),  zero_grad_pos) )
        envelope_up = _Interpolate(maxima, x_maxima, evaluate_points, zero_grad_pos).view(batch_num, -1)
        del maxima, x_maxima, zero_grad_pos
        
        # ---- the lower envelope ----
        minima_bool = torch.cat( ( ( x[..., 0] <= x[..., 1]).view(-1, 1), 
                                     (tmp <= tmpright) & (tmp <= tmpleft), 
                                     (x[..., -1] <= x[..., -2]).view(-1, 1), 
                                     torch.ones((batch_num, 1), dtype = torch.bool, device = device)
                                    ), dim = 1 )
        is_residual.logical_or_( minima_bool.sum(dim = -1) - 1 <= thres_num_extrema)
        del tmp, tmpleft, tmpright
        minima = minima_bool.nonzero(as_tuple = False).double()
        zero_grad_pos = (minima[:, 1] < N).logical_not()
        x_minima = torch.zeros(minima.shape[0], device = x.device, dtype = x.dtype)
        x_minima[zero_grad_pos.logical_not()] = x[minima_bool[:, :N]]
        del minima_bool
        minima[zero_grad_pos, 1] = N + (N-1)/2
        minima = minima[:, 1] + minima[:, 0] * 2 * N 
        minima = torch.cat( (torch.tensor(-(N+1)/2, device = device).view(1),  minima) )
        x_minima = torch.cat( (torch.tensor(0, device = device).view(1),  x_minima) )
        zero_grad_pos = torch.cat( (torch.tensor(0, device = device).view(1),  zero_grad_pos) )
        envelope_down = _Interpolate(minima, x_minima, evaluate_points, zero_grad_pos).view(batch_num, -1)
        del minima, x_minima, zero_grad_pos
            
        # sift and obtain an IMF candidate
        x = x - (envelope_up + envelope_down) / 2
    
    x[is_residual] = 0
    return x.view(batch_dim + torch.Size([N]))

def emd(x, 
        num_imf : int = 10, 
        ret_residual : bool = False, 
        **kwargs):
    '''
        Perform empirical mode decomposition.
        
        Parameters:
        -------------
        x : Tensor, of shape (..., # sampling points)
            Signal data.
        num_imf : int, optional. 
            The number of IMFs to be extracted from `x`.
            ( Default: 10 )
        num_sifting , thres_num_extrema : int, optional.
            See `help(find_IMF)`
        ret_residual : bool, optional. ( Default: False )
            Whether to return the residual signal as well.
        
        Returns:
        -------------
        imfs                 if `ret_residual` is False;
        (imfs, residual)     if `ret_residual` is True.
        
        imfs : Tensor, of shape ( ..., num_imf, # sampling points )
            The extrated IMFs. 
        residual : Tensor, of shape ( ...,  # sampling points )
            The residual term.
    '''
    x = torch.as_tensor(x).double()
    
    imfs = []
    for _ in range(num_imf):
        imf = find_IMF(x, **kwargs)
        imfs.append(imf)
        x = x - imf
    imfs = torch.stack(imfs, dim = -2)
    
    return (imfs, x) if ret_residual else imfs

def hilbert_huang(x, fs, 
                  num_imf : int = 10, 
                  **kwargs):
    '''
        Perform Hilbert-Huang transform on the signal `x`, and return the amplitude and 
        instantaneous frequency function of each intrinsic mode.
        
        Parameters:
        -----------
        x : Tensor, of shape (..., # sampling points)
            Signal data. 
        fs : real. 
            Sampling frequencies in Hz.
        num_imf : int, optional. 
            The number of IMFs to be extracted from `x`.
            ( Default: 10 )
        num_sifting , thres_num_extrema : int, optional.
            See `help(find_IMF)`
            
        Returns:
        -----------
        (imfs, imfs_env, imfs_freq) - 1
        
        imfs : Tensor, of shape (..., num_imf, # sampling points)
            IMFs obtained from `emd`.
        imfs_env : Tensor, of shape (..., num_imf, # sampling points - 1)
            The envelope functions of all IMFs.
        imfs_freq :Tensor, of shape (..., num_imf, # sampling points - 1)
            The instantaneous frequency functions of all IMFs, measured in 'Hz'.
    '''
    imfs = emd(x, num_imf = num_imf, **kwargs)
    imfs_env, imfs_freq = get_envelope_frequency(imfs, fs, **kwargs)
    return imfs, imfs_env, imfs_freq

def hilbert_spectrum(imfs_env, imfs_freq, fs, 
                     freq_lim = None, freq_res = None,
                     time_range = None, time_scale = 1 ):
    '''
        Compute the Hilbert spectrum H(t, f) (which quantify the changes of frequencies of all IMFs over time).
        
        Parameters:
        ------------
        imfs_env : Tensor, of shape (..., # IMFs, # sampling points )
                The envelope functions of all IMFs.
        imfs_freq : Tensor, of shape (..., # IMFs, # sampling points )
                The instantaneous frequency functions of all IMFs. 
        fs : real. 
            Sampling frequencies in Hz.
        freq_max : real, Optional.
            Specifying the maximum instantaneous frequency. If not given, it will be
            automatically selected.
        freq_res : real. Optional.
            Specifying the frequency resolution. 
            If not given, it will be 1 / (total_time_length) = fs / N.
        time_range : (real, real)-tuple. Optional.
            Specifying the range of time domain. If not given, it will be the time span
            of the whole signal, i.e. (0, N*fs).
        time_scale : int. Optional. ( Default : 1 )
            Specifying the scale for the time axis. 
            Thus temporal resolution will be exactly `1/fs * time_scale`.
        
        Returns: 
        ----------
        (spectrum, time_axis, freq_axis)
        
        spectrum : Tensor, of shape ( ..., # time_bins, # freq_bins ). 
            A pytorch tensor, representing the Hilbert spectrum H(t, f).
            The tensor will be on the same device as `imfs_env` and `imfs_freq`.
        time_axis : Tensor, 1D, of shape ( # time_bins )
            The label for the time axis of the spectrum.
        freq_axis : Tensor, 1D, of shape ( # freq_bins )
            The label for the frequency axis (in `freq_unit`) of the spectrum. 
         
    '''
    
    imfs_freq = imfs_freq.double()
    imfs_env = imfs_env.double()
    device = imfs_freq.device
    
    N = imfs_freq.shape[-1]  # total number of sampling points
    T = N / fs               # total time length

    if (freq_lim is None):
        freq_min, freq_max = 0, fs / 2
    else:
        freq_min, freq_max = freq_lim

    if (freq_res is None):
        freq_res = (freq_max - freq_min) / 200       # frequency resolution

    dim_batch = imfs_env.shape[:-2]
    num_imfs = imfs_env.shape[-2]
    imfs_env = imfs_env.view(-1, num_imfs, N)
    imfs_freq = imfs_freq.view(-1, num_imfs, N)
    num_batches = imfs_env.shape[0]
    
    if (time_range):
        L, R = time_range
        L, R = min(int(L * fs), N-1), min(int(R * fs)+1, N)
        imfs_env, imfs_freq = imfs_env[..., L:R], imfs_freq[..., L:R]
        N = R-L
       
    freq_bins = int((freq_max - freq_min) / freq_res) + 1
    time_bins = N // time_scale + 1

    spectrum = torch.zeros( (num_batches, time_bins, freq_bins + 1), device = device )
    
    batch_idx = (torch.arange(num_batches, dtype=torch.long, device=device)).view(-1, 1, 1)
    time_idx = (torch.arange(N, dtype=torch.long, device=device) // time_scale).view(1, 1, -1)
    freq_idx = ((imfs_freq - freq_min) / freq_res).long()

    # out-of-range frequency values will be discarded later
    freq_idx[ freq_idx < 0 ] = freq_bins         
    freq_idx[ freq_idx > freq_bins ] = freq_bins
    
    spectrum[batch_idx, time_idx, freq_idx] += (imfs_env ** 2)
    #spectrum = spectrum / freq_res * fs / time_scale (density spectrum)
    del batch_idx, time_idx, freq_idx
    
    time_axis = torch.arange(N // time_scale + 1, dtype=torch.double) * time_scale / fs \
                + (L / fs if time_range is not None else 0)
    freq_axis = torch.arange(freq_bins, dtype=torch.double) * freq_res + freq_min    

    return ( spectrum[:, :, :freq_bins].view( dim_batch + torch.Size([time_bins, freq_bins]) ), 
             time_axis, 
             freq_axis )

