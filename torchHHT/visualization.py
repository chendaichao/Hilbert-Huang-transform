import torch
import matplotlib.pyplot as plt

def plot_IMFs(x, imfs, fs, index = None, align_yticks = True, time_scale = 1, time_range = None, save_fig = None, title = None):
    '''
        Visualize the IMFs extracted from the original signal by empirical model decomposition.
        
        Parameters:
        ------------
        x (Tensor) : 
            Signal data. The last dimension of `x` is considered as time. 
        imfs (Tensor) : 
            IMFs obtained from `emd`.
        fs (real) : 
            Sampling frequencies in Hz.
        index (list of integers, Optional) : 
            The indices of the IMF component to be shown.
        align_yticks (bool, optional) : 
            Whether to align the y-axises of all IMFs.
             Default: True.
        time_scale (int, optional) : 
            The scale for the time axis for ploting. 
        time_range (an (L, R)-tuple, Optional) : 
            The range of time axis to be shown.
        save_fig (list of string or string, Optional) : 
            Save the image as `save_fig` if specified.
        title (str, optional) :
            Specifying the title of the figure. 
        
        Returns: None
    '''
    
    assert(type(time_scale) is int)
    assert( (save_fig is None) or (type(save_fig) in [list, str]) )
    
    all_x = torch.as_tensor(x).view(-1, x.shape[-1])
    all_imfs = torch.as_tensor(imfs).view(-1, imfs.shape[-2], imfs.shape[-1])
    
    for batch_idx in range(all_x.shape[0]):
        
        imfs, x = all_imfs[batch_idx], all_x[batch_idx]
    
        # scaling the time axis
        if (time_scale > 1):
            N = x.shape[0]
            fs /= time_scale
            M = N // time_scale
            x = x[:M*time_scale].view(M, time_scale).mean(dim = 1)
            imfs = imfs[:, :M*time_scale].view(imfs.shape[0], M, time_scale).mean(dim = 2)

        # the IMF indexes to be shown
        num_imfs = len(imfs)
        if (index is None):
            index = list(range(num_imfs))
        else:
            num_imfs = len(index)

        # the time range of interest
        t = torch.arange(x.shape[0], dtype = torch.double) / fs
        if (time_range):
            L, R = time_range
            L, R = min(int(L * fs), len(t)-1), min(int(R * fs)+1, len(t))
            t = t[L:R]

        # initialize pyplot
        plt.figure(figsize=(10, num_imfs+2))
        plt.subplots_adjust(hspace = 0.3, left = 0.3)

        # plot signals
        ax = plt.subplot(num_imfs+2, 1, 1)
        scale = max(torch.abs(imfs[:, L:R] if time_range else imfs).max(), (x[L:R] if time_range else x).max())
        scale = scale.cpu()
        ax.plot(t, x[L:R].cpu() if time_range else x.cpu(), linewidth = 1)
        ax.set_title("Empirical Mode Decomposition" if title is None else title)
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
            ax.plot(t, imf[L:R].cpu() if time_range else imf.cpu(), linewidth = 1)
            ax.set_ylim(-scale, scale)
            ax.set_ylabel("IMF %d" % (i))
            ax.set_xticks([])

        # plot the residual
        x = x - imfs.sum(axis = 0)
        if not align_yticks:
            scale = torch.abs(x[L:R] if time_range else x).max().cpu()
        ax = plt.subplot(num_imfs+2, 1, num_imfs+2)
        ax.plot(t, x[L:R].cpu() if time_range else x.cpu(), linewidth = 1)
        ax.set_ylabel("residual")
        ax.set_ylim(-scale, scale)
        ax.set_xlabel("time")

        if (save_fig is not None):
            plt.savefig(save_fig if type(save_fig) is str else save_fig[batch_idx])
        plt.show()
        
def plot_HilbertSpectrum(spectrum, time_axis, freq_axis, energy_scale = "log", save_spectrum = None, 
                         save_marginal = None, title = None):
    '''
        Visualize the Hilbert spectrum, by plotting all the IMFs on a time-frequency plane.
        
        Parameters:
        ------------
        spectrum : Tensor, of shape ( ..., # time_bins, # freq_bins ). 
            A pytorch tensor, representing the Hilbert spectrum H(t, f).
            The tensor will be on the same device as `imfs_env` and `imfs_freq`.
        time_axis : Tensor, 1D, of shape ( # time_bins )
            The label for the time axis of the spectrum.
        freq_axis : Tensor, 1D, of shape ( # freq_bins )
            The label for the frequency axis (in `freq_unit`) of the spectrum.
        energy_scale : string ('linear' or 'log')
            Specifying whether to visualize the energy in linear/log scale.
        save_spectrum : string or list of string, Optional.
            If specified, the Hilbert spectrum will be saved as a file named `save_spectrum`.
        save_maeginal : string or list of string, Optional.
            If specified, the Hilbert marginal spectrum will be saved as a file named `save_marginal`.     
        title : string, optional. 
            Specifying the title of the figure. 
    '''    
    spectrum = spectrum.view(-1, spectrum.shape[-2], spectrum.shape[-1])
    time_axis = time_axis.cpu()
    freq_axis = freq_axis.cpu()
    
    for batch_idx in range(spectrum.shape[0]):

        if (energy_scale == "log"):
            eps = spectrum[batch_idx, :, :].max() * (1e-5)
            coutour = plt.pcolormesh(time_axis.cpu(), freq_axis.cpu(), 
                                    10*torch.log10(spectrum[batch_idx, :, :].T + eps).cpu(), 
                                    shading='auto',
                                    cmap = plt.cm.jet)
            plt.colorbar(coutour, label = "energy (dB)")
        else:
            coutour = plt.pcolormesh(time_axis.cpu(), freq_axis.cpu(), 
                                    spectrum[batch_idx, :, :].T.cpu(), 
                                    shading='auto',
                                    cmap = plt.cm.jet)
            plt.colorbar(coutour, label = "energy")

        plt.xlabel("time")
        plt.ylabel("frequency (Hz)")
        plt.title("Hilbert spectrum" if title is None else title)
        if (save_spectrum is not None):
            plt.savefig(save_spectrum if type(save_spectrum) is str else save_spectrum[batch_idx], dpi = 600)
        plt.show()

    if (save_marginal is None) :
        return 
    
    marginal = spectrum.sum(dim = -2)
    for batch_idx in range(spectrum.shape[0]):
        if (energy_scale == "log"):
            eps = marginal[batch_idx, :].max() * (1e-5)
            plt.plot(freq_axis.cpu(), 10*torch.log10(marginal[batch_idx, :].cpu() + eps) )
            plt.ylabel("energy (dB)")
        else:
            plt.plot(freq_axis.cpu(), marginal[batch_idx, :].cpu())
            plt.ylabel("energy")
        plt.xlabel("frequency (Hz)")
        plt.title("Marginal spectrum" if title is None else title + " (marginal)")
        plt.savefig(save_marginal if type(save_marginal) is str else save_marginal[batch_idx], dpi = 600)
        plt.show()