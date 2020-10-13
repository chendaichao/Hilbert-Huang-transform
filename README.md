# Hilbert-Huang Transform

A demo of using Hilbert-Huang Transform (HHT) for non-stationary and non-linear signal analysis.



## Introduction

Time-frequency analysis is a fundamental topic in non-stationary signal processing.  Window-based methods (including short-time Fourier transform and wavelet transform)  assume that the system is linear, and thus compute the integration of the signal multiplied by a family of predefined base functions (indexed by time and frequency) to quantify the time-frequency distribution. Defining the time-frequency spectrum by integration, however, unavoidably results in the uncertainty rule: the frequency and temporal resolution cannot be too fine at the same time. In addition, linear assumption is not always justified, especially when modulation exists. Although one might argue that for modulated signal there still exists a representation with the base functions (say, cos A cos B = 1/2 cos(A+B) + 1/2 cos(A-B), just a frequency shift), this would unnecessarily induce many annoying harmonics, which are indeed mathematical artifacts.

> Here for illustration we generate a summation of two Gaussian-modulated chirp signal, with gradually increasing frequencies starting from 5Hz and 40Hz, respectively. The resulting signal is shown below.
>
> ![signal](img/signal.png)
>
> The STFT results is also illustrated below. From the STFT spectrum, the changes of the frequencies over time can be roughly seen; however, due to the uncertainty principle, the frequency and temporal resolutions can not be fine enough at the same time. The Fourier analysis treats the non-linear modulation linearly, resulting in a blurry spectrum (consisting of a lot of energy leakage and unnecessary harmonics).
>
> ![STFT_spectrum](img/STFT_spectrum.png)

![readme_formula1](img/readme_formula1.png)

However, the results produced by Hilbert transform are not always physically meaningful. In Huang et al. [1], they suggest that the signal should be decomposed into several components, such that each component (designated as intrinsic mode function (IMF) ) satisfies:

1. the upper envelope (defined by its maxima) and the lower envelope (defined by its minima) should be symmetric, and
2. there is exactly one zero-crossing between every two neighboring extrema.

Therefore, meaningful instantaneous frequency can be obtained from the Hilbert transform on each IMF component. To decompose the signal into several IMFs, we first extract an IMF function c<sub>1</sub>(t) from the data x(t), and then extract another IMF c<sub>2</sub>(t)  from x(t) - c<sub>1</sub>(t) , ..., until the residual r(t) = x(t) - c<sub>1</sub>(t)  - c<sub>2</sub>(t) - ... - c<sub>n</sub>(t) becomes ignorable or monotonic. In terms of extracting each IMF component, we also iteratively find the upper and lower envelope function (by using a cubic spline function crossing all maxima (for the upper envelope) or all minima (for the lower one) ), and sift the data by subtracting the envelopes' mean; this process also repeats several times until the envelopes' mean becomes close enough to zero. 

The above algorithm is call empirical mode decomposition (EMD) by [1]. After performing the Hilbert transform, we indeed have a decomposition of the form

![readme_formula2](img/readme_formula2.png)

> Here is an illustration for the EMD on the mixing chirps. We can see than the two modulated chirps are successfully separated and represented by `IMF 0` and `IMF 1`. Other IMFs and the residual is indeed ignorable since their scales are small relative to the original data.
>
> ![EMD](img/EMD.png)
>
> Now let's further visualize the time-frequency spectrum. We discard the spectrum in the first an d last 0.25s since the endpoint effect severely corrupt the low-frequency components in the spectrum.
>
> ![Hilbert_spectrum](img/Hilbert_spectrum.png)![Hilbert_marginal](img/Hilbert_marginal.png)
>
> In the illustration, the variation of frequencies over time can be clearly seen - one increases linearly from 40Hz, reaching 50Hz at 1.2s, and the other increases quadratically from 5Hz, reaching 10Hz at 0.8s. From the color map one can observe that both of their amplitudes are modulated. All of these match our parameters for the chirps. The marginal spectrum also show two peaks of the frequency distribution. 



## Implementation

We implement the Hilbert-Huang transform in *python*. The main algorithm is implement in [hht.py](hht.py). The example of the mixing chirps shown above is given in the *Jupyter notebook* [demo.ipynb](demo.ipynb). 



## Acknowledgement

Special thanks to professor Norden E. Huang for his substantial help. I have learned a lot from his remarkable insights into signal analysis and HHT.

 

## References

[1] Huang, Norden E., et al. "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis." *Proceedings of the Royal Society of London. Series A: mathematical, physical and engineering sciences* 454.1971 (1998): 903-995.

