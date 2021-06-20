import torch

# Function `_Hermite_spline` and `_Interpolate` are borrowed from @chausies and @Julius's answers 
# to the question "How to do cubic spline interpolation and integration in Pytorch" on stackoverflow.com, 
# with some modifications.     -- Daichao Chen, 2021.3.14 
# (see https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch)

A = torch.tensor([[1, 0, -3, 2],
                  [0, 1, -2, 1],
                  [0, 0, 3, -2],
                  [0, 0, -1, 1]] )

def _Hermite_spline(t):
    '''
        Helper function for cubic Hermite spline interpolation. 
    '''
    global A
    A = torch.as_tensor(A, dtype = t.dtype, device=t.device)
    return A @ (t.view(1, -1) ** torch.arange(4, device=t.device).view(-1, 1) )

def _Interpolate(x, y, xs, zero_grad_pos = None):
    '''
        Cubic Hermite spline interpolation for finding upper and lower envolopes 
        during the sifting process. 
    '''
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat( ( m[0].unsqueeze(0), 
                     (m[1:] + m[:-1]) / 2, 
                     m[-1].unsqueeze(0) )
                    )
    if (zero_grad_pos is not None):
        m[zero_grad_pos] = 0
    
    idxs = torch.searchsorted(x, xs) - 1
    
    dx = x[idxs + 1] - x[idxs]
    
    h = _Hermite_spline((xs - x[idxs]) / dx)
    return    h[0] * y[idxs] \
            + h[1] * m[idxs] * dx  \
            + h[2] * y[idxs + 1] \
            + h[3] * m[idxs + 1] * dx