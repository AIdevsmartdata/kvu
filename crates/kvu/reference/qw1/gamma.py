"""Methode Gamma de Wolff (hep-lat/0306017) pour une observable primaire."""
import numpy as np


def uwerr(x, S=1.5):
    x = np.asarray(x, float)
    N = x.size
    m = x.mean()
    d = x - m
    # autocorrelation par FFT
    f = np.fft.rfft(d, n=2 * N)
    G = np.fft.irfft(f * np.conj(f))[:N] / np.arange(N, 0, -1)
    G0 = G[0]
    if G0 <= 0:
        return m, 0.0, 0.5, 0
    W = 0
    tau = 0.5
    for W in range(1, N // 2):
        tau = 0.5 + G[1:W + 1].sum() / G0
        tau_c = max(tau, 1e-12)
        tauW = S / np.log((2 * tau_c + 1) / (2 * tau_c - 1)) if tau_c > 0.5 else 1e-12
        g = np.exp(-W / tauW) - tauW / np.sqrt(W * N)
        if g < 0:
            break
    CW = G0 + 2 * G[1:W + 1].sum()
    CW = CW * (1 + (2 * W + 1) / N)  # correction de biais
    tau = CW / (2 * G0)
    tau = tau * (1 + 0)  # pas d'autre correction
    err = np.sqrt(max(CW, 0) / N)
    return m, err, tau, W
