# Copyrights 2013 Pierre Chanial


import multiprocessing

import numpy as np

from pyoperators.memory import empty

try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    pass


def _fold_psd(p):
    """Convert even two-sided PSD into one-sided PSD."""
    p = np.asarray(p)
    n = p.shape[-1] // 2 + 1
    out = empty(p.shape[:-1] + (n,))
    out[..., 0] = p[..., 0]
    np.multiply(p[..., 1 : n - 1], 2, out[..., 1:-1])
    out[..., -1] = p[..., n]
    return out


def _unfold_psd(p):
    """Convert one-sided PSD into even two-sided PSD."""
    p = np.asarray(p)
    n = p.shape[-1]
    out = empty(p.shape[:-1] + (2 * (n - 1),))
    out[..., 0] = p[..., 0]
    np.multiply(p[..., 1 : n - 1], 0.5, out[..., 1 : n - 1])
    out[..., n - 1] = p[..., n - 1]
    out[..., n:] = out[..., n - 2 : 0 : -1]
    return out


def _gaussian_psd_1f(
    nsamples, sampling_frequency, sigma, fknee, fslope, twosided=False
):
    """
    Generate a 0-mean Power Spectrum Density with the distribution:
        psd = sigma**2 * (1 + (fknee/f)**fslope) / B
    where B is equal to sampling_frequency / nsamples.
    The PSD frequency increment is B.

    Parameter
    ---------
    nsamples : int
        Number of time samples.
    sampling_frequency : float
        Sampling frequency [Hz].
    sigma : float or array-like
        The standard deviation of the white noise component.
    fknee : float or array-like
        The 1/f knee [Hz].
    fslope : float or array-like
        The 1/f^fslope slope index.
    twosided : boolean, optional
        If true, a two-sided PSD (with positive and negative frequencies)
        is returned and a one-sided PSD otherwise (only positive frequencies).

    Returns
    -------
    psd : np.ndarray
        The Power Spectrum Density [signal unit**2/Hz]. The frequency
        increment is sampling_frequency / nsamples.

    """
    sigma = np.asarray(sigma)[..., None]
    fknee = np.asarray(fknee)[..., None]
    fslope = np.asarray(fslope)[..., None]

    frq = np.fft.fftfreq(nsamples, d=1 / sampling_frequency)
    frq[0] = np.inf
    psd = sigma**2 * (1 + np.abs(fknee / frq) ** fslope) / sampling_frequency
    if not twosided:
        psd = _fold_psd(psd)
    return psd


def _gaussian_sample(
    nsamples,
    sampling_frequency,
    psd,
    twosided=False,
    out=None,
    fftw_flag='FFTW_MEASURE',
):
    """
    Generate a gaussian N-sample sampled at fs from a one- or two-sided
    Power Spectrum Density sampled at fs/N.

    Parameter
    ---------
    nsamples : int
        Number of time samples.
    sampling_frequency : float
        Sampling frequency [Hz].
    psd : array-like
        One- or two-sided Power Spectrum Density [signal unit**2/Hz].
    twosided : boolean, optional
        Whether or not the input psd is one-sided (only positive frequencies)
        or two-sided (positive and negative frequencies).
    out : ndarray
        Placeholder for the output buffer.

    """
    psd = np.asarray(psd)
    if out is None:
        out = empty(psd.shape[:-1] + (nsamples,))

    if not twosided:
        psd = _unfold_psd(psd)

    shape = psd.shape[:-1] + (nsamples,)
    gauss = np.random.randn(*shape)
    nthreads = multiprocessing.cpu_count()
    ftgauss = fft.fft(gauss, planner_effort=fftw_flag, threads=nthreads)
    ftgauss[..., 0] = 0
    spec = ftgauss * np.sqrt(psd)
    out[...] = fft.ifft(
        spec, planner_effort=fftw_flag, threads=nthreads
    ).real * np.sqrt(sampling_frequency)
    return out


def _sampling2psd(sampling, sampling_frequency, fftw_flag='FFTW_MEASURE'):
    """
    Return folded PSD without binning.

    """
    sampling = np.asarray(sampling)
    n = sampling.size
    nthreads = multiprocessing.cpu_count()
    psd = (
        np.abs(fft.fft(sampling, axis=-1, planner_effort=fftw_flag, threads=nthreads))
        ** 2
    )
    freq = np.fft.fftfreq(n, d=1 / sampling_frequency)[: n // 2 + 1]
    freq[-1] += sampling_frequency
    psd = np.concatenate([[0.0], 2 * psd[1 : n // 2], [psd[n // 2]]]) / (
        n * sampling_frequency
    )
    return freq, psd


def _psd2invntt(psd, bandwidth, ncorr, fftw_flag='FFTW_MEASURE'):
    """
    Compute the first row of the inverse of the noise time-time correlation
    matrix from two-sided PSD with frequency increment fsamp/psd.size.

    """
    fftsize = psd.shape[-1]
    fsamp = bandwidth * fftsize
    psd = (1 / fsamp) / psd
    nthreads = multiprocessing.cpu_count()
    out = fft.fft(psd, axis=-1, planner_effort=fftw_flag, threads=nthreads)
    invntt = out[..., : ncorr + 1].real / fftsize
    return invntt


def _logloginterp_psd(f, bandwidth, psd, out=None):
    """Loglog-interpolation of one-sided PSD."""
    f = np.asarray(f)
    psd = np.asarray(psd)
    frequency = np.arange(psd.shape[-1], dtype=float) * bandwidth
    if out is None:
        out = empty(psd.shape[:-1] + (f.size,))
    out[..., 0] = psd[..., 0]
    _interp(
        np.log(f[1:]), np.log(frequency[1:-1]), np.log(psd[..., 1:-1]), out=out[..., 1:]
    )
    np.exp(out[..., 1:], out[..., 1:])
    out[..., -1] /= 2
    return out


def _interp(z, x, y, out=None):
    """Interpolate / extrapolate y(x) in z. x and z increasing."""
    z = np.asarray(z)
    x = np.asarray(x)
    y = np.asarray(y)
    if out is None:
        out = empty(y.shape[:-1] + (z.size,))
    z = np.array(z, ndmin=1, copy=False)
    ix = 1
    x1 = x[0]
    x2 = x[1]
    for iz, z_ in enumerate(z):
        while z_ > x2 and ix < x.size - 1:
            ix += 1
            x1 = x2
            x2 = x[ix]
        out[..., iz] = ((z_ - x1) * y[..., ix] + (x2 - z_) * y[..., ix - 1]) / (x2 - x1)
    return out
