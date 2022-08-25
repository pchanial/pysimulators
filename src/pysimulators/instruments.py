import copy

import numpy as np

from pyoperators import MPI, DiagonalOperator, SymmetricBandToeplitzOperator, asoperator
from pyoperators.memory import empty
from pyoperators.utils import operation_assignment, split

from .noises import (
    _fold_psd,
    _gaussian_psd_1f,
    _gaussian_sample,
    _logloginterp_psd,
    _psd2invntt,
    _unfold_psd,
)
from .packedtables import Layout

__all__ = ['Instrument', 'Imager']


class Instrument:
    """
    Class storing information about the instrument.

    Attributes
    ----------
    layout : Layout
        The detector layout.
    name : str, optional
        The instrument configuration name.

    """

    def __init__(self, layout, name=None):
        # allow obsolete calling sequence Instrument(name, layout)
        if isinstance(layout, str):
            layout, name = name, layout
        if name is not None:
            self.name = name
        self.detector = layout

    def __getitem__(self, selection):
        """
        Shallow copy of the Instrument for the selected deep-copied
        non-removed detectors.

        """
        out = copy.copy(self)
        out.detector = self.detector[selection]
        return out

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.detector)

    def pack(self, x):
        return self.detector.pack(x)

    pack.__doc__ = Layout.pack.__doc__

    def unpack(self, x):
        return self.detector.unpack(x)

    unpack.__doc__ = Layout.unpack.__doc__

    def scatter(self, comm=None):
        """
        MPI-scatter of the instrument.

        Parameter
        ---------
        comm : MPI.Comm
            The MPI communicator of the group of processes in which
            the instrument will be scattered.

        """
        if self.detector.comm.size > 1:
            raise ValueError('The instrument is already distributed.')
        if comm is None:
            comm = MPI.COMM_WORLD
        out = copy.copy(self)
        out.detector = out.detector.scatter(comm)
        return out

    def split(self, n):
        """
        Split the instrument in partitioning groups.

        Example
        -------
        >>> instr = Instrument('instr', Layout((4, 4)))
        >>> [len(_) for _ in instr.split(2)]
        [8, 8]

        """
        return tuple(self[_] for _ in split(len(self), n))

    def plot(self, **keywords):
        """
        Plot the instrument detector footprint.

        Parameters
        ----------
        autoscale : boolean, optional
            If true, the axes of the plot will be updated to match the
            boundaries of the detectors.
        transform : callable, optional
            Operator to be used to transform the detector coordinates into
            the data coordinate system.

        Example
        -------
        # overlay the detector grid on the observation pointings
        acq = MyAcquisition(...)
        acq.sampling.plot()
        transform = convert_coords_instrument2xy(...)
        acq.instrument.plot(autoscale=False, transform=transform)

        """
        self.detector.plot(**keywords)

    def get_operator(self, sampling, scene):
        """
        Return the acquisition model for the specified sampling and scene
        as an operator.

        """
        raise NotImplementedError()

    def get_invntt_operator(
        self,
        sampling,
        psd=None,
        bandwidth=None,
        twosided=False,
        sigma=None,
        nep=None,
        fknee=0,
        fslope=1,
        ncorr=None,
        fftw_flag='FFTW_MEASURE',
        nthreads=None,
    ):
        """
        Return the inverse time-time noise correlation matrix as an Operator.

        The input Power Spectrum Density can either be fully specified by using
        the 'bandwidth' and 'psd' keywords, or by providing the parameters of
        the gaussian distribution:
            psd = sigma**2 * (1 + (fknee/f)**fslope) / B
        where B is the sampling bandwidth equal to sampling_frequency / N.

        Parameters
        ----------
        sampling : Sampling
            The temporal sampling.
        psd : array-like, optional
            The one-sided or two-sided power spectrum density
            [signal unit/sqrt Hz].
        bandwidth : float, optional
            The PSD frequency increment [Hz].
        twosided : boolean, optional
            Whether or not the input psd is one-sided (only positive
            frequencies) or two-sided (positive and negative frequencies).
        sigma : float
            Standard deviation of the white noise component.
        fknee : float
            The 1/f noise knee frequency [Hz].
        fslope : float
            The 1/f noise slope.
        sampling_frequency : float
            The sampling frequency [Hz].
        ncorr : int
            The correlation length of the time-time noise correlation matrix.
        fftw_flag : string, optional
            The flags FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT and
            FFTW_EXHAUSTIVE can be used to describe the increasing amount of
            effort spent during the planning stage to create the fastest
            possible transform. Usually, FFTW_MEASURE is a good compromise
            and is the default.
        nthreads : int, optional
            Tells how many threads to use when invoking FFTW or MKL. Default is
            the number of cores.

        """
        if (
            bandwidth is None
            and psd is not None
            or bandwidth is not None
            and psd is None
        ):
            raise ValueError('The bandwidth or the PSD is not specified.')
        if nep is not None:
            sigma = nep / np.sqrt(2 * sampling.period)
        if bandwidth is None and psd is None and sigma is None:
            raise ValueError('The noise model is not specified.')

        shapein = len(self), len(sampling)

        # handle the non-correlated case first
        if bandwidth is None and fknee == 0:
            return DiagonalOperator(
                1 / sigma**2, broadcast='rightward', shapein=shapein
            )

        sampling_frequency = 1 / sampling.period

        nsamples_max = len(sampling)
        fftsize = 2
        while fftsize < nsamples_max:
            fftsize *= 2

        new_bandwidth = sampling_frequency / fftsize
        if bandwidth is not None and psd is not None:
            if twosided:
                psd = _fold_psd(psd)
            f = np.arange(fftsize // 2 + 1, dtype=float) * new_bandwidth
            p = _unfold_psd(_logloginterp_psd(f, bandwidth, psd))
        else:
            p = _gaussian_psd_1f(
                fftsize, sampling_frequency, sigma, fknee, fslope, twosided=True
            )
        p[..., 0] = p[..., 1]
        invntt = _psd2invntt(p, new_bandwidth, ncorr, fftw_flag=fftw_flag)

        return SymmetricBandToeplitzOperator(
            shapein, invntt, fftw_flag=fftw_flag, nthreads=nthreads
        )

    def get_noise(
        self,
        sampling,
        psd=None,
        bandwidth=None,
        twosided=False,
        sigma=None,
        nep=None,
        fknee=0,
        fslope=1,
        out=None,
        operation=operation_assignment,
    ):
        """
        Return the noise realization following a given PSD.

        The input Power Spectrum Density can either be fully specified by using
        the 'bandwidth' and 'psd' keywords, or by providing the parameters of
        the gaussian distribution:
            psd = sigma**2 * (1 + (fknee/f)**fslope) / B
        where B is equal to sampling_frequency / N.

        Parameters
        ----------
        sampling : Sampling
            The temporal sampling.
        psd : array-like, optional
            The one-sided or two-sided Power Spectrum Density,
            [signal unit**2/Hz].
        bandwidth : float, optional
            The PSD frequency increment [Hz].
        twosided : boolean, optional
            Whether or not the output psd is one-sided (only positive
            frequencies) or two-sided (positive and negative frequencies).
        sigma : float, optional
            Standard deviation of the white noise component.
        fknee : float, optional
            The 1/f noise knee frequency [Hz].
        fslope : float, optional
            The 1/f noise slope.
        out : ndarray, optional
            Placeholder for the output noise.

        """
        if (
            bandwidth is None
            and psd is not None
            or bandwidth is not None
            and psd is None
        ):
            raise ValueError('The bandwidth or the PSD is not specified.')
        if nep is not None:
            sigma = nep / np.sqrt(2 * sampling.period)
        if bandwidth is None and psd is None and sigma is None:
            raise ValueError('The noise model is not specified.')
        if out is None and operation is not operation_assignment:
            raise ValueError('The output buffer is not specified.')

        shape = (len(self), len(sampling))

        # handle non-correlated case first
        if bandwidth is None and fknee == 0:
            sigma = np.atleast_1d(sigma)
            noise = np.random.standard_normal(shape) * sigma[:, None]
            if out is None:
                return noise
            operation(out, noise)
            return out

        if out is None:
            out = empty((len(self), len(sampling)))

        sampling_frequency = 1 / sampling.period

        # fold two-sided input PSD
        if bandwidth is not None and psd is not None:
            if twosided:
                psd = _fold_psd(psd)
                twosided = False
        else:
            twosided = True

        n = len(sampling)
        if bandwidth is None and psd is None:
            p = _gaussian_psd_1f(
                n, sampling_frequency, sigma, fknee, fslope, twosided=twosided
            )
        else:
            # log-log interpolation of one-sided PSD
            f = np.arange(n // 2 + 1, dtype=float) * (sampling_frequency / n)
            p = _logloginterp_psd(f, bandwidth, psd)

        # looping over the detectors
        for out_ in out:
            operation(
                out_, _gaussian_sample(n, sampling_frequency, p, twosided=twosided)
            )

        return out


class Imager(Instrument):
    """
    An Imager is an Instrument for which a relationship between the world
    coordinates of the object plane and the image plane does exist (unlike
    an interferometer).

    Attributes
    ----------
    object2image : Operator
        Transform from object plane to image plane coordinates.
    image2object : Operator
        Transform from image plane to object plane coordinates.

    """

    def __init__(self, name, layout, image2object=None, object2image=None):
        if image2object is None and object2image is None:
            raise ValueError(
                'Neither the image2object nor the object2image tr'
                'ansforms are speficied.'
            )
        Instrument.__init__(self, name, layout)
        if object2image is not None:
            self.object2image = asoperator(object2image)
            self.image2object = self.object2image.I
        else:
            self.image2object = asoperator(image2object)
            self.object2image = self.image2object.I
