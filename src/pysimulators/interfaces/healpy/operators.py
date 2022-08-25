try:
    import healpy as hp
    import healpy._healpy_pixel_lib as pixlib
except ImportError:
    hp = None
    pixlib = None
import numpy as np

from pyoperators import CompositionOperator, IdentityOperator, Operator
from pyoperators.flags import inplace, real, square, symmetric
from pyoperators.utils import pi, strenum

from ...sparse import FSRMatrix, SparseOperator

__all__ = [
    'Healpix2CartesianOperator',
    'Cartesian2HealpixOperator',
    'Healpix2SphericalOperator',
    'Spherical2HealpixOperator',
    'HealpixConvolutionGaussianOperator',
    'HealpixLaplacianOperator',
]


@real
class _HealPixCartesian(Operator):
    def __init__(self, nside, nest=False, dtype=float, **keywords):
        if hp is None:
            raise ImportError('The package healpy is not installed.')
        self.nside = int(nside)
        self.nest = bool(nest)
        Operator.__init__(self, dtype=dtype, **keywords)

    @staticmethod
    def _reshapehealpix(shape):
        return shape + (3,)

    @staticmethod
    def _reshapecartesian(shape):
        return shape[:-1]

    @staticmethod
    def _validatecartesian(shape):
        if len(shape) == 0 or shape[-1] != 3:
            raise ValueError('Invalid cartesian shape.')

    @staticmethod
    def _rule_identity(o1, o2):
        if o1.nside == o2.nside and o1.nest == o2.nest:
            return IdentityOperator()


class Healpix2CartesianOperator(_HealPixCartesian):
    """
    Convert Healpix pixels into cartesian coordinates.

    """

    def __init__(self, nside, nest=False, **keywords):
        """
        nside : int
            Value of the map resolution parameter.
        nest : boolean, optional
            For the nested numbering scheme, set it to True. Default is
            ring scheme.

        """
        super().__init__(
            nside,
            nest=nest,
            reshapein=self._reshapehealpix,
            reshapeout=self._reshapecartesian,
            validateout=self._validatecartesian,
            **keywords,
        )
        self.set_rule('I', lambda s: Cartesian2HealpixOperator(s.nside, nest=s.nest))
        self.set_rule(
            ('.', Cartesian2HealpixOperator), self._rule_identity, CompositionOperator
        )

    def direct(self, input, output):
        input = input.astype(int)
        func = pixlib._pix2vec_nest if self.nest else pixlib._pix2vec_ring
        func(self.nside, input, output[..., 0], output[..., 1], output[..., 2])


class Cartesian2HealpixOperator(_HealPixCartesian):
    """
    Convert cartesian coordinates into Healpix pixels.

    """

    def __init__(self, nside, nest=False, **keywords):
        """
        nside : int
            Value of the map resolution parameter.
        nest : boolean, optional
            For the nested numbering scheme, set it to True. Default is
            ring scheme.

        """
        super().__init__(
            nside,
            nest=nest,
            reshapein=self._reshapecartesian,
            reshapeout=self._reshapehealpix,
            validatein=self._validatecartesian,
            **keywords,
        )
        self.set_rule('I', lambda s: Healpix2CartesianOperator(s.nside, nest=s.nest))
        self.set_rule(
            ('.', Healpix2CartesianOperator), self._rule_identity, CompositionOperator
        )

    def direct(self, input, output):
        func = pixlib._vec2pix_nest if self.nest else pixlib._vec2pix_ring
        func(self.nside, input[..., 0], input[..., 1], input[..., 2], output)


@real
class _HealPixSpherical(Operator):
    CONVENTIONS = (
        'zenith,azimuth',
        'azimuth,zenith',
        'elevation,azimuth',
        'azimuth,elevation',
    )

    def __init__(self, nside, convention, nest=False, dtype=float, **keywords):
        if hp is None:
            raise ImportError('The package healpy is not installed.')
        if not isinstance(convention, str):
            raise TypeError(f"The input convention '{convention}' is not a string.")
        convention_ = convention.replace(' ', '').lower()
        if convention_ not in self.CONVENTIONS:
            raise ValueError(
                f'Invalid spherical convention {convention!r}. Expected values are '
                f'{strenum(self.CONVENTIONS)}.'
            )
        self.nside = int(nside)
        self.convention = convention_
        self.nest = bool(nest)
        Operator.__init__(self, dtype=dtype, **keywords)

    @staticmethod
    def _reshapehealpix(shape):
        return shape + (2,)

    @staticmethod
    def _reshapespherical(shape):
        return shape[:-1]

    @staticmethod
    def _validatespherical(shape):
        if len(shape) == 0 or shape[-1] != 2:
            raise ValueError('Invalid spherical shape.')

    @staticmethod
    def _rule_identity(o1, o2):
        if (
            o1.nside == o2.nside
            and o1.convention == o2.convention
            and o1.nest == o2.nest
        ):
            return IdentityOperator()


class Healpix2SphericalOperator(_HealPixSpherical):
    """
    Convert Healpix pixels into spherical coordinates in radians.


    The last dimension of the operator's output is 2 and it encodes
    the two spherical angles. Four conventions define what these angles are:
       - 'zenith,azimuth': (theta, phi) angles commonly used
       in physics or the (colatitude, longitude) angles used
       in the celestial and geographical coordinate systems
       - 'azimuth,zenith': (longitude, colatitude) convention
       - 'elevation,azimuth: (latitude, longitude) convention
       - 'azimuth,elevation': (longitude, latitude) convention

    """

    def __init__(self, nside, convention, nest=False, **keywords):
        """
        nside : int
            Value of the map resolution parameter.
        convention : string
            One of the following spherical coordinate conventions:
            'zenith,azimuth', 'azimuth,zenith', 'elevation,azimuth' and
            'azimuth,elevation'.
        nest : boolean, optional
            For the nested numbering scheme, set it to True. Default is
            ring scheme.

        """
        super().__init__(
            nside,
            convention,
            nest=nest,
            reshapein=self._reshapehealpix,
            reshapeout=self._reshapespherical,
            validateout=self._validatespherical,
            **keywords,
        )
        self.set_rule(
            'I', lambda s: Spherical2HealpixOperator(s.nside, s.convention, nest=s.nest)
        )
        self.set_rule(
            ('.', Spherical2HealpixOperator), self._rule_identity, CompositionOperator
        )

    def direct(self, input, output):
        input = input.astype(int)
        func = pixlib._pix2ang_nest if self.nest else pixlib._pix2ang_ring
        if self.convention.startswith('azimuth'):
            o1, o2 = output[..., 1], output[..., 0]
        else:
            o1, o2 = output[..., 0], output[..., 1]
        func(self.nside, input, o1, o2)
        if 'elevation' in self.convention:
            np.subtract(0.5 * pi(self.dtype), o1, o1)


class Spherical2HealpixOperator(_HealPixSpherical):
    """
    Convert spherical coordinates in radians into Healpix pixels.

    The last dimension of the operator's output is 2 and it encodes
    the two spherical angles. Four conventions define what these angles are:
       - 'zenith,azimuth': (theta, phi) angles commonly used
       in physics or the (colatitude, longitude) angles used
       in the celestial and geographical coordinate systems
       - 'azimuth,zenith': (longitude, colatitude) convention
       - 'elevation,azimuth: (latitude, longitude) convention
       - 'azimuth,elevation': (longitude, latitude) convention

    """

    def __init__(self, nside, convention, nest=False, **keywords):
        """
        nside : int
            Value of the map resolution parameter.
        convention : string
            One of the following spherical coordinate conventions:
            'zenith,azimuth', 'azimuth,zenith', 'elevation,azimuth' and
            'azimuth,elevation'.
        nest : boolean, optional
            For the nested numbering scheme, set it to True. Default is
            ring scheme.

        """
        super().__init__(
            nside,
            convention,
            nest=nest,
            reshapein=self._reshapespherical,
            reshapeout=self._reshapehealpix,
            validatein=self._validatespherical,
            **keywords,
        )
        self.set_rule(
            'I', lambda s: Healpix2SphericalOperator(s.nside, s.convention, nest=s.nest)
        )
        self.set_rule(
            ('.', Healpix2SphericalOperator), self._rule_identity, CompositionOperator
        )

    def direct(self, input, output):
        func = pixlib._ang2pix_nest if self.nest else pixlib._ang2pix_ring
        if self.convention.startswith('azimuth'):
            i1, i2 = input[..., 1], input[..., 0]
        else:
            i1, i2 = input[..., 0], input[..., 1]
        if 'elevation' in self.convention:
            i1 = 0.5 * pi(self.dtype) - i1
        func(self.nside, i1, i2, output)


@inplace
@real
@square
@symmetric
class HealpixConvolutionGaussianOperator(Operator):
    """
    Convolve a Healpix map by a gaussian kernel.

    """

    def __init__(
        self,
        fwhm=None,
        sigma=None,
        iter=3,
        lmax=None,
        mmax=None,
        use_weights=False,
        datapath=None,
        dtype=float,
        pol=True,
        **keywords,
    ):
        """
        Keywords are passed to the Healpy function smoothing.

        """
        if hp is None:
            raise ImportError('The package healpy is not installed.')
        if fwhm is None and sigma is None:
            raise ValueError('The convolution width is not specified.')
        if fwhm is not None and sigma is not None:
            raise ValueError('Ambiguous convolution width specification.')
        if fwhm is not None and fwhm < 0 or sigma is not None and sigma < 0:
            raise ValueError('The convolution width is not positive.')
        if fwhm in (0, None) and sigma in (0, None):
            self.__class__ = IdentityOperator
            self.__init__(**keywords)
            return
        if fwhm is None:
            fwhm = sigma * np.sqrt(8 * np.log(2))
        if sigma is None:
            sigma = fwhm / np.sqrt(8 * np.log(2))
        self.fwhm = fwhm
        self.sigma = sigma
        self.iter = iter
        self.lmax = lmax
        self.mmax = mmax
        self.use_weights = use_weights
        self.datapath = datapath
        self.pol = pol
        Operator.__init__(self, dtype=dtype, **keywords)

    def direct(self, input, output):
        if input.ndim > 1:
            input = [_ for _ in input.T]
        output.T[...] = hp.smoothing(
            input,
            fwhm=self.fwhm,
            iter=self.iter,
            lmax=self.lmax,
            mmax=self.mmax,
            use_weights=self.use_weights,
            datapath=self.datapath,
            pol=self.pol,
            verbose=False,
        )

    def validatein(self, shape):
        if len(shape) == 0 or len(shape) > 2:
            raise ValueError('Invalid number of dimensions.')
        nside = int(np.round(np.sqrt(shape[0] / 12)))
        if 12 * nside**2 != shape[0]:
            raise ValueError(
                f'The nside value cannot be inferred from the input number of pixels '
                f"'{shape[0]}'."
            )


@real
@square
@symmetric
class HealpixLaplacianOperator(SparseOperator):
    """
    9-Point stencil laplacian for Healpix maps.

    """

    def __init__(self, nside):
        """
        Parameters
        ----------
        nside : integer
            The Healpix map nside.

        """
        if hp is None:
            raise ImportError('The package healpy is not installed.')
        npix = 12 * nside**2
        ipix = np.arange(npix, dtype=np.int32)
        neighbours = hp.get_all_neighbours(nside, ipix)
        s = FSRMatrix((npix, npix), ncolmax=9, dtype_index=np.int32)
        s.data.index[:, 0] = ipix
        s.data.index[:, 1:] = neighbours.T
        h2 = 4 * np.pi / npix
        s.data.value[:, 0] = -20 / (6 * h2)
        s.data.value[:, 1:] = np.array([1, 4, 1, 4, 1, 4, 1, 4]) / (6 * h2)
        self.nside = nside
        SparseOperator.__init__(self, s)
