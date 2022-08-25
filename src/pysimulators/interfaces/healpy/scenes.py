import numpy as np
from scipy.constants import c, h, k

from pyoperators import ReciprocalOperator, asoperator
from pyoperators.utils import strenum

from ...packedtables import Scene
from .operators import Spherical2HealpixOperator

__all__ = ['SceneHealpix', 'SceneHealpixCMB']


class SceneHealpix(Scene):
    """
    Class for Healpix scenes.

    """

    def __init__(
        self, nside, kind='I', nest=False, convention='zenith,azimuth', **keywords
    ):
        """
        nside : int
            Value of the map resolution parameter.
        kind : 'I', 'QU' or 'IQU'
            The kind of sky: intensity-only, Q and U Stokes parameters only or
            intensity, Q and U.
        nest : boolean, optional
            For the nested numbering scheme, set it to True. Default is
            the ring scheme.
        convention : string, optional
            Convention used by the 'topixel' operator to convert spherical
            coordinates into Healpix pixel numbers. It can be:
            'zenith,azimuth', 'azimuth,zenith', 'elevation,azimuth' and
            'azimuth,elevation'.

        """
        kinds = 'I', 'QU', 'IQU'
        if not isinstance(kind, str):
            raise TypeError(
                f'Invalid type {type(kind).__name__!r} for the scene kind. Expected '
                f'type is string.'
            )
        kind = kind.upper()
        if kind not in kinds:
            raise ValueError(
                f'Invalid scene kind {kind!r}. Expected kinds are: {strenum(kinds)}.'
            )
        nside = int(nside)
        topixel = Spherical2HealpixOperator(nside, convention, nest)
        shape = (12 * nside**2,)
        if kind != 'I':
            shape += (len(kind),)
        Scene.__init__(self, shape, topixel, ndim=1, **keywords)
        self.nside = nside
        self.npixel = 12 * nside**2
        self.kind = kind
        self.convention = convention
        self.nest = bool(nest)
        self.solid_angle = 4 * np.pi / self.npixel


class SceneHealpixCMB(SceneHealpix):
    def __init__(self, nside, kind='I', absolute=False, temperature=2.7255, **keywords):
        """
        Parameters
        ----------
        nside : int
            The Healpix scene's nside.
        kind : 'I', 'QU' or 'IQU', optional
            The sky kind: 'I' for intensity-only, 'QU' for Q and U maps,
            and 'IQU' for intensity plus QU maps.
        nest : boolean, optional
            For the nested numbering scheme, set it to True. Default is
            the ring scheme.
        absolute : boolean, optional
            If true, the scene pixel values include the CMB background and the
            fluctuations in units of Kelvin, otherwise it only represents the
            fluctuations, in microKelvin.
        temperature : float, optional
            The CMB black-body temperature which is used (if absolute is False)
            to convert a temperature fluctuation into power fluctuations.
            The default value is taken from Fixsen et al. 2009.

        """
        self.absolute = bool(absolute)
        self.temperature = float(temperature)
        SceneHealpix.__init__(self, nside, kind=kind, **keywords)

    def get_unit_conversion_operator(self, nu):
        """
        Return an operator to convert sky temperatures into
        W / m^2 / Hz / pixel.

        If the scene has been initialized with the 'absolute' keyword, the
        scene is assumed to include the CMB background and the fluctuations
        (in Kelvin) and the operator follows the non-linear Planck law.
        Otherwise, the scene only includes the fluctuations (in microKelvin)
        and the operator is linear (i.e. the output also corresponds to power
        fluctuations).

        Parameter
        ---------
        nu : float
            The frequency, at which the conversion is performed [Hz].

        Example
        -------
        > scene = SceneHealpixCMB(256, absolute=False)
        > op = scene.get_unit_conversion_operator(150e9)
        > dT = 200  # µK
        > print(op(dT))  # W / m^2 / Hz / pixel
        1.2734621598076659e-26

        > scene = SceneHealpixCMB(256, absolute=True)
        > op = scene.get_unit_conversion_operator(150e9)
        > T = 2.7  # K
        > print(op(T))  # W / m^2 / Hz / pixel
        5.94046610468e-23

        """
        a = 2 * self.solid_angle * h * nu**3 / c**2
        if self.absolute:
            hnu_k = h * nu / k
            return a / asoperator(np.expm1)(hnu_k * ReciprocalOperator())
        T = self.temperature
        hnu_kT = h * nu / (k * T)
        val = 1e-6 * a * hnu_kT * np.exp(hnu_kT) / (np.expm1(hnu_kT) ** 2 * T)
        return asoperator(val)
