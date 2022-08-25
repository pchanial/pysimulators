import numexpr as ne
import numpy as np

from pyoperators.utils import reshape_broadcast

__all__ = ['BeamGaussian', 'BeamUniformHalfSpace']


class Beam:
    def __init__(self, solid_angle):
        """
        Parameter
        ---------
        solid_angle : float
            The beam solid angle [sr].

        """
        self.solid_angle = float(solid_angle)

    def __call__(self, theta_rad, phi_rad):
        raise NotImplementedError()

    def healpix(self, nside):
        """
        Return the beam as a Healpix map.

        Parameter
        ---------
        nside : int
             The Healpix map's nside.

        """
        import healpy as hp

        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        return self(theta, phi)


class BeamGaussian(Beam):
    """
    Axisymmetric gaussian beam.

    """

    def __init__(self, fwhm, backward=False):
        """
        Parameters
        ----------
        fwhm : float
            The Full-Width-Half-Maximum of the beam, in radians.
        backward : boolean, optional
            If true, the maximum of the beam is at theta=pi.

        """
        self.fwhm = fwhm
        self.sigma = fwhm / np.sqrt(8 * np.log(2))
        self.backward = bool(backward)
        Beam.__init__(self, 2 * np.pi * self.sigma**2)

    def __call__(self, theta, phi):
        if self.backward:
            theta = np.pi - theta
        coef = -0.5 / self.sigma**2
        out = ne.evaluate(
            'exp(coef * theta**2)', local_dict={'coef': coef, 'theta': theta}
        )
        return reshape_broadcast(out, np.broadcast(theta, phi).shape)


class BeamUniformHalfSpace(Beam):
    """
    Uniform beam in half-space.

    """

    def __init__(self):
        Beam.__init__(self, 2 * np.pi)

    def __call__(self, theta, phi):
        out = 1.0
        return reshape_broadcast(out, np.broadcast(theta, phi).shape)
