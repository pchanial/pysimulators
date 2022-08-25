# flake8: noqa: E402
import locale
from importlib.metadata import version as _version

# force gfortran's read statement to always use the dot sign as fraction
# separator (PR47007)
locale.setlocale(locale.LC_NUMERIC, 'POSIX')
del locale

from .acquisitions import Acquisition, MaskPolicy
from .beams import BeamGaussian, BeamUniformHalfSpace
from .datatypes import FitsArray, Map, Tod
from .datautils import (
    airy_disk,
    aperture_circular,
    distance,
    ds9,
    gaussian,
    integrated_profile,
    phasemask_fourquadrant,
    profile,
    profile_psd2,
    psd2,
)
from .instruments import Imager, Instrument
from .operators import (
    BlackBodyOperator,
    CartesianEquatorial2GalacticOperator,
    CartesianEquatorial2HorizontalOperator,
    CartesianGalactic2EquatorialOperator,
    CartesianHorizontal2EquatorialOperator,
    ConvolutionTruncatedExponentialOperator,
    PointingMatrix,
    PowerLawOperator,
    ProjectionInMemoryOperator,
    ProjectionOnFlyOperator,
    ProjectionOperator,
    RollOperator,
    SphericalEquatorial2GalacticOperator,
    SphericalEquatorial2HorizontalOperator,
    SphericalGalactic2EquatorialOperator,
    SphericalHorizontal2EquatorialOperator,
)
from .packedtables import (
    Layout,
    LayoutGrid,
    LayoutGridSquares,
    PackedTable,
    Sampling,
    SamplingEquatorial,
    SamplingHorizontal,
    SamplingSpherical,
    Scene,
    SceneGrid,
)
from .quantities import Quantity, UnitError, units
from .wcsutils import (
    DistortionOperator,
    RotationBoresightEquatorialOperator,
    WCSToPixelOperator,
    WCSToWorldOperator,
    angle_lonlat,
    barycenter_lonlat,
    create_fitsheader,
    create_fitsheader_for,
    fitsheader2shape,
    str2fitsheader,
)

__all__ = [
    'Acquisition',
    'MaskPolicy',
    'BeamGaussian',
    'BeamUniformHalfSpace',
    'FitsArray',
    'Map',
    'Tod',
    'airy_disk',
    'aperture_circular',
    'distance',
    'ds9',
    'gaussian',
    'integrated_profile',
    'phasemask_fourquadrant',
    'profile',
    'profile_psd2',
    'psd2',
    'Instrument',
    'Imager',
    'BlackBodyOperator',
    'CartesianEquatorial2GalacticOperator',
    'CartesianGalactic2EquatorialOperator',
    'CartesianEquatorial2HorizontalOperator',
    'CartesianHorizontal2EquatorialOperator',
    'ConvolutionTruncatedExponentialOperator',
    'PointingMatrix',
    'PowerLawOperator',
    'ProjectionOperator',
    'ProjectionInMemoryOperator',
    'ProjectionOnFlyOperator',
    'RollOperator',
    'SphericalEquatorial2GalacticOperator',
    'SphericalGalactic2EquatorialOperator',
    'SphericalEquatorial2HorizontalOperator',
    'SphericalHorizontal2EquatorialOperator',
    'PackedTable',
    'Layout',
    'LayoutGrid',
    'LayoutGridSquares',
    'Sampling',
    'SamplingSpherical',
    'SamplingEquatorial',
    'SamplingHorizontal',
    'Scene',
    'SceneGrid',
    'Quantity',
    'UnitError',
    'units',
    'angle_lonlat',
    'barycenter_lonlat',
    'create_fitsheader',
    'create_fitsheader_for',
    'fitsheader2shape',
    'str2fitsheader',
    'DistortionOperator',
    'RotationBoresightEquatorialOperator',
    'WCSToPixelOperator',
    'WCSToWorldOperator',
]
__version__ = _version('pysimulators')
