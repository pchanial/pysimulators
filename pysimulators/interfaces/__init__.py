from __future__ import absolute_import

try:
    import healpy
    del healpy
except:
    pass
else:
    from . import healpy
from . import madmap1

del absolute_import
