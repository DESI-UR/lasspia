from astropy.io import fits
import numpy as np

def openSDSS(files):
    """Return galaxy catalog from Sloan Digital Sky Survey files."""
    return np.concatenate([
        fits.open(f)[1].data
        for f in files
    ])


class wrapDR9(object):
    """Catalog wrapper interface for SDSS DR9 Catalogs.

    Wrappers must provide the following properties:
    * weight
    * z: redshift
    * ra: right ascension
    * dec: declination
    """
    def __init__(self, files, shiftRA=False):
        self.ctlg = openSDSS(files)
        self.shiftRA = shiftRA

    @property
    def z(self): return self.ctlg['z']

    @property
    def ra(self):
        a = self.ctlg['ra']
        return np.where(a < 180, a, a - 360) if self.shiftRA else a

    @property
    def dec(self): return self.ctlg['dec']

    @property
    def weight(self):
        return np.ones(len(self.ctlg), dtype=int)

    def __len__(self): return len(self.ctlg)


class wrapRandomDR9(wrapDR9):
    """Catalog wrapper with SDSS random catalog weights."""
    def __init__(self, files, shiftRA=False):
        super(wrapRandomDR9, self).__init__(files,shiftRA)

    @property
    def weight(self): return self.weightZ * self.weightNoZ

    @property
    def weightZ(self):
        '''Z-dependent weight.'''
        return self.ctlg['weight']

    @property
    def weightNoZ(self):
        '''Z-independent weight.'''
        return np.ones(len(self.ctlg), dtype=int)

class wrapObservedDR9(wrapDR9):
    """Catalog wrapper with SDSS observed catalog weights."""
    def __init__(self, files, shiftRA=False):
        super(wrapObservedDR9, self).__init__(files,shiftRA)

    @property
    def weight(self):
        ct = self.ctlg
        return (ct['weight_sdc'] * ct['weight_fkp'] *
                (ct['weight_noz'] + ct['weight_cp'] - 1))
