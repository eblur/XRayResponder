import numpy as np
import os

from clarsach.respond import RMF, ARF, _Angs_keV
from astropy.io import fits
from astropy.units import si

CONST_HC = 12.398418573430595   # Copied from ISIS, [keV angs]
KEV      = ['kev','keV']
ANGS     = ['angs','angstrom','Angstrom','angstroms','Angstroms']

ALLOWED_UNITS = KEV + ANGS
ALLOWED_TELESCOPES = ['HETG','ACIS']

__all__ = ['XSpectrum']

# Not a very smart reader, but it works for HETG
class XSpectrum(object):
    def __init__(self, filename, telescope='HETG'):
        assert telescope in ALLOWED_TELESCOPES

        self.__store_path(filename)

        if telescope == 'HETG':
            self._read_chandra(filename)
        elif telescope == 'ACIS':
            self._read_chandra(filename)

        if self.bin_unit != self.arf.bin_unit:
            print("Warning: ARF units and pha file units are not the same!!!")

        if self.bin_unit != self.rmf.energ_unit:
            print("Warning: RMF units and pha file units are not the same!!!")

    def __store_path(self, filename):
        self.path = '/'.join(filename.split('/')[0:-1]) + "/"
        return

    def apply_resp(self, mflux):
        # Given a model flux spectrum, apply the response
        # mflux must be in units of phot/s/cm^2/keV
        # 1. Apply ARF
        phrate  = self.arf.apply_arf(mflux)  # phot/s/keV
        phots   = phrate * self.exposure * self.arf.fracexpo  # phot/keV
        # 2. Apply RMF
        result = self.rmf.apply_rmf(phots)  # counts per bin
        return result

    @property
    def bin_mid(self):
        return 0.5 * (self.bin_lo + self.bin_hi)

    def _return_in_units(self, unit):
        assert unit in ALLOWED_UNITS
        if unit == self.bin_unit:
            return (self.bin_lo.value, self.bin_hi.value, self.bin_mid.value, self.counts)
        else:
            # Need to use reverse values if the bins are listed in increasing order
            new_lo, sl = _Angs_keV(self.bin_hi)
            new_hi, sl = _Angs_keV(self.bin_lo)
            new_mid = 0.5 * (new_lo + new_hi)
            new_cts = self.counts[sl]
            return (new_lo.value, new_hi.value, new_mid.value, new_cts)

    def plot(self, ax, xunit=si.keV, **kwargs):
        lo, hi, mid, cts = self._return_in_units(xunit)
        counts_err       = np.sqrt(cts)
        ax.errorbar(mid, cts, yerr=counts_err,
                    ls='', marker=None, color='k', capsize=0, alpha=0.5)
        ax.step(lo, cts, where='post', **kwargs)
        ax.set_xlabel("%s" % xunit)
        ax.set_ylabel('Counts')

    def _read_chandra(self, filename):
        this_dir = os.path.dirname(os.path.abspath(filename))
        ff   = fits.open(filename)
        data = ff[1].data

        # Deal with units
        bin_unit = data.columns["BIN_LO"].unit
        if bin_unit in ANGS:
            self.bin_unit = si.Angstrom
        elif bin_unit in KEV:
            self.bin_unit = si.keV
        else:
            print("WARNING: %s is not a supported bin unit" % bin_unit)
            self.bin_unit = 1.0

        self.bin_lo   = np.array(data.field("BIN_LO")) * self.bin_unit
        self.bin_hi   = np.array(data.field("BIN_HI")) * self.bin_unit
        self.counts   = data['COUNTS']
        self.rmf_file = this_dir + "/" + ff[1].header['RESPFILE']
        self.arf_file = this_dir + "/" + ff[1].header['ANCRFILE']
        self.rmf = RMF(self.rmf_file)
        self.arf = ARF(self.arf_file)
        self.exposure = ff[1].header['EXPOSURE']  # seconds
        ff.close()

        # Let's just keep everything in keV units
        #if self.bin_unit in ANGS:
        #    self._setbins_to_keV()
        return

    def _setbins_to_keV(self):
        assert self.bin_unit in ANGS
        new_bhi, sl = _Angs_keV(self.bin_lo)
        new_blo, sl = _Angs_keV(self.bin_hi)
        new_cts  = self.counts[sl]

        # Now hard set everything
        self.bin_lo = new_blo
        self.bin_hi = new_bhi
        self.counts = new_cts
        self.bin_unit = si.keV
        return
