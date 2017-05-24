import numpy as np
import os

from respond import RMF, ARF
from astropy.io import fits

ALLOWED_UNITS      = ['keV','angs','angstrom','kev']
ALLOWED_TELESCOPES = ['HETG','ACIS']

CONST_HC    = 12.398418573430595   # Copied from ISIS, [keV angs]
UNIT_LABELS = dict(zip(ALLOWED_UNITS, ['Energy (keV)', 'Wavelength (angs)']))

__all__ = ['XSpectrum']

# Not a very smart reader, but it works for HETG
class XSpectrum(object):
    def __init__(self, filename, telescope='HETG'):
        assert telescope in ALLOWED_TELESCOPES
        if telescope == 'HETG':
            self._read_chandra(filename)
        elif telescope == 'ACIS':
            self._read_chandra(filename)

        if self.bin_unit != self.arf.e_unit:
            print("Warning: ARF units and pha file units are not the same!!!")

        if self.bin_unit != self.rmf.energ_unit:
            print("Warning: RMF units and pha file units are not the same!!!")

        # Might need to notice or group some day
        #self.notice = np.ones(len(self.counts), dtype=bool)
        #self.group  = np.zeros(len(self.counts), dtype=int)

    @property
    def bin_mid(self):
        return 0.5 * (self.bin_lo + self.bin_hi)

    @property
    def is_monotonically_increasing(self):
        return all(self.bin_lo[1:] > self.bin_lo[:-1])

    def _change_units(self, unit):
        assert unit in ALLOWED_UNITS
        if unit == self.bin_unit:
            return (self.bin_lo, self.bin_hi, self.bin_mid, self.counts)
        else:
            if self.is_monotonically_increasing:
                sl  = slice(None, None, -1)
            else:
                sl  = slice(None, None, 1)
            new_lo  = CONST_HC/self.bin_hi[sl]
            new_hi  = CONST_HC/self.bin_lo[sl]
            new_mid = 0.5 * (new_lo + new_hi)
            new_cts = self.counts[sl]
            return (new_lo, new_hi, new_mid, new_cts)

    def hard_set_units(self, unit):
        new_lo, new_hi, new_mid, new_cts = self._change_units(unit)
        self.bin_lo = new_lo
        self.bin_hi = new_hi
        self.counts = new_cts
        self.bin_unit = unit

    def plot(self, ax, xunit='keV', ret=False, **kwargs):
        lo, hi, mid, cts = self._change_units(xunit)
        counts_err       = np.sqrt(cts)
        ax.errorbar(mid, cts, yerr=counts_err,
                    ls='', marker=None, color='k', capsize=0, alpha=0.5)
        ax.step(lo, cts, where='post', **kwargs)
        ax.set_xlabel(UNIT_LABELS[xunit])
        ax.set_ylabel('Counts')
        if ret:
            return dict(zip(['lo','hi','mid','cts','cts_err'],
                            [lo, hi, mid, cts, counts_err]))

    def _eff_exposure(self):
        no_mod  = np.ones(len(self.arf.specresp))
        area    = self.arf.apply_arf(no_mod)
        eff_exp = self.rmf.apply_rmf(area)
        eff_exp *= self.exposure * self.arf.fracexpo
        return eff_exp  # cm^2 sec count phot^-1

    def plot_unfold(self, ax, xunit='keV', ret=False, **kwargs):
        eff_exp  = self._eff_exposure()  # cm^2 sec count phot^-1

        # Have to take account of zero values in effective exposure
        flux, f_err = np.zeros(len(eff_exp)), np.zeros(len(eff_exp))
        ii        = (eff_exp != 0.0)
        flux[ii]  = self.counts[ii] / eff_exp[ii]
        f_err[ii] = np.sqrt(self.counts[ii]) / eff_exp[ii]

        # Now deal with desired xunit
        lo, hi, mid, cts = self._change_units(xunit)
        if self.bin_unit != xunit:
            flx, fe = flux[::-1], f_err[::-1]
        else:
            flx, fe = flux, f_err
        ax.errorbar(mid, flx, yerr=fe,
                    ls='', marker=None, color='k', capsize=0, alpha=0.5)
        ax.step(lo, flx, where='post', **kwargs)
        ax.set_xlabel(UNIT_LABELS[xunit])
        ax.set_ylabel('Flux [phot cm$^{-2}$ s$^{-1}$ bin$^{-1}$]')
        if ret:
            return dict(zip(['lo','hi','mid','flx','flx_err'],
                            [lo, hi, mid, flx, fe]))

    def _read_chandra(self, filename):
        this_dir = os.path.dirname(os.path.abspath(filename))
        ff   = fits.open(filename)
        data = ff[1].data
        self.bin_lo   = data['BIN_LO']
        self.bin_hi   = data['BIN_HI']
        self.bin_unit = data.columns['BIN_LO'].unit
        self.counts   = data['COUNTS']
        self.rmf_file = this_dir + "/" + ff[1].header['RESPFILE']
        self.arf_file = this_dir + "/" + ff[1].header['ANCRFILE']
        self.rmf = RMF(self.rmf_file)
        self.arf = ARF(self.arf_file)
        self.exposure = ff[1].header['EXPOSURE']  # seconds
        ff.close()
