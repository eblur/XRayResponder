import numpy as np
import os

from clarsach.respond import RMF, ARF, _Angs_keV
from astropy.io import fits
from astropy.units import si

__all__ = ['XSpectrum']

KEV      = ['kev','keV']
ANGS     = ['angs','angstrom','Angstrom','angstroms','Angstroms']

ALLOWED_UNITS = KEV + ANGS
ALLOWED_TELESCOPES = ['HETG','ACIS']

# Not a very smart reader, but it works for HETG
class XSpectrum(object):
    def __init__(self, filename, telescope='HETG'):
        assert telescope in ALLOWED_TELESCOPES

        self.__store_path(filename)

        if telescope == 'HETG':
            self._read_chandra(filename)
        elif telescope == 'ACIS':
            self._read_chandra(filename)

        if self.bin_unit != self.arf.e_unit:
            print("Warning: ARF units and pha file units are not the same!!!")

        if self.bin_unit != self.rmf.energ_unit:
            print("Warning: RMF units and pha file units are not the same!!!")

        return

    def __store_path(self, filename):
        self.path = '/'.join(filename.split('/')[0:-1]) + "/"
        return

    def apply_resp(self, mflux, exposure=None):
        """
        Given a model flux spectrum, apply the response. In cases where the
        spectrum has both an ARF and an RMF, apply both. Otherwise, apply
        whatever response is in RMF.

        The model flux spectrum *must* be created using the same units and
        bins as in the ARF (where the ARF exists)!

        Parameters
        ----------
        mflux : iterable
            A list or array with the model flux values in ergs/keV/s/cm^-2

        exposure : float, default None
            By default, the exposure stored in the ARF will be used to compute
            the total counts per bin over the effective observation time.
            In cases where this might be incorrect (e.g. for simulated spectra
            where the pha file might have a different exposure value than the
            ARF), this keyword provides the functionality to override the
            default behaviour and manually set the exposure time to use.

        Returns
        -------
        count_model : numpy.ndarray
            The model spectrum in units of counts/bin
        """

        if self.arf is not None:
            mrate  = self.arf.apply_arf(mflux, exposure=exposure)
        else:
            mrate = mflux

        count_model = self.rmf.apply_rmf(mrate)

        return count_model

    @property
    def bin_mid(self):
        return 0.5 * (self.bin_lo + self.bin_hi)

    def _read_chandra(self, filename):
        this_dir = os.path.dirname(os.path.abspath(filename))
        ff   = fits.open(filename)
        data = ff[1].data
        self.bin_lo   = data['BIN_LO']
        self.bin_hi   = data['BIN_HI']
        self.bin_unit = data.columns['BIN_LO'].unit
        self.counts   = data['COUNTS']
        self.exposure = ff[1].header['EXPOSURE']  # seconds
        # Attempt to read in ARFs and RMFs specified in FITS header
        try:
            rmf_prefix = ''
            rmf_header = ff[1].header['RESPFILE']
            if rmf_header[0] != '/':
                rmf_prefix = this_dir + "/"
            self.rmf_file = rmf_prefix + rmf_header
            print("Reading RMF from {}".format(self.rmf_file))
            self.rmf = RMF(self.rmf_file)
        except:
            print("RMF file not found, rmf values set to None")
            self.rmf_file = None
            self.rmf = None
        try:
            arf_prefix = ''
            arf_header = ff[1].header['ANCRFILE']
            if arf_header[0] != '/':
                arf_prefix = this_dir + "/"
            self.arf_file = arf_prefix + arf_header
            print("Reading ARF from {}".format(self.arf_file))
            self.arf = ARF(self.arf_file)
        except:
            print("No ARF file found, arf values set to None")
            self.arf_file = None
            self.arf = None
        ff.close()

    def _return_in_units(self, unit):
        assert unit in ALLOWED_UNITS
        if unit == self.bin_unit:
            return (self.bin_lo, self.bin_hi, self.bin_mid, self.counts)
        else:
            # Need to use reverse values if the bins are listed in increasing order
            new_lo, sl = _Angs_keV(self.bin_hi)
            new_hi, sl = _Angs_keV(self.bin_lo)
            new_mid = 0.5 * (new_lo + new_hi)
            new_cts = self.counts[sl]
            return (new_lo, new_hi, new_mid, new_cts)

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

        return

    def plot(self, ax, xunit='keV', **kwargs):
        lo, hi, mid, cts = self._change_units(xunit)
        counts_err       = np.sqrt(cts)
        ax.errorbar(mid, cts, yerr=counts_err,
                    ls='', marker=None, color='k', capsize=0, alpha=0.5)
        ax.step(lo, cts, where='post', **kwargs)
        ax.set_xlabel(UNIT_LABELS[xunit])
        ax.set_ylabel('Counts')
