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
    def __init__(self, filename, telescope='HETG', row=None, arf=None, rmf=None, verbose=True):
        assert telescope in ALLOWED_TELESCOPES

        self.__store_path(filename)

        if telescope == 'HETG':
            self._read_hetg(filename, arf=arf, rmf=rmf, row=row)
        elif telescope == 'ACIS':
            self._read_acis(filename, arf=arf, rmf=rmf)

        if verbose:
            if (self.arf is not None) and (self.bin_unit != self.arf.e_unit):
                print("Warning: ARF units and pha file units are not the same!!!")

            if (self.rmf is not None) and (self.bin_unit != self.rmf.energ_unit):
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

        If no ARF file exists, it will return the model flux after applying the RMF
        If no RMF file exists, it will return the model flux after applying the ARF (with a warning)
        If no ARF and no RMF, it will return the model flux spectrum (with a warning)
        """

        if self.arf is not None:
            mrate  = self.arf.apply_arf(mflux, exposure=exposure)
        else:
            mrate = mflux

        if self.rmf is not None:
            count_model = self.rmf.apply_rmf(mrate)
            return count_model
        else:
            print("Caution: no response file specified")
            return mrate

    @property
    def bin_mid(self):
        return 0.5 * (self.bin_lo + self.bin_hi)

    def _read_acis(self, filename, arf=None, rmf=None):
        this_dir = os.path.dirname(os.path.abspath(filename))
        ff   = fits.open(filename)
        data = ff[1].data

        if rmf is not None:
            self.rmf_file = rmf
        else:
            self.rmf_file = _file_from_header(ff[1].header, 'RESPFILE', this_dir)

        if arf is not None:
            self.arf_file = arf
        else:
            self.arf_file = _file_from_header(ff[1].header, 'ANCRFILE', this_dir)

        self._assign_arf(self.arf_file)
        self._assign_rmf(self.rmf_file)

        self.counts = data['COUNTS']
        self.bin_lo = data['CHANNEL'] - 1.0  # it starts with 1
        self.bin_hi = data['CHANNEL']
        self.bin_unit = 'channel_number'
        self.exposure = ff[1].header['EXPOSURE']  # seconds
        ff.close()
        return

    def _read_hetg(self, filename, arf=None, rmf=None, row=None):
        TG_PART = {1:'HEG', 2:'MEG'}
        this_dir = os.path.dirname(os.path.abspath(filename))
        ff   = fits.open(filename)
        data = ff[1].data

        if row is not None:
            assert row > 0
            self.bin_lo = data['BIN_LO'][row-1]
            self.bin_hi = data['BIN_HI'][row-1]
            self.counts = data['COUNTS'][row-1]
            tgp, tgm    = data['TG_PART'][row-1], data['TG_M'][row-1]
            self.name   = "%s m=%d" % (TG_PART[tgp], tgm)
        else:
            self.bin_lo = data['BIN_LO']
            self.bin_hi = data['BIN_HI']
            self.counts = data['COUNTS']

        # Deal with ARF and RMF
        # Allow user to override file choice at the start, otherwise read filenames from header
        # By default, the arf and rmf will be set to None (see kwargs in __init__ function call)
        # If the filename specified is 'none', the ARF or RMF will be set to None
        # If the filename is not 'none', the appropriate file will be loaded automatically
        if rmf is not None:
            self.rmf_file = rmf
        else:
            self.rmf_file = _file_from_header(ff[1].header, 'RESPFILE', this_dir)

        if arf is not None:
            self.arf_file = arf
        else:
            self.arf_file = _file_from_header(ff[1].header, 'ANCRFILE', this_dir)

        self._assign_arf(self.arf_file)
        self._assign_rmf(self.rmf_file)
        self.bin_unit = data.columns['BIN_LO'].unit
        self.exposure = ff[1].header['EXPOSURE']  # seconds
        ff.close()

    def _assign_arf(self, arf_inp):
        if isinstance(arf_inp, str):
            self.arf = ARF(arf_inp)
        else:
            self.arf = arf_inp

    def _assign_rmf(self, rmf_inp):
        if isinstance(rmf_inp, str):
            self.rmf = RMF(rmf_inp)
        else:
            self.rmf = rmf_inp

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

## ------------ Helper functions

def _file_from_header(header, keyword, this_dir):
    ## Search the FITS header for a filename under the keyword specified
    ## If the FITS header does not include that keyword, returns None
    result = None
    try:
        fname = header[keyword]
        if fname != 'none':
            result = this_dir + "/" + fname
    except:
        pass
    return result
