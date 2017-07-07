# Contains functionality for responses

import numpy as np
import astropy.io.fits as fits

__all__ = ["RMF", "ARF"]

CONST_HC    = 12.398418573430595   # Copied from ISIS, [keV angs]
ALLOWED_UNITS      = ['keV','angs','angstrom','kev']

class RMF(object):

    def __init__(self, filename):

        self._load_rmf(filename)
        pass

    def _load_rmf(self, filename):
        """
        Load an RMF from a FITS file.

        Parameters
        ----------
        filename : str
            The file name with the RMF file

        Attributes
        ----------
        n_grp : numpy.ndarray
            the Array with the number of channels in each
            channel set

        f_chan : numpy.ndarray
            The starting channel for each channel group;
            If an element i in n_grp > 1, then the resulting
            row entry in f_chan will be a list of length n_grp[i];
            otherwise it will be a single number

        n_chan : numpy.ndarray
            The number of channels in each channel group. The same
            logic as for f_chan applies

        matrix : numpy.ndarray
            The redistribution matrix as a flattened 1D vector

        energ_lo : numpy.ndarray
            The lower edges of the energy bins

        energ_hi : numpy.ndarray
            The upper edges of the energy bins

        detchans : int
            The number of channels in the detector

        """
        # open the FITS file and extract the MATRIX extension
        # which contains the redistribution matrix and
        # anxillary information
        hdulist = fits.open(filename)

        # get all the extension names
        extnames = np.array([h.name for h in hdulist])

        # figure out the right extension to use
        if "MATRIX" in extnames:
            h = hdulist["MATRIX"]

        elif "SPECRESP MATRIX" in extnames:
            h = hdulist["SPECRESP MATRIX"]

        data = h.data
        hdr = h.header
        hdulist.close()

        # extract + store the attributes described in the docstring
        n_grp = np.array(data.field("N_GRP"))
        f_chan = np.array(data.field('F_CHAN'))
        n_chan = np.array(data.field("N_CHAN"))
        matrix = np.array(data.field("MATRIX"))

        self.energ_lo = np.array(data.field("ENERG_LO"))
        self.energ_hi = np.array(data.field("ENERG_HI"))
        self.energ_unit = data.columns["ENERG_LO"].unit
        self.detchans = hdr["DETCHANS"]
        self.offset = self.__get_tlmin(h)

        # flatten the variable-length arrays
        self.n_grp, self.f_chan, self.n_chan, self.matrix = \
                self._flatten_arrays(n_grp, f_chan, n_chan, matrix)

        return

    def __get_tlmin(self, h):
        """
        Get the tlmin keyword for `F_CHAN`.

        Parameters
        ----------
        h : an astropy.io.fits.hdu.table.BinTableHDU object
            The extension containing the `F_CHAN` column

        Returns
        -------
        tlmin : int
            The tlmin keyword
        """
        # get the header
        hdr = h.header
        # get the keys of all
        keys = np.array(list(hdr.keys()))

        # find the place where the tlmin keyword is defined
        t = np.array(["TLMIN" in k for k in keys])

        # get the index of the TLMIN keyword
        tlmin_idx = np.hstack(np.where(t))[0]

        # get the corresponding value
        tlmin = np.int(list(hdr.items())[tlmin_idx][1])

        return tlmin

    def _flatten_arrays(self, n_grp, f_chan, n_chan, matrix):

        if not len(n_grp) == len(f_chan) == len(n_chan) == len(matrix):
            raise ValueError("Arrays must be of same length!")

        # find all non-zero groups
        nz_idx = (n_grp > 0)

        # stack all non-zero rows in the matrix
        matrix_flat = np.hstack(matrix[nz_idx])

        # stack all nonzero rows in n_chan and f_chan
        n_chan_flat = np.hstack(n_chan[nz_idx])
        f_chan_flat = np.hstack(f_chan[nz_idx])

        # if n_chan is zero, we'll remove those as well.
        nz_idx2 = (n_chan_flat > 0)
        n_chan_flat = n_chan_flat[nz_idx2]
        f_chan_flat = f_chan_flat[nz_idx2]

        return n_grp, f_chan_flat, n_chan_flat, matrix_flat

    def apply_rmf(self, spec):
        """
        Fold the spectrum through the redistribution matrix.

        The redistribution matrix is saved as a flattened 1-dimensional
        vector to save space. In reality, for each entry in the flux
        vector, there exists one or more sets of channels that this
        flux is redistributed into. The additional arrays `n_grp`,
        `f_chan` and `n_chan` store this information:
            * `n_group` stores the number of channel groups for each
              energy bin
            * `f_chan` stores the *first channel* that each channel
              for each channel set
            * `n_chan` stores the number of channels in each channel
              set

        As a result, for a given energy bin i, we need to look up the
        number of channel sets in `n_grp` for that energy bin. We
        then need to loop over the number of channel sets. For each
        channel set, we look up the first channel into which flux
        will be distributed as well as the number of channels in the
        group. We then need to also loop over the these channels and
        actually use the corresponding elements in the redistribution
        matrix to redistribute the photon flux into channels.

        All of this is basically a big bookkeeping exercise in making
        sure to get the indices right.

        Parameters
        ----------
        spec : numpy.ndarray
            The (model) spectrum to be folded

        Returns
        -------
        counts : numpy.ndarray
            The (model) spectrum after folding, in
            counts/s/channel

        """
        # get the number of channels in the data
        nchannels = spec.shape[0]

        # an empty array for the output counts
        counts = np.zeros(nchannels)

        # index for n_chan and f_chan incrementation
        k = 0

        # index for the response matrix incrementation
        resp_idx = 0

        # loop over all channels
        for i in range(nchannels):

            # this is the current bin in the flux spectrum to
            # be folded
            source_bin_i = spec[i]

            # get the current number of groups
            current_num_groups = self.n_grp[i]

            # loop over the current number of groups
            for j in range(current_num_groups):
                # get the right index for the start of the counts array
                # to put the data into
                counts_idx = int(self.f_chan[k] - self.offset)

                # this is the current number of channels to use
                current_num_chans = int(self.n_chan[k])

                # iterate k for next round
                k += 1

                # add the flux to the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels
                counts[counts_idx:counts_idx + current_num_chans] += self.matrix[
                                                              resp_idx:resp_idx + current_num_chans] * \
                                                              np.float(
                                                                  source_bin_i)
                # iterate the response index for next round
                resp_idx += current_num_chans

        return counts


class ARF(object):

    def __init__(self, filename):

        self._load_arf(filename)
        pass

    def _load_arf(self, filename):
        """
        Load an ARF from a FITS file.

        Parameters
        ----------
        filename : str
            The file name with the RMF file

        Attributes
        ----------

        """
        # open the FITS file and extract the MATRIX extension
        # which contains the redistribution matrix and
        # anxillary information
        hdulist = fits.open(filename)
        h = hdulist["SPECRESP"]
        data = h.data
        hdr = h.header
        hdulist.close()

        # extract + store the attributes described in the docstring

        self.bin_lo  = np.array(data.field("BIN_LO"))
        self.bin_hi = np.array(data.field("BIN_HI"))
        self.bin_unit = data.columns["BIN_LO"].unit
        self.specresp = np.array(data.field("SPECRESP"))

        if "FRACEXPO" in data.columns.names:
            self.fracexpo = data["FRACEXPO"]
        else:
            self.fracexpo = 1.0
        return

    @property
    def is_monotonically_increasing(self):
        return all(self.bin_lo[1:] > self.bin_lo[:-1])

    def _change_units(self, unit):
        assert unit in ALLOWED_UNITS
        if unit == self.bin_unit:
            return (self.bin_lo, self.bin_hi, self.bin_mid, self.counts)
        else:
            # Need to use reverse values if the bins are listed in increasing order
            if self.is_monotonically_increasing:
                sl  = slice(None, None, -1)
            # Sometimes its listed in reverse angstrom values (to match energies),
            # in which case, no need to reverse
            else:
                sl  = slice(None, None, 1)
            new_lo  = CONST_HC/self.bin_hi[sl]
            new_hi  = CONST_HC/self.bin_lo[sl]
            new_eff = self.specresp[sl]
            new_frac = self.fracexpo  # 1 if there is no fracexpo set
            if np.size(self.fracexpo > 1):
                new_frac = self.fracexpo[sl]
            return (new_lo, new_hi, new_eff, new_frac)

    def hard_set_units(self, unit):
        new_lo, new_hi, new_eff, new_frac = self._change_units(unit)
        self.bin_lo = new_lo
        self.bin_hi = new_hi
        self.bin_unit = unit
        self.specresp = new_eff
        self.fracexpo = new_frac

    def apply_arf(self, spec):
        """
        Fold the spectrum through the ARF.

        The ARF is a single vector encoding the effective area information
        about the detector. A such, applying the ARF is a simple
        multiplication with the input spectrum.

        Parameters
        ----------
        spec : numpy.ndarray
            The (model) spectrum to be folded

        Returns
        -------
        s_arf : numpy.ndarray
            The (model) spectrum after folding, in
            counts/s/channel

        """
        assert spec.shape[0] == self.specresp.shape[0], "The input spectrum must " \
                                                      "be of same size as the " \
                                                      "ARF array."

        return np.array(spec) * self.specresp
