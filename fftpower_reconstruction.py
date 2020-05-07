"""
This module is a modified version of the fftpower module from nbodykit v3.9
to enable computation of the power spectrum and its monopoles for reconstructed
catalogs.

The classes implementing this receive 2 instead of 1 catalog, corresponding to
the displaced catalog, obtained by the reconstruction algorithm action on the
original objects, and the corresponding shifted randoms catalogs.

From these two catalogs, the reconstructed overdensity field is computed as
the difference between the interpolated densities of displaced and random catalog,
and then the power spectrum computation proceeds identically to the usual case.

As things were done here, it is not possible to compute cross-correlations,
unfortunately, but this might be done in the future to enable propagator computation, e.g.
"""
import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.binned_statistic import BinnedStatistic
from nbodykit.meshtools import SlabIterator
from nbodykit.base.catalog import CatalogSourceBase
from nbodykit.base.mesh import MeshSource
from nbodykit.algorithms.fftpower import FFTBase, FFTPower, ProjectedFFTPower
from nbodykit.algorithms.fftpower import project_to_basis, _cast_source, _find_unique_edges

class FFTBaseReconstruction(FFTBase):
    """
    Base class provides functions for periodic FFT based Power spectrum code for reconstructed catalog.
    Parameters
    ----------
    displaced : CatalogSource
        displaced particles
    random : CatalogSource
        random particles
    Nmesh : int, 3-vector
        the number of cells per mesh size
    BoxSize : 3-vector
        the size of the box
    """
    def __init__(self, displaced, random, Nmesh, BoxSize):
        super(FFTBaseReconstruction, self).__init__(self, displaced, random, Nmesh, BoxSize)

    def _compute_3d_power(self, displaced, random):
        """
        Compute and return the power as a function of k vector, for given reconstructed objects
        (displaced) and shifted randoms (random).
        Returns
        -------
        p3d : array_like (complex)
            the 3D complex array holding the power spectrum
        attrs : dict
            meta data of the 3d power
        """
        attrs = {}
        # add self.attrs
        attrs.update(self.attrs)

        delta_d = displaced.compute(mode='complex', Nmesh=self.attrs['Nmesh'])
        delta_s = random.compute(mode='complex', Nmesh=self.attrs['Nmesh'])

        delta = delta_d - delta_s

        c1 = delta
        c2 = delta

        # calculate the 3d power spectrum, slab-by-slab to save memory
        p3d = c1
        for (s0, s1, s2) in zip(p3d.slabs, c1.slabs, c2.slabs):
            s0[...] = s1 * s2.conj()

        for i, s0 in zip(p3d.slabs.i, p3d.slabs):
            # clear the zero mode.
            mask = True
            for i1 in i:
                mask = mask & (i1 == 0)
            s0[mask] = 0

        # the complex field is dimensionless; power is L^3
        # ref to http://icc.dur.ac.uk/~tt/Lectures/UA/L4/cosmology.pdf
        p3d[...] *= self.attrs['BoxSize'].prod()

        # get the number of objects (in a safe manner)
        #N1 = c1.attrs.get('N', 0)
        #N2 = c2.attrs.get('N', 0)
        #attrs.update({'N1':N1, 'N2':N2})

        # add shotnoise (nonzero only for auto-spectra)
        Pshot = 0.
        if 'shotnoise' in delta_d.attrs:
            Pshot += delta_d.attrs['shotnoise']
        if 'shotnoise' in delta_s.attrs:
            Pshot += delta_s.attrs['shotnoise']
        #if self.first is self.second:
        #    if 'shotnoise' in c1.attrs:
         #       Pshot = c1.attrs['shotnoise']
        attrs['shotnoise'] = Pshot


class FFTPowerReconstruction(FFTPower, FFTBaseReconstruction):
    """
    Algorithm to compute the 1d or 2d power spectrum and/or multipoles
    for reconstructed catalogs in a periodic box, using a Fast Fourier Transform (FFT).
    This computes the power spectrum as the square of the Fourier modes of the
    density field, estimated as difference of the displaced objects and shifted randoms
    density fields, all of them computed via a FFT.
    Results are computed when the object is inititalized. See the documenation
    of :func:`~FFTPower.run` for the attributes storing the results.

    Parameters
    ----------
    displaced : CatalogSource
        displaced particles
    random : CatalogSource
        shifted random particles
    mode : {'1d', '2d'}
        compute either 1d or 2d power spectra
    Nmesh : int, optional
        the number of cells per side in the particle mesh used to paint the source
    BoxSize : int, 3-vector, optional
        the size of the box
    los : array_like , optional
        the direction to use as the line-of-sight; must be a unit vector
    Nmu : int, optional
        the number of mu bins to use from :math:`\mu=[0,1]`;
        if `mode = 1d`, then ``Nmu`` is set to 1
    dk : float, optional
        the linear spacing of ``k`` bins to use; if not provided, the
        fundamental mode  of the box is used; if `dk=0` is set, use fine bins
        such that the modes contributing to the bin has identical modulus.
    kmin : float, optional
        the lower edge of the first ``k`` bin to use
    kmin : float, optional
        the upper limit of the last ``k`` bin to use (not exact)
    poles : list of int, optional
        a list of multipole numbers ``ell`` to compute :math:`P_\ell(k)`
        from :math:`P(k,\mu)`
    """
    logger = logging.getLogger('FFTPower')

    def __init__(self, displaced, random, mode, Nmesh=None, BoxSize=None,
                 los=[0, 0, 1], Nmu=5, dk=None, kmin=0., kmax=None, poles=[]):

        super(FFTPowerReconstruction, self).__init__(displaced, second=random, mode=mode,
                                                     Nmesh=Nmesh, BoxSize=BoxSize, los=los,
                                                     Nmu=Nmu, dk=dk, kmin=kmin, kmax=kmax, poles=poles)


class ProjectedFFTPowerReconstruction(ProjectedFFTPower, FFTBaseReconstruction):
    """
    The power spectrum of a reconstructed field (estimated as the difference
    between the displaced objects and shifted randoms densities) in a periodic box,
    projected over certain axes.
    This is not really always physically meaningful, but convenient for
    making sense of Lyman-Alpha forest or lensing maps.
    This is usually called the 1d power spectrum or 2d power spectrum.
    Results are computed when the object is inititalized. See the documenation
    of :func:`~ProjectedFFTPower.run` for the attributes storing the results.
    Parameters
    ----------
    displaced: CatalogSource, MeshSource
        the source displaced particles/field. if a CatalogSource is provided, it
        is automatically converted to MeshSource using the default painting
        parameters (via :func:`~nbodykit.base.catalogmesh.CatalogMesh.to_mesh`)
    Nmesh : int, optional
        the number of cells per side in the particle mesh used to paint the source
    BoxSize : int, 3-vector, optional
        the size of the box
    second : CatalogSource, MeshSource
        the associated random shifted particles
    axes : tuple
        axes to measure the power on. The axes not in the list will be averaged out.
        For example:
        - (0, 1) : project to x,y and measure power
        - (0) : project to x and measure power.
    dk : float, optional
        the linear spacing of ``k`` bins to use; if not provided, the
        fundamental mode  of the box is used
    kmin : float, optional
        the lower edge of the first ``k`` bin to use
    """
    logger = logging.getLogger('ProjectedFFTPowerReconstruction')

    def __init__(self, displaced, random, Nmesh=None, BoxSize=None,
                 axes=(0, 1), dk=None, kmin=0.):
        super(ProjectedFFTPowerReconstruction, self).__init__(first=displaced, second=random,
                                                              Nmesh=Nmesh, BoxSize=BoxSize,
                                                              axes=axes, dk=dk, kmin=kmin)

    def run(self):
        """
        Run the algorithm. This attaches the following attributes to the class:
        - :attr:`edges`
        - :attr:`power`
        Attributes
        ----------
        edges : array_like
            the edges of the wavenumber bins
        power : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object that holds the projected power.
            It stores the following variables:
            - k :
                the mean value for each ``k`` bin
            - power :
                complex array holding the real and imaginary components of the
                projected power
            - modes :
                the number of Fourier modes averaged together in each bin
        """
        c_d = self.first.compute(Nmesh=self.attrs['Nmesh'], mode='complex')
        r_d = c_d.preview(self.attrs['Nmesh'], axes=self.attrs['axes'])
        # average along projected axes;
        # part of product is the rfftn vs r2c (for axes)
        # the rest is for the mean (Nmesh - axes)
        c_d = numpy.fft.rfftn(r_d) / self.attrs['Nmesh'].prod()

        c_s = self.second.compute(Nmesh=self.attrs['Nmesh'], mode='complex')
        r_s = c_s.preview(self.attrs['Nmesh'], axes=self.attrs['axes'])
        c_s = numpy.fft.rfftn(r_s) / self.attrs['Nmesh'].prod() # average along projected axes

        delta = c_d - c_s

        c1 = delta
        c2 = delta

        pk = c1 * c2.conj()
        # clear the zero mode
        pk.flat[0] = 0

        shape = numpy.array([self.attrs['Nmesh'][i] for i in self.attrs['axes']], dtype='int')
        boxsize = numpy.array([self.attrs['BoxSize'][i] for i in self.attrs['axes']])
        I = numpy.eye(len(shape), dtype='int') * -2 + 1

        k = [numpy.fft.fftfreq(N, 1. / (N * 2 * numpy.pi / L))[:pkshape].reshape(kshape) for N, L, kshape, pkshape in zip(shape, boxsize, I, pk.shape)]

        kmag = sum(ki ** 2 for ki in k) ** 0.5
        W = numpy.empty(pk.shape, dtype='f4')
        W[...] = 2.0
        W[..., 0] = 1.0
        W[..., -1] = 1.0

        dk = self.attrs['dk']
        kmin = self.attrs['kmin']
        axes = list(self.attrs['axes'])
        kedges = numpy.arange(kmin, numpy.pi * self.attrs['Nmesh'][axes].min() / self.attrs['BoxSize'][axes].max() + dk/2, dk)

        xsum = numpy.zeros(len(kedges) + 1)
        Psum = numpy.zeros(len(kedges) + 1, dtype='complex128')
        Nsum = numpy.zeros(len(kedges) + 1)

        dig = numpy.digitize(kmag.flat, kedges)
        xsum.flat += numpy.bincount(dig, weights=(W * kmag).flat, minlength=xsum.size)
        Psum.real.flat += numpy.bincount(dig, weights=(W * pk.real).flat, minlength=xsum.size)
        Psum.imag.flat += numpy.bincount(dig, weights=(W * pk.imag).flat, minlength=xsum.size)
        Nsum.flat += numpy.bincount(dig, weights=W.flat, minlength=xsum.size)

        self.power = numpy.empty(len(kedges) - 1,
                dtype=[('k', 'f8'), ('power', 'c16'), ('modes', 'f8')])

        with numpy.errstate(invalid='ignore', divide='ignore'):
            self.power['k'] = (xsum / Nsum)[1:-1]
            self.power['power'] = (Psum / Nsum)[1:-1] * boxsize.prod() # dimension is 'volume'
            self.power['modes'] = Nsum[1:-1]

        self.edges = kedges

        self.power = BinnedStatistic(['k'], [self.edges], self.power)

