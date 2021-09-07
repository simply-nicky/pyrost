"""Multislice beam propagation simulation. Generate a profile of
the beam that propagates through a bulky sample.
:class:`pyrost.simulation.MLL` generates a transmission profile of
a wedged Multilayer Laue lens (MLL). :class:`pyrost.simulations.MSPropagator`
calculates the beam propagation and spits out the wavefield profile of
the propagated beam together with the transmission profile of the sample
at each slice.

Examples
--------

Initialize a MLL object with the :class:`pyrost.simulations.MSParams`
parameters object `params`:

>>> import pyrost.simulation as sim
>>> mll = sim.MSParams(params)

Initialize a multislice beam propagator with `params` and `mll`:

>>> ms_prgt = sim.MSPropagator(params, mll)

Perform the multislice beam propagation as follows:

>>> ms_prgt.beam_propagate()

All the results are saved into `ms_prgt.beam_profile` and
`ms_prgt.smp_profile` attributes.
"""
import os
from multiprocessing import cpu_count
import re
from tqdm.auto import tqdm
import numpy as np
from ..data_container import DataContainer, dict_to_object
from ..bin import mll_profile, FFTW, empty_aligned, rsc_wp
from ..ini_parser import ROOT_PATH

ELEMENTS = ('None', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
            'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
            'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
            'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U')

class BasicElement:
    """Bacis chemical element class.

    Parameters
    ----------
    name : str
        Name of the element.
    mass : float
        Atomic mass of the element.
    density : float
        Atomic density of the element.

    Attributes
    ----------
    name : str
        Name of the element.
    mass : float
        Atomic mass of the element.
    density : float
        Atomic density of the element.
    """
    en_to_wl = 12398.419297617678 # h * c / e [eV * A]
    ref_cf = 2.7008646837561236e-6 # Avogadro * r_el / 2 / pi [A]

    def __init__(self, name, mass, density):
        self.name, self.mass, self.density = name, mass, density

    def __repr__(self):
        out = {attr: self.__getattribute__(attr) for attr in ['name', 'density', 'mass']}
        return out.__repr__()

    def __str__(self):
        out = {attr: self.__getattribute__(attr) for attr in ['name', 'density', 'mass']}
        return out.__str__()

    def export_dict(self):
        return {'formula': self.name, 'density': self.density}

    def get_ref_index(self, energy):
        r"""Calculates the complex refractive index for the given photon
        `energy`.

        Parameters
        ----------
        energy : float or numpy.ndarray
            Photon energy [eV].

        Returns
        -------
        mu : float or numpy.ndarray
            Linear absorption coefficient [cm^-1]

        Notes
        -----
        The complex refractive index is customary denoted as:
        .. math::
            n = 1 - \delta + i \betta
        The real and imaginary components, :math:`\delta` and :math:`\betta` are
        given by [GISAXS]_:
        .. math::
            \delta = \frac{\rho N_a r_e \lambda f_1}{2 \pi m_a}
        .. math::
            \betta = \frac{\rho N_a r_e \lambda f_2}{2 \pi m_a}
        where $\rho$ is physical density, $N_a$ is Avogadro constant, $m_a$ is
        atomic molar mass, $r_e$ is radius of electron, $lambda$ is wavelength,
        $f_1$ and f_2$ are real and imaginary components of scattering factor.

        Reference
        ---------
        .. [GISAXS] http://gisaxs.com/index.php/Refractive_index
        """
        wavelength = self.en_to_wl / energy
        ref_idx = self.ref_cf * wavelength**2 * self.density * self.get_sf(energy) / self.mass
        return ref_idx

    def get_absorption_coefficient(self, energy):
        r"""Calculates the linear absorption coefficientfor the given photon
        `energy`.

        Parameters
        ----------
        energy : float or numpy.ndarray
            Photon energy [eV].

        Returns
        -------
        mu : float or numpy.ndarray
            Linear absorption coefficient [cm^-1]

        Notes
        -----
        The absorption coefficient is the inverse of the absorption length and
        is given by [GISAXS]_:
        .. math::
            \mu = \frac{\rho N_a}{m_a} 2 r_e \lambda f_2 = \frac{4 \pi \betta}{\lambda}
        where $\rho$ is physical density, $N_a$ is Avogadro constant, $m_a$ is
        atomic molar mass, $r_e$ is radius of electron, $lambda$ is wavelength,
        and $f_2$ is imaginary part of scattering factor.

        Reference
        ---------
        .. [GISAXS] http://gisaxs.com/index.php/Absorption_length
        """
        wavelength = self.en_to_wl / energy
        return 4 * np.pi * self.get_ref_index(energy).imag / wavelength

class Element(BasicElement):
    """This class serves for accessing the scattering factors f1 and f2
    and atomic scattering factor of a chemical element `elem`.

    Parameters
    ----------
    elem : str or int
        The element can be specified by its name (case sensitive) or its
        atomic number.
    dbase : {'Henke', 'Chantler', 'BrCo'}, optional
        Database of the tabulated scattering factors of each element. The
        following keywords are allowed:

        * 'Henke' : (10 eV < E < 30 keV) [Henke]_.
        * 'Chantler' : (11 eV < E < 405 keV) [Chantler]_.
        * 'BrCo' : (30 eV < E < 509 keV) [BrCo]_.

    Attributes
    ----------
    name : str
        Name of the chemical element `elem`.
    atom_num : int
        Atomic number of `elem`.
    dbase : str
        Database of scattering factors.
    asf_coeffs : numpy.ndarray
        Coefficients of atomic scattering factor.
    sf_coeffs : numpy.ndarray
        Scattering factors (`energy`, `f1`, `f2`).
    mass : float
        Atomic mass [u].
    radius : float
        Atomic radius [Angstrom].
    density : float
        Density [g / cm^3].

    References
    ----------
    .. [Henke] http://henke.lbl.gov/optical_constants/asf.html
               B.L. Henke, E.M. Gullikson, and J.C. Davis, *X-ray interactions:
               photoabsorption, scattering, transmission, and reflection at
               E=50-30000 eV, Z=1-92*, Atomic Data and Nuclear Data Tables
               **54** (no.2) (1993) 181-342.
    .. [Chantler] http://physics.nist.gov/PhysRefData/FFast/Text/cover.html
                  http://physics.nist.gov/PhysRefData/FFast/html/form.html
                  C. T. Chantler, *Theoretical Form Factor, Attenuation, and
                  Scattering Tabulation for Z = 1 - 92 from E = 1 - 10 eV to E = 0.4 -
                  1.0 MeV*, J. Phys. Chem. Ref. Data **24** (1995) 71-643.
    .. [BrCo] http://www.bmsc.washington.edu/scatter/periodic-table.html
              ftp://ftpa.aps.anl.gov/pub/cross-section_codes/
              S. Brennan and P.L. Cowan, *A suite of programs for calculating
              x-ray absorption, reflection and diffraction performance for a
              variety of materials at arbitrary wavelengths*, Rev. Sci. Instrum.
              **63** (1992) 850-853.
    """
    dbase_lookup = {'Henke': 'data/henke_f1f2.npz', 'Chantler': 'data/chantler_f1f2.npz',
                    'BrCo': 'data/brco_f1f2.npz'}
    asf_dbase = 'data/henke_f0.npz'
    atom_dbase = 'data/atom_data.npz'

    def __init__(self, elem, dbase='Chantler'):
        if isinstance(elem, str):
            name = elem
            self.atom_num = ELEMENTS.index(elem)
        elif isinstance(elem, int):
            name = ELEMENTS[elem]
            self.atom_num = elem
        else:
            raise ValueError('Wrong element: {:s}'.format(str(elem)))
        if dbase in self.dbase_lookup:
            self.dbase = dbase
        else:
            raise ValueError('Wrong database: {:s}'.format(dbase))

        with np.load(os.path.join(ROOT_PATH, self.atom_dbase)) as dbase:
            mass = dbase['mass'][self.atom_num - 1]
            density = dbase['density'][self.atom_num - 1]
            self.radius = dbase['radius'][self.atom_num - 1]

        super(Element, self).__init__(name, mass, density)

        with np.load(os.path.join(ROOT_PATH, self.asf_dbase)) as dbase:
            self.asf_coeffs = dbase[self.name]
        with np.load(os.path.join(ROOT_PATH, self.dbase_lookup[self.dbase])) as dbase:
            self.sf_coeffs = np.stack((dbase[self.name + '_E'], dbase[self.name + '_f1'],
                                       dbase[self.name + '_f2']))

    def get_asf(self, scat_vec):
        """Calculate atomic scattering factor for the given magnitude of
        scattering vector `scat_vec`.

        Parameters
        ----------
        scat_vec : float
            Scattering vector magnitude [Angstrom^-1].

        Returns
        -------
        asf : float
            Atomic scattering factor.
        """
        q_ofpi = scat_vec / 4 / np.pi
        asf = self.asf_coeffs[5] + sum(a * np.exp(-b * q_ofpi**2)
                                       for a, b in zip(self.asf_coeffs[:5], self.asf_coeffs[6:]))
        return asf

    def get_sf(self, energy):
        """Return a complex scattering factor (`f1` + 1j * `f2`) for the given photon
        `energy`.

        Parameters
        ----------
        energy : float or numpy.ndarray
            Photon energy [eV].

        Returns
        -------
        f1 + 1j * f2 : numpy.ndarray
            Complex scattering factor.
        """
        if np.any(energy < self.sf_coeffs[0, 0]) or np.any(energy > self.sf_coeffs[0, -1]):
            exc_txt = 'Energy is out of bounds: ({0:.2f} - {1:.2f})'.format(self.sf_coeffs[0, 0],
                                                                            self.sf_coeffs[0, -1])
            raise ValueError(exc_txt)
        f_one = np.interp(energy, self.sf_coeffs[0], self.sf_coeffs[1]) + self.atom_num
        f_two = np.interp(energy, self.sf_coeffs[0], self.sf_coeffs[2])
        return f_one + 1j * f_two

class Material(BasicElement):
    """
    :class:`Material` serves for getting refractive index and absorption
    coefficient of a material specified by its chemical formula and density.

    Parameters
    ----------
    elements : str or sequence of str
        List of all the constituent elements (symbols).
    quantities: None or sequence of float, optional
        Coefficients in the chemical formula. If None, the coefficients
        are all equal to 1.
    dbase : {'Henke', 'Chantler', 'BrCo'}, optional
        Database of the tabulated scattering factors of each element. The
        following keywords are allowed:

        * 'Henke' : (10 eV < E < 30 keV) [Henke]_.
        * 'Chantler' : (11 eV < E < 405 keV) [Chantler]_.
        * 'BrCo' : (30 eV < E < 509 keV) [BrCo]_.

    Attributes
    ----------
    elements : sequence of :class:`Element`
        List of elements.
    quantities : sequence of float
        Coefficients in the chemical formula.
    mass : float
        Molar mass [u].

    See Also
    --------
    Element - see for full description of the databases.
    """
    FORMULA_MATCHER = '(' + '|'.join(ELEMENTS[1:]) + ')'
    NUM_MATCHER = r'\d+'

    def __init__(self, formula, density, dbase='Chantler'):
        self.elements = []
        self.quantities = []
        mass = 0

        for m in re.finditer(self.FORMULA_MATCHER, formula):
            new_elem = Element(elem=m.group(0), dbase=dbase)
            m_num = re.match(self.NUM_MATCHER, formula[m.end():])
            quant = int(m_num.group(0)) if m_num else 1
            mass += new_elem.mass * quant
            self.elements.append(new_elem)
            self.quantities.append(quant)

        super(Material, self).__init__(formula, mass, density)

    def get_sf(self, energy):
        """Return a complex scattering factor (`f1` + 1j * `f2`) for the given photon
        `energy`.

        Parameters
        ----------
        energy : float or numpy.ndarray
            Photon energy [eV].

        Returns
        -------
        f1 + 1j * f2 : numpy.ndarray
            Complex scattering factor.
        """
        return np.sum([elem.get_sf(energy) * quant
                       for elem, quant in zip(self.elements, self.quantities)], axis=0)

class MLL(DataContainer):
    """
    Multilayer Laue lens class.

    Parameters
    ----------
    layers : numpy.ndarray
        MLL's bilayers x coordinates [um].
    mat1_r, mat2_r : complex
        Fresnel transmission coefficients for each of the materials,
        that the MLL's bilayers are composed of. The coefficients are the
        ratio of a wavefield propagating through a single slice of the
        lens.
    sigma : float
        Bilayer's interdiffusion length [um].

    Attributes
    ----------
    attr_set : set
        Set of the attributes in the container which are necessary
        to initialize in the constructor.
    """
    attr_set = {'layers', 'mat1_r', 'mat2_r', 'sigma'}
    en_to_wl = 1.239841929761768 # h * c / e [eV * um]

    def __init__(self, mat1_r, mat2_r, sigma, layers):
        super(MLL, self).__init__(mat1_r=mat1_r, mat2_r=mat2_r, sigma=sigma,
                                  layers=layers)

    @classmethod
    def import_params(cls, params):
        """Return a new :class:`MLL` object import from the
        :class:`MSParams` multislice parameters object.

        Parameters
        ----------
        """
        n_arr = np.arange(params.n_min, params.n_max)
        z_coords = params.z_step * np.arange(params.mll_depth // params.z_step)
        layers = np.sqrt(n_arr * params.focus * params.mll_wl + \
                              n_arr**2 * params.mll_wl**2 / 4)
        layers = layers * (1 - z_coords[:, None] / (2 * params.focus))
        mat1_r = params.get_mat1_r(cls.en_to_wl / params.wl)
        mat2_r = params.get_mat2_r(cls.en_to_wl / params.wl)
        sigma = params.mll_sigma
        return cls(mat1_r=mat1_r, mat2_r=mat2_r, sigma=sigma,
                   layers=layers)

    @property
    def n_slices(self):
        "Total number of slices"
        return self.layers.shape[0]

    @dict_to_object
    def update_interdiffusion(self, sigma):
        """Return a new :class:`MLL` object with the updated `sigma`.

        Parameters
        ----------
        sigma : float
            Bilayer's interdiffusion length [um].

        Returns
        -------
        MLL
            New :class:`MLL` object with the updated `sigma`.
        """
        return {'sigma': sigma}

    @dict_to_object
    def update_materials(self, mat1_r, mat2_r):
        """Return a new :class:`MLL` object with the updated materials
        `mat1_r` and `mat2_r`.

        Parameters
        ----------
        mat1_r, mat2_r : complex
            Fresnel transmission coefficients for each of the materials,
            that the MLL's bilayers are composed of. The coefficients are the
            ratio of a wavefield propagating through a single slice of the
            lens.

        Returns
        -------
        MLL
            New :class:`MLL` object with the updated `mat1_r` and `mat2_r`.
        """
        return {'mat1_r': mat1_r, 'mat2_r': mat2_r}

    def get_span(self):
        return (self.layers.min(), self.layers.max())

    def get_profile(self, x_arr, output, num_threads=1):
        """Return a generator, that yields transmission profiles of
        the lens at each slice and writes to `output` array.

        Parameters
        ----------
        x_arr : numpy.ndarray
            Coordinates array [um].
        output: numpy.ndarray
            Output array.
        num_threads : int, optional
            Number of threads.

        Returns
        -------
        slices : iterable
            The generator, which yields lens' transmission profiles.
        """
        for idx, layer in enumerate(self.layers):
            output[idx, :] = mll_profile(x_arr=x_arr, layers=layer, mt0=self.mat1_r,
                                         mt1=self.mat2_r, sigma=self.sigma,
                                         num_threads=num_threads)
            yield output[idx]

class MSPropagator(DataContainer):
    """One-dimensional Multislice beam propagation class.
    Generates beam profile, that propagates through the sample
    using multislice approach.

    Parameters
    ----------
    params : MSParams
        Experimental parameters
    sample : MLL
        Sample class, that generates the sample's
        transmission profile.
    num_threads : int, optional
        Number of threads used in the calculations.
    **kwargs : dict
        Attributes specified in `init_set`.

    Attributes
    ----------
    attr_set : set
        Set of attributes in the container which are necessary
        to initialize in the constructor.
    init_set : set
        Set of optional data attributes.

    Notes
    -----
    Necessary attributes:

    * sample : Sample's transmission profile generator.
    * num_threads : Number of threads used in the calculations.
    * params : Experimental parameters.

    Optional attributes:

    * beam_profile : Beam profiles at each slice.
    * fx_arr : Spatial frequencies array [um^-1].
    * kernel : Diffraction kernel.
    * smp_profile : Sample's transmission profile at each slice.
    * x_arr : Coordinates array in the transverse plane [um].
    * wf_inc : Wavefront at the entry surface.
    * z_arr : Coordinates array along the propagation axis [um].
    """
    attr_set = {'num_threads', 'params', 'sample'}
    init_set = {'beam_profile', 'fx_arr', 'kernel', 'smp_profile', 'x_arr', 'wf_inc', 'z_arr'}

    def __init__(self, params, sample, num_threads=None, **kwargs):
        if num_threads is None:
            num_threads = np.clip(1, 64, cpu_count())
        super(MSPropagator, self).__init__(params=params, sample=sample,
                                           num_threads=num_threads, **kwargs)
        self._init_dict()

    def _init_dict(self):
        if self.x_arr is None or self.fx_arr is None:
            self.x_arr = self.params.get_xcoords()
            self.z_arr = self.params.get_zcoords()
            self.fx_arr = np.fft.fftfreq(self.size, self.params.x_step)

        if self.kernel is None:
            self.kernel = self.params.get_kernel(self.fx_arr) / self.fx_arr.size

        if self.wf_inc is None:
            self.wf_inc = np.ones(self.x_arr.shape, dtype=np.complex128)
            x_min, x_max = self.sample.get_span()
            self.wf_inc[(self.x_arr < x_min) | (self.x_arr > x_max)] = 0.

    @property
    def size(self):
        "Number of points in a single slice."
        return self.x_arr.size

    @dict_to_object
    def update_inc_wavefront(self, wf_inc):
        """Return a new :class:`MSPropagator` object with the updated `wf_inc`.

        Parameters
        ----------
        wf_inc : numpy.ndarray
            Wavefront at the entry surface.

        Returns
        -------
        MSPropagator
            A new :class:`MSPropagator` object with the updated `wf_inc`.
        """
        if wf_inc.shape != self.x_arr.shape:
            raise ValueError(f'Wavefront shape must be equal to {self.x_arr.shape:s}')
        if wf_inc.dtype != np.complex128:
            raise ValueError("Wavefront datatype must be 'complex128'")
        return {'beam_profile': None, 'wf_inc': wf_inc}

    def generate_sample(self, verbose=True):
        """Generate the transmission profile of the sample. The results are
        written to `smp_profile` attribute.

        Parameters
        ----------
        verbose : bool, optional
            Set verbosity of the computation process.
        """
        self.smp_profile = empty_aligned((self.sample.n_slices, self.x_arr.size),
                                         dtype='complex128')
        itor = self.sample.get_profile(x_arr=self.x_arr, output=self.smp_profile,
                                       num_threads=self.num_threads)
        
        if verbose:
            itor = tqdm(enumerate(itor), total=self.sample.n_slices,
                        bar_format='{desc} {percentage:3.0f}% {bar} Slice {n_fmt} / {total_fmt} '\
                                '[{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        for idx, _ in itor:
            if verbose:
                itor.set_description(f"z = {self.z_arr[idx]:.2f} um")

    def beam_propagate(self, verbose=True):
        """Perform the multislice beam propagation. The results are
        written to `beam_profile` attribute. Generates `smp_profile`
        attribute if it wasn't initialized before.

        Parameters
        ----------
        verbose : bool, optional
            Set verbosity of the computation process.
        """
        self.beam_profile = empty_aligned((self.sample.n_slices + 1, self.x_arr.size),
                                          dtype='complex128')
        current_slice = empty_aligned(self.x_arr.size, dtype='complex128')
        self.beam_profile[0] = self.wf_inc
        current_slice[:] = self.wf_inc

        fft_obj = FFTW(current_slice, current_slice, flags=('FFTW_ESTIMATE',),
                       threads=self.num_threads)
        ifft_obj = FFTW(current_slice, self.beam_profile[1], direction='FFTW_BACKWARD',
                        flags=('FFTW_ESTIMATE',), threads=self.num_threads)

        if self.smp_profile is None:
            self.smp_profile = empty_aligned((self.sample.n_slices, self.x_arr.size),
                                             dtype='complex128')
            itor = self.sample.get_profile(x_arr=self.x_arr, output=self.smp_profile,
                                           num_threads=self.num_threads)
        else:
            itor = (layer for layer in self.smp_profile)

        if verbose:
            itor = tqdm(itor, total=self.sample.n_slices,
                        bar_format='{desc} {percentage:3.0f}% {bar} Slice {n_fmt} / {total_fmt} '\
                                   '[{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        for idx, layer in enumerate(itor):
            if verbose:
                itor.set_description(f"z = {self.z_arr[idx]:.2f} um")
            current_slice *= layer
            fft_obj.execute()
            current_slice *= self.kernel
            ifft_obj.update_arrays(current_slice, self.beam_profile[idx + 1])
            ifft_obj.execute()
            current_slice[:] = ifft_obj.output_array

    def beam_downstream(self, z_arr, step=None, return_coords=True, verbose=True, backend='fftw'):
        """Return the wavefront at distance `z_arr` downstream from
        the exit surface using Rayleigh-Sommerfeld convolution.

        Parameters
        ----------
        z_arr : numpy.ndarray
            Array of distances [um].
        step : float, optional
            Sampling interval of the downstream coordinate array [um].
            Equals to the sampling interval at the exit surface if it's
            None.
        return_coords : bool, optional
            Return the coordinates array of the downstream plane.
        verbose : bool, optional
            Set verbosity of the computation process.

        Returns
        -------
        wavefronts : numpy.ndarray
            Set of wavefronts calculated at the distance `z_arr`
            downstream from the exit surface.
        x_arr : numpy.ndarray
            Array of coordinates at the plane downstream [um].
            Only if `return_coords` is True.

        Raises
        ------
        AttributeError
            If `beam_profile` has not been generated. Call
            :func:`MSPropagator.beam_propagate` to generate.
        """
        if self.beam_profile is None:
            raise AttributeError('The beam profile has not been generated')
        if step is None:
            step = self.params.x_step
        size = int(self.x_arr.size**2 * max(self.params.x_step, step) * \
                   (step + self.params.x_step) / self.params.wl / np.min(z_arr))
        wf0 = self.beam_profile[-1]
        if size > self.x_arr.size:
            wf0 = np.pad(wf0, ((size - self.x_arr.size) // 2, (size - self.x_arr.size) // 2))
        else:
            size = self.x_arr.size

        itor = np.atleast_1d(z_arr)
        if verbose:
            itor = tqdm(itor, total=itor.size)

        wavefronts = []
        for dist in itor:
            if verbose:
                itor.set_description(f"z = {dist:.2f} um")
            wavefronts.append(rsc_wp(wft=wf0, dx0=self.params.x_step, dx=step,
                                     z=dist, wl=self.params.wl, backend=backend,
                                     num_threads=self.num_threads))
        if return_coords:
            x_arr = step * np.arange(-size // 2, size - size // 2) + np.mean(self.x_arr)
            return np.squeeze(np.stack(wavefronts, axis=1)), x_arr
        return np.squeeze(np.stack(wavefronts, axis=1))
