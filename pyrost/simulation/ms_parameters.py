"""
Examples
--------

:func:`pyrost.ms_parameters` generates the multislice experimental
parameters, which could be later parsed to
:class:`pyrost.simulation.MSPropagator` in order to perform the simulation.

>>> import pyrost.simulation as sim
>>> ms_params = sim.MSParams()
>>> print(ms_params)
{'multislice': {'x_max': 30.0, 'x_min': 0.0, '...': '...'},
'mll_mat1': {'formula': 'W', 'density': 21.0}, 'mll_mat2': {'formula': 'SiC', 'density': 3.21},
'mll': {'focus': 1500.0, 'n_max': 8000, 'n_min': 100, '...': '...'}}
"""
from __future__ import annotations
import os
import re
from typing import Dict, Optional, Union
import numpy as np
from ..ini_parser import INIParser, ROOT_PATH
from ..bin import next_fast_len

MS_PARAMETERS = os.path.join(ROOT_PATH, 'config/ms_parameters.ini')

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

    def __init__(self, name: str, mass: float, density: float) -> None:
        self.name, self.mass, self.density = name, mass, density

    def __repr__(self) -> str:
        out = {attr: self.__getattribute__(attr) for attr in ['name', 'density', 'mass']}
        return out.__repr__()

    def __str__(self) -> str:
        out = {attr: self.__getattribute__(attr) for attr in ['name', 'density', 'mass']}
        return out.__str__()

    def export_dict(self) -> Dict[str, Union[str, float]]:
        """Export object to a dictionary.

        Returns
        -------
        dct : dict
            Dictionary with element's name, mass, and density.
        """
        return {'formula': self.name, 'mass': self.mass, 'density': self.density}

    def get_ref_index(self, energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

    def get_absorption_coefficient(self, energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

    def __init__(self, elem: Union[str, int], dbase: str='Chantler') -> None:
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

    def get_asf(self, scat_vec: float) -> float:
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

    def get_sf(self, energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
    formula : str
        Chemical formula of the given compound.
    density : float
        Atomic density of the compound.
    dbase : {'Henke', 'Chantler', 'BrCo'}, optional
        Database of the tabulated scattering factors of each element. The
        following keywords are allowed:

        * 'Henke' : (10 eV < E < 30 keV) [Henke]_.
        * 'Chantler' : (11 eV < E < 405 keV) [Chantler]_.
        * 'BrCo' : (30 eV < E < 509 keV) [BrCo]_.

    Attributes
    ----------
    name : str
        Name of the compound.
    elements : sequence of :class:`Element`
        List of elements in the chemical formula.
    quantities : sequence of float
        Coefficients in the chemical formula.
    mass : float
        Molar mass [u].
    density : float
        Atomic density of the element.

    See Also
    --------
    Element - see for full description of the databases.
    """
    FORMULA_MATCHER = '(' + '|'.join(ELEMENTS[1:]) + ')'
    NUM_MATCHER = r'\d+'

    def __init__(self, formula: str, density: float, dbase: str='Chantler') -> None:
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

    def get_sf(self, energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

class MSParams(INIParser):
    """Container with the experimental parameters of
    one-dimensional multislice beam propagation simulation.
    All the experimental parameters are enlisted in :mod:`ms_parameters`.

    Parameters
    ----------
    multislice : dict, optional
        Multislice parameters.
    mll_mat1 : Material, optional
        The first MLL material.
    mll_mat2 : Material, optional
        The second MLL material.
    mll : dict, optional
        Multilayer Laue lens parameters.

    Attributes
    ----------
    x_min, x_max : float
        Wavefront span along the x axis [um].
    x_step : float
        Beam sample interval along the x axis [um].
    z_step : float
        Distance between the slices [um].
    wl : float
        Beam's wavelength [um].
    focus : float
        MLL's focal distance [um].
    n_min, n_max : int
        Zone number of the first and the last layer in
        the MLL.
    mll_sigma : float
        MLL's bilayer interdiffusion length [um].
    mll_depth : float
        MLL's thickness [um].
    mll_wl : float
        Wavelength oh the MLL [um].
    mll_mat1 : Material
        The first MLL material.
    mll_mat2 : Material
        The second MLL material.

    See Also
    --------
    ms_parameters : Full list of experimental parameters.
    """
    attr_dict = {'multislice':  ('x_max', 'x_min', 'x_step', 'z_step', 'wl'),
                 'material1':   ('formula', 'density'),
                 'material2':   ('formula', 'density'),
                 'mll':         ('focus', 'n_max', 'n_min', 'mll_sigma', 'mll_depth', 'mll_wl')}

    fmt_dict = {'multislice': 'float', 'material1/formula': 'str', 'material1/density': 'float',
                'material2/formula': 'str', 'material2/density': 'float', 'mll/n_min': 'int',
                'mll/n_max': 'int', 'mll': 'float'}

    def __init__(self, multislice: Dict[str, float], mll_mat1: Dict[str, Union[str, float]],
                 mll_mat2: Dict[str, Union[str, float]], mll: Dict[str, Union[int, float]]) -> None:
        super(MSParams, self).__init__(multislice=multislice, mll=mll,
                                       material1=mll_mat1.export_dict(),
                                       material2=mll_mat2.export_dict())
        self.mll_mat1, self.mll_mat2 = mll_mat1, mll_mat2

    @classmethod
    def _lookup_dict(cls) -> Dict[str, str]:
        lookup = {}
        for section in ('multislice', 'mll'):
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    @classmethod
    def import_default(cls, mll_mat1: Optional[Material]=None, mll_mat2: Optional[Material]=None,
                       **kwargs: Union[int, float]) -> MSParams:
        """Return the default :class:`MSParams`. Extra arguments
        override the default values if provided.

        Parameters
        ----------
        mll_mat1 : Material, optional
            The first MLL material.
        mll_mat2 : Material, optional
            The second MLL material.
        **kwargs : dict, optional
            Experimental parameters enlisted in :mod:`ms_parameters`.

        Returns
        -------
        MSParams
            A :class:`MSParams` object with the default parameters.

        See Also
        --------
        st_parameters : Full list of the experimental parameters.
        """
        return cls.import_ini(MS_PARAMETERS, mll_mat1, mll_mat2, **kwargs)

    @classmethod
    def import_ini(cls, ini_file: str, mll_mat1: Optional[Material]=None,
                   mll_mat2: Optional[Material]=None, **kwargs: Union[int, float]) -> MSParams:
        """Initialize a :class:`MSParams` object class with an
        ini file.

        Parameters
        ----------
        ini_file : str
            Path to the ini file. Load the default parameters if None.
        mll_mat1 : Material, optional
            The first MLL material. Initialized with `ini_file`
            if None.
        mll_mat2 : Material, optional
            The second MLL material. Initialized with `ini_file`
            if None.
        **kwargs : dict, optional
            Experimental parameters enlisted in :mod:`ms_parameters`.
            Initialized with `ini_file` if not provided.

        Returns
        -------
        ms_params : MSParams
            A :class:`MSParams` object with all the attributes imported
            from the ini file.

        See Also
        --------
        ms_parameters : Full list of the experimental parameters.
        """
        attr_dict = cls._import_ini(ini_file)
        if mll_mat1 is None:
            mll_mat1 = Material(**attr_dict['material1'])
        if mll_mat2 is None:
            mll_mat2 = Material(**attr_dict['material2'])
        for option, section in cls._lookup_dict().items():
            if option in kwargs:
                attr_dict[section][option] = kwargs[option]
        return cls(multislice=attr_dict['multislice'], mll_mat1=mll_mat1,
                   mll_mat2=mll_mat2, mll=attr_dict['mll'])

    def get_mat1_r(self, energy: float) -> complex:
        """Return the Fresnel transmission coefficient of the first
        material in MLL bilayer.

        Parameters
        ----------
        energy : float
            Beam's photon energy [keV].

        Returns
        -------
        complex
            Fresnel transmission coefficient.
        """
        ref_idx = self.mll_mat1.get_ref_index(energy)
        return 2 * np.pi / self.wl * self.z_step * ref_idx

    def get_mat2_r(self, energy: float) -> complex:
        """Return the Fresnel transmission coefficient of the second
        material in MLL bilayer.

        Parameters
        ----------
        energy : float
            Beam's photon energy [keV].

        Returns
        -------
        complex
            Fresnel transmission coefficient.
        """
        ref_idx = self.mll_mat2.get_ref_index(energy)
        return 2 * np.pi / self.wl * self.z_step * ref_idx

    def get_wavefront_size(self) -> int:
        """Return slice array size.

        Returns
        -------
        n_x : int
            Slice size.
        """
        return next_fast_len(int((self.x_max - self.x_min) // self.x_step))

    def get_xcoords(self, size: Optional[int]=None) -> np.ndarray:
        """Return a coordinate array of a slice.

        Parameters
        ----------
        size : int, optional
            Size of the array. Equals to
            :func:`MSParams.get_wavefront_size` if
            it's None.

        Returns
        -------
        x_arr : numpy.ndarray
            Coordinate array [um].
        """
        if size is None:
            size = self.get_wavefront_size()
        return (self.x_min + self.x_max) / 2 + self.x_step * np.arange(-size // 2, size // 2)

    def get_zcoords(self) -> np.ndarray:
        """Return a coordinate array along the propagation axis.

        Returns
        -------
        z_arr : numpy.ndarray
            Coordinate array [um].
        """
        return self.z_step * np.arange(self.mll_depth // self.z_step)

    def get_kernel(self, fx_arr: np.ndarray) -> np.ndarray:
        r"""Return diffraction kernel for the multislice propagation.

        Parameters
        ----------
        fx_arr : numpy.ndarray
            Spatial frequencies.

        Returns
        -------
        numpy.ndarray
            Diffraction kernel.

        Notes
        -----
        The diffraction kernel is given by:

        .. math::
            k(f_x, f_y) = \exp{-j \frac{2 \pi \Delta z}{\lambda}
            \sqrt{1 - \lambda^2 (f_x^2 + f_y^2)}}

        where $\Delta z$ --- slice thickness, $f_x, f_y$ ---
        spatial frequencies, and $\lambda$ --- wavelength.
        """
        kernel = np.exp(-2j * np.pi / self.wl * self.z_step * \
                        np.sqrt(1 - self.wl**2 * fx_arr.astype(complex)**2))
        return kernel
