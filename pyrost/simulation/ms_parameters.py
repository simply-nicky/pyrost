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
import os
import numpy as np
from ..ini_parser import INIParser, ROOT_PATH
from ..bin import next_fast_len
from .mslice import Material

MS_PARAMETERS = os.path.join(ROOT_PATH, 'config/ms_parameters.ini')

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
                 'material1':    ('formula', 'density'),
                 'material2':    ('formula', 'density'),
                 'mll':         ('focus', 'n_max', 'n_min', 'mll_sigma', 'mll_depth', 'mll_wl')}

    fmt_dict = {'multislice': 'float', 'material1/formula': 'str', 'material1/density': 'float',
                'material2/formula': 'str', 'material2/density': 'float', 'mll/n_min': 'int',
                'mll/n_max': 'int', 'mll': 'float'}

    def __init__(self, multislice=None, mll_mat1=None, mll_mat2=None, mll=None):
        if multislice is None:
            multislice = self._import_ini(MS_PARAMETERS)['multislice']
        if mll_mat1 is None:
            mll_mat1 = Material(**self._import_ini(MS_PARAMETERS)['material1'])
        if mll_mat2 is None:
            mll_mat2 = Material(**self._import_ini(MS_PARAMETERS)['material2'])
        if mll is None:
            mll = self._import_ini(MS_PARAMETERS)['mll']
        super(MSParams, self).__init__(multislice=multislice, mll=mll,
                                       material1=mll_mat1.export_dict(),
                                       material2=mll_mat2.export_dict())
        self.mll_mat1, self.mll_mat2 = mll_mat1, mll_mat2

    @classmethod
    def _lookup_dict(cls):
        lookup = {}
        for section in ('multislice', 'mll'):
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    @classmethod
    def import_default(cls, mll_mat1=None, mll_mat2=None, **kwargs):
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
    def import_ini(cls, ini_file, mll_mat1=None, mll_mat2=None, **kwargs):
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

    def get_mat1_r(self, energy):
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

    def get_mat2_r(self, energy):
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

    def get_wavefront_size(self):
        """Return slice array size.

        Returns
        -------
        n_x : int
            Slice size.
        """
        return next_fast_len(int((self.x_max - self.x_min) // self.x_step))

    def get_xcoords(self, size=None):
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

    def get_zcoords(self):
        """Return a coordinate array along the propagation axis.

        Returns
        -------
        z_arr : numpy.ndarray
            Coordinate array [um].
        """
        return self.z_step * np.arange(self.mll_depth // self.z_step)

    def get_kernel(self, fx_arr):
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
