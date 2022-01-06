Working with CXI files
======================

:class:`CXIProtocol <pyrost.CXIProtocol>`
-----------------------------------------

CXI protocol (:class:`pyrost.CXIProtocol`) is a helper class for a :class:`pyrost.STData`
data container, which tells it where to look for the necessary data fields in a CXI
file. The class is fully customizable so you can tailor it to your particular data
structure of CXI file. The protocol consists of the following attributes for each
data field (`data`, `whitefield`, etc.):

* `datatypes` : Data type (`float`, `int`, or `bool`).
* `default_paths` : CXI file path.
* `is_data` : Flag all the attributes whether they are of the data type.

.. note::
    Attribute is of data type if the data array is 2- or 3-dimensional and
    has the shape identical to the detector pixel grid.

.. note::

    You can save protocol to an INI file with :func:`pyrost.CXIProtocol.export_ini`
    and import protocol from INI file with :func:`pyrost.CXIProtocol.import_ini`.

The default protocol can be accessed with :func:`pyrost.CXIProtocol.import_default`. The protocol
is given by:

.. code-block:: ini

    [config]
    float_precision = float64

    [datatypes]
    basis_vectors = float
    data = uint
    defocus_x = float
    defocus_y = float
    distance = float
    energy = float
    error_frame = float
    flatfields = uint
    good_frames = uint
    m0 = int
    mask = bool
    n0 = int
    phase = float
    pixel_aberrations = float
    pixel_map = float
    pixel_translations = float
    reference_image = float
    roi = uint
    translations = float
    wavelength = float
    whitefield = float
    x_pixel_size = float
    y_pixel_size = float

    [default_paths]
    basis_vectors = /speckle_tracking/basis_vectors
    data = /entry/data/data
    defocus_x = /speckle_tracking/defocus_x
    defocus_y = /speckle_tracking/defocus_y
    distance = /entry/instrument/detector/distance
    energy = /entry/instrument/source/energy
    error_frame = /speckle_tracking/error_frame
    flatfields = /speckle_tracking/flatfields
    good_frames = /speckle_tracking/good_frames
    m0 = /speckle_tracking/m0
    mask = /speckle_tracking/mask
    n0 = /speckle_tracking/n0
    phase = /speckle_tracking/phase
    pixel_aberrations = /speckle_tracking/pixel_aberrations
    pixel_map = /speckle_tracking/pixel_map
    pixel_translations = /speckle_tracking/pixel_translations
    reference_image = /speckle_tracking/reference_image
    roi = /speckle_tracking/roi
    translations = /speckle_tracking/translations
    wavelength = /entry/instrument/source/wavelength
    whitefield = /speckle_tracking/whitefield
    x_pixel_size = /entry/instrument/detector/x_pixel_size
    y_pixel_size = /entry/instrument/detector/y_pixel_size

    [is_data]
    basis_vectors = False
    data = True
    defocus_x = False
    defocus_y = False
    distance = False
    energy = False
    error_frame = False
    flatfields = True
    good_frames = False
    m0 = False
    mask = True
    n0 = False
    phase = True
    pixel_aberrations = True
    pixel_map = True
    pixel_translations = False
    reference_image = False
    roi = False
    translations = False
    wavelength = False
    whitefield = True
    x_pixel_size = False
    y_pixel_size = False

:class:`CXILoader <pyrost.CXILoader>`
-------------------------------------

CXI file loader class (:class:`pyrost.CXILoader`) uses a protocol to
automatically load all the necessary data fields from a CXI file. Other than
the information provided by a protocol, a loader class requires the following
attributes:

* `policy` : Loading policy for each attribute enlisted in protocol. If it's
  True, the corresponding attribute will be loaded.
* `load_paths` : List of extra CXI file paths, where the loader will look for
  the data field.

.. note::

    You can save loader to an INI file with :func:`pyrost.CXILoader.export_ini`
    and import loader from an INI file with :func:`pyrost.CXILoader.import_ini`.

The default loader can be accessed with :func:`pyrost.CXILoader.import_default`. The loader
is given by:

.. code-block:: ini

    [load_paths]
    basis_vectors = [/entry_1/instrument_1/detector_1/basis_vectors, /entry/instrument/detector/basis_vectors]
    data = [/entry/instrument/detector/data, /entry_1/instrument_1/detector_1/data, /entry_1/data_1/data]
    defocus_y = [/speckle_tracking/defocus_ss]
    defocus_x = [/speckle_tracking/defocus_fs, /speckle_tracking/defocus]
    distance = [/entry_1/instrument_1/detector_1/distance,]
    energy = [/entry_1/instrument_1/detector_1/distance,]
    good_frames = [/frame_selector/good_frames, /process_3/good_frames]
    mask = [/mask_maker/mask, /entry_1/instrument_1/detector_1/mask, /entry/instrument/detector/mask]
    translations = [/entry_1/sample_1/geometry/translations, /entry/sample/geometry/translations, /entry_1/sample_1/geometry/translation, /pos_refine/translation, /entry_1/sample_3/geometry/translation]
    wavelength = [/entry_1/instrument_1/source_1/wavelength,]
    whitefield = [/process_1/whitefield, /make_whitefield/whitefield, /process_2/whitefield, /process_3/whitefield]
    x_pixel_size = [/entry_1/instrument_1/detector_1/x_pixel_size,]
    y_pixel_size = [/entry_1/instrument_1/detector_1/y_pixel_size,]

    [policy]
    basis_vectors = True
    data = True
    defocus_x = True
    defocus_y = True
    distance = True
    energy = False
    error_frame = False
    flatfields = False
    good_frames = True
    m0 = False
    mask = True
    n0 = False
    phase = False
    pixel_aberrations = False
    pixel_map = False
    pixel_translations = False
    reference_image = False
    roi = True
    translations = True
    wavelength = True
    whitefield = True
    x_pixel_size = True
    y_pixel_size = True

.. automodule:: pyrost.cxi_protocol

Contents
--------

.. toctree::
    :maxdepth: 2

    classes/cxi_protocol
    classes/cxi_loader