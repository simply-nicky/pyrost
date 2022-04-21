Working with CXI files
======================

:class:`CXIProtocol <pyrost.CXIProtocol>`
-----------------------------------------

CXI protocol (:class:`pyrost.CXIProtocol`) is a helper class for a :class:`pyrost.STData`
data container, which tells it where to look for the necessary data fields in a CXI
file. The class is fully customizable so you can tailor it to your particular data
structure of CXI file. The protocol consists of the following parts for each data
attribute (`data`, `whitefield`, etc.):

* **datatypes** : Data type (`float`, `int`, `uint`, or `bool`) of the given attribute.
* **load_paths** : List of paths inside a HDF5 file, where the given data attribute may be
  saved.
* **kinds** : The attribute's kind, that specifies data dimensionality. This information
  is required to know how load, save and process the data. The attribute may be one of
  the four following kinds:

  * *scalar* : Data is either 0D, 1D, or 2D. The data is saved and loaded plainly
    without any transforms or indexing.
  * *sequence* : A time sequence array. Data is either 1D, 2D, or 3D. The data is
    indexed, so the first dimension of the data array must be a time dimension. The
    data points for the given index are not transformed.
  * *frame* : Frame array. Data must be 2D, it may be transformed with any of
    :class:`pyrost.Transform` objects. The data shape is identical to the detector
    pixel grid.
  * *stack* : A time sequnce of frame arrays. The data must be 3D. It's indexed in the
    same way as `sequence` attributes. Each frame array may be transformed with any of
    :class:`pyrost.Transform` objects.

.. note::

    You can save protocol to an INI file with :func:`pyrost.CXIProtocol.export_ini`
    and import protocol from INI file with :func:`pyrost.CXIProtocol.import_ini`.

The default protocol can be accessed with :func:`pyrost.CXIProtocol.import_default`. The protocol
is given by:

.. code-block:: ini

    [datatypes]
    basis_vectors = float
    data = uint
    defocus_x = float
    defocus_y = float
    distance = float
    good_frames = uint
    mask = bool
    phase = float
    pixel_aberrations = float
    pixel_map = float
    pixel_translations = float
    reference_image = float
    scale_map = float
    translations = float
    wavelength = float
    whitefield = float
    whitefields = float
    x_pixel_size = float
    y_pixel_size = float

    [load_paths]
    basis_vectors = [/speckle_tracking/basis_vectors, /entry_1/instrument_1/detector_1/basis_vectors, /entry/instrument/detector/basis_vectors]
    data = [/entry/data/data, /entry/instrument/detector/data, /entry_1/instrument_1/detector_1/data, /entry_1/data_1/data]
    defocus_x = [/speckle_tracking/defocus_x, /speckle_tracking/defocus_fs, /speckle_tracking/defocus]
    defocus_y = [/speckle_tracking/defocus_y, /speckle_tracking/defocus_ss]
    distance = [/entry/instrument/detector/distance, /entry_1/instrument_1/detector_1/distance]
    good_frames = [/speckle_tracking/good_frames, /frame_selector/good_frames, /process_3/good_frames]
    mask = [/speckle_tracking/mask, /mask_maker/mask, /entry_1/instrument_1/detector_1/mask, /entry/instrument/detector/mask]
    phase = /speckle_tracking/phase
    pixel_aberrations = /speckle_tracking/pixel_aberrations
    pixel_map = /speckle_tracking/pixel_map
    pixel_translations = /speckle_tracking/pixel_translations
    reference_image = /speckle_tracking/reference_image
    scale_map = /speckle_tracking/scale_map
    translations = [/speckle_tracking/translations, /entry_1/sample_1/geometry/translations, /entry/sample/geometry/translations, /entry_1/sample_1/geometry/translation, /pos_refine/translation, /entry_1/sample_3/geometry/translation]
    wavelength = [/entry/instrument/source/wavelength, /entry_1/instrument_1/source_1/wavelength]
    whitefield = [/speckle_tracking/whitefield, /process_1/whitefield, /make_whitefield/whitefield, /process_2/whitefield, /process_3/whitefield]
    whitefields = [/speckle_tracking/whitefields, /speckle_tracking/flatfields]
    x_pixel_size = [/entry/instrument/detector/x_pixel_size, /entry_1/instrument_1/detector_1/x_pixel_size]
    y_pixel_size = [/entry/instrument/detector/y_pixel_size, /entry_1/instrument_1/detector_1/y_pixel_size]

    [kinds]
    basis_vectors = sequence
    data = stack
    defocus_x = scalar
    defocus_y = scalar
    distance = scalar
    good_frames = sequence
    mask = stack
    phase = frame
    pixel_aberrations = frame
    pixel_map = frame
    pixel_translations = sequence
    reference_image = scalar
    scale_map = frame
    translations = sequence
    wavelength = scalar
    whitefield = frame
    whitefields = stack
    x_pixel_size = scalar
    y_pixel_size = scalar

:class:`CXIStore <pyrost.CXIStore>`
-----------------------------------

CXI file handler class (:class:`pyrost.CXIStore`) accepts a set of paths to the files together with
a protocol object. :class:`pyrost.CXIStore` searches the files for any data attributes defined by
the protocol. It provides an interface to load the data of the given attribute from the files
(see :func:`pyrost.CXIStore.load`) and save the data of the attribute to the first file in the set
(see :func:`pyrost.CXIStore.save`). The files may be multiple or a single one.

.. automodule:: pyrost.cxi_protocol

Contents
--------

.. toctree::
    :maxdepth: 2

    classes/cxi_protocol
    classes/cxi_store