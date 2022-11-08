Working at the Sigray laboratory
================================

LogProtocol
-----------

Log protocol (:class:`pyrost.LogProtocol`) is a helper class to
retrieve the data from the log files generated at the Sigray laboratory,
which contain the readouts from the motors and other instruments
during the speckle tracking scan. The data extracted from the log
files is used to generate CXI datasets by :class:`pyrost.KamzikConverter`
(look :doc:`classes/kamzik_converter`). The protocol consists of
the log keys of the attributes that are required to extract from the
header part of a log file and their corresponding data types:

* **datatypes** : Data type of the attributes (`float`, `int`, `str`,
  or `bool`).
* **log_keys** : Log key to find the attribute in the log file.
* **part_keys** : The name of the part where the attribute is stored in the log
  log file.

.. note::

    You can save protocol to an INI file with :func:`pyrost.LogProtocol.export_ini`
    and import protocol from INI file with :func:`pyrost.LogProtocol.import_ini`.

The default protocol can be accessed with :func:`pyrost.LogProtocol.import_default`.
The protocol is given by:

.. code-block:: ini

    [datatypes]
    lens_down_dist = float
    lens_up_dist = float
    exposure = float
    n_points = int
    n_steps = int
    scan_type = str
    step_size = float
    x_sample = float
    y_sample = float
    z_sample = float

    [log_keys]
    lens_down_dist = [Z-LENSE-DOWN_det_dist]
    lens_up_dist = [Z-LENSE-UP_det_dist]
    exposure = [Exposure]
    n_points = [Points count]
    n_steps = [Steps count]
    scan_type = [Device]
    step_size = [Step size]
    x_sample = [X-SAM, SAM-X, SCAN-X]
    y_sample = [Y-SAM, SAM-Y, SCAN-Y]
    z_sample = [Z-SAM, SAM-Z, SCAN-Z]

    [part_keys]
    lens_down_dist = Session logged attributes
    lens_up_dist = Session logged attributes
    exposure = Type: Method
    n_points = Type: Scan
    n_steps = Type: Scan
    scan_type = Type: Scan
    step_size = Type: Scan
    x_sample = Session logged attributes
    y_sample = Session logged attributes
    z_sample = Session logged attributes

KamzikConverter
---------------

Log data converter class provides an interface to read Kamzik log files
(:func:`pyrost.KamzikConverter.read_logs`), and convert data to CXI attributes,
that can be parsed to :class:`pyrost.STData` container. A converter object
needs a log protocol (:class:`pyrost.LogProtocol`), a fast scan detector axis, and
a slow scan detector axis. One can obtain a list of CXI datasets available to
generate with :func:`pyrost.KamzikConverter.cxi_keys`, and generate a dictionary of
CXI datasets with :func:`pyrost.KamzikConverter.cxi_get`.

The following attributes can be extracted from log files:

* **basis_vectors** : A set of detector axes defined for each image.
* **dist_down** : down-lens-to-detector distance in meters.
* **dist_up** : up-lens-to-detector distance in meters.
* **sim_translations** : A set of sample translations simulated from the log attributes.
* **log_translations** : A set of sample translations read from the log file.
* **x_pixel_size** : Detector pixel size along the x-axis in meters.
* **y_pixel_size** : Detector pixel size along the y-axis in meters.

Contents
--------

.. toctree::
    :maxdepth: 2

    classes/log_protocol
    classes/kamzik_converter