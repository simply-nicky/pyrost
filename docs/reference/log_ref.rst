Working at the Sigray laboratory
================================

LogProtocol
-----------

Log protocol (:class:`pyrost.LogProtocol`) is a helper class to
retrieve the data from the log files generated at the Sigray laboratory,
which contain the readouts from the motors and other instruments
during the speckle tracking scan. The data extracted from the log
files is used to generate CXI files (look :doc:`funcs/cxi_converter_sigray`).
The protocol consists of the log keys of the attributes that are
requisite to extract from the header part of a log file and their
corresponding data types:

* `datatypes` : Data type of the attributes (`float`, `int`, `str`,
  or `bool`).
* `log_keys` : Log key to find the attribute in the log file.
* `part_keys` : The name of the part where the attribute is stored in the log
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

Contents
--------

.. toctree::
    :maxdepth: 2

    classes/log_protocol
    funcs/cxi_converter_sigray
    funcs/tilt_converter_sigray