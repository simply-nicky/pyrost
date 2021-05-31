Working with Log Files
======================

LogProtocol
-----------

Log protocol (:class:`pyrost.LogProtocol`) is a helper class to
retrieve the data from the log files, which contain the readouts
from the motors and other instruments during the speckle tracking
scan. In the most cases, the data extracted from the log files is
used to generate the CXI files (look :doc:`cxi_ref.rst`). The protocol
consists of the log keys of the attributes that are requisite to
extract from the header part of a log file and their corresponding
data types:

* `datatypes` : Data type of the attributes (`float`, `int`, `str`,
  or `bool`).
* `log_keys` : Log key to find the attribute in the log file.

.. note::

    You can save protocol to an INI file with :func:`pyrost.LogProtocol.export_ini`
    and import protocol from INI file with :func:`pyrost.LogProtocol.import_ini`.

The default protocol can be accessed with :func:`pyrost.LogProtocol.import_default`.
The protocol is given by:

..code-block:: ini

    [datatypes]
    det_dist = float
    exposure = float
    n_steps = int
    scan_type = str
    step_size = float
    x_sample = float
    y_sample = float
    z_sample = float

    [log_keys]
    det_dist = [Session logged attributes, Z-LENSE-DOWN_det_dist]
    exposure = [Type: Method, Exposure]
    n_steps = [Type: Scan, Points count]
    scan_type = [Type: Scan, Device]
    step_size = [Type: Scan, Step size]
    x_sample = [Session logged attributes, X-SAM]
    y_sample = [Session logged attributes, Y-SAM]
    z_sample = [Session logged attributes, Z-SAM]

Contents
--------

..toctree::
    :maxdepth: 1

    classes/log_protocol