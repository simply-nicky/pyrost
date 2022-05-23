Image transforms
================

Transforms are common image transformations. They can be chained together using :class:`pyrost.ComposeTransforms`.
You pass a :class:`pyrost.Transform` instance to a data container :class:`pyrost.STData`. All transform classes
are inherited from the abstract :class:`pyrost.Transform` class. Use :func:`pyrost.Transform.forward` to apply
transform to an image.

:class:`Transform <pyrost.Transform>`
-----------------------------------------------------

.. autoclass:: pyrost.Transform
    :members:

:class:`ComposeTransforms <pyrost.ComposeTransforms>`
-----------------------------------------------------

.. autoclass:: pyrost.ComposeTransforms
    :members:
    :inherited-members:


Transforms on images
--------------------

.. toctree::
    :maxdepth: 1

    classes/crop
    classes/mirror
    classes/downscale