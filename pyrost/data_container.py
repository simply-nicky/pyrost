""":class:`DataContainer` class implementation.
"""
import numpy as np

class dict_to_object:
    def __init__(self, finstance):
        self.finstance = finstance

    def __get__(self, instance, cls):
        return return_obj_method(self.finstance.__get__(instance, cls), instance, cls)

class return_obj_method:
    sig_attrs = {'__annotations__', '__doc__', '__module__',
                 '__name__', '__qualname__'}

    def __init__(self, func, instance, cls):
        self.instance, self.cls = instance, cls
        self.__wrapped__ = func
        for attr in self.sig_attrs:
            self.__dict__[attr] = getattr(func, attr)

    def __call__(self, *args, **kwargs):
        dct = {}
        for key, val in self.instance.items():
            if isinstance(val, np.ndarray):
                dct[key] = np.copy(val)
            else:
                dct[key] = val
        dct.update(self.__wrapped__(*args, **kwargs))
        return self.cls(**dct)

    def inplace_update(self, *args, **kwargs):
        dct = self.__wrapped__(*args, **kwargs)
        for key, val in dct.items():
            self.instance.__setattr__(key, val)

class DataContainer:
    """Abstract data container class.

    Parameters
    ----------
    **kwargs : dict
        Values of the attributes specified in `attr_set` and `init_set`.

    Attributes
    ----------
    attr_set : set
        Set of attributes in the container which are necessary
        to initialize in the constructor.
    init_set : set
        Set of optional data attributes.

    Raises
    ------
    ValueError
        If an attribute specified in `attr_set` has not been provided.
    """
    attr_set, init_set = {}, {}

    def __init__(self, **kwargs):
        self.__dict__['attr_dict'] = {}
        for attr in self.attr_set:
            if kwargs.get(attr) is None:
                raise ValueError('Attribute {:s} has not been provided'.format(attr))
            else:
                self.attr_dict[attr] = kwargs.get(attr)
        for attr in self.init_set:
            self.attr_dict[attr] = kwargs.get(attr)

    def __iter__(self):
        return self.attr_dict.__iter__()

    def __contains__(self, attr):
        return attr in self.attr_dict

    def __getattr__(self, attr):
        if attr in self:
            return self.attr_dict[attr]
        else:
            raise AttributeError(attr + " doesn't exist")

    def __repr__(self):
        return self.attr_dict.__repr__()

    def __setattr__(self, attr, value):
        if attr in self.__dict__['attr_dict']:
            self.__dict__['attr_dict'][attr] = value
        else:
            super(DataContainer, self).__setattr__(attr, value)

    def __str__(self):
        return self.attr_dict.__str__()

    def get(self, attr, value=None):
        """Retrieve a dataset, return `value` if the attribute is not found.

        Parameters
        ----------
        attr : str
            Data attribute.
        value : object, optional
            Data which is returned if the attribute is not found.

        Returns
        -------
        object
            Attribute's data stored in the container,
            `value` if `attr` is not found.
        """
        return self.attr_dict.get(attr, value)

    def keys(self):
        """Return the list of attributes stored in the container.

        Returns
        -------
        dict_keys
            List of attributes stored in the container.
        """
        return self.attr_dict.keys()

    def items(self):
        """Return (key, value) pairs of the datasets stored in the container.

        Returns
        -------
        dict_items
            (key, value) pairs of the datasets stored in the container.
        """
        return self.attr_dict.items()

    def values(self):
        """Return the attributes' data stored in the container.

        Returns
        -------
        dict_values
            List of data stored in the container.
        """
        return self.attr_dict.values()
