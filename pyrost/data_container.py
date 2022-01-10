""":class:`DataContainer` class implementation.
"""
from __future__ import annotations
from typing import (Any, Callable, Dict, ItemsView, Iterable,
                    KeysView, Optional, ValuesView, TypeVar, Type)

Desc = TypeVar('Desc')
T = TypeVar('T')

class dict_to_object:
    """Dictionary to a new object dictionary. Creates a new object with
    the attrbiutes modified from the dictionary returned by the method.

    Attributes:
        finstance : Class bound method.
    """
    def __init__(self, finstance: Desc) -> None:
        """
        Args:
            finstance : Class bound method
        """
        self.finstance = finstance

    def __get__(self, instance: T, cls: Type[T]):
        return return_obj_method(self.finstance.__get__(instance, cls), instance, cls)

class return_obj_method:
    """Factory class that creates an object from the dictionary.

    Attributes:
        instance : Object instance.
        cls : Object class.
        sig_attrs : Signature attributes.
    """
    sig_attrs = {'__annotations__', '__doc__', '__module__',
                 '__name__', '__qualname__'}

    def __init__(self, func: Callable[..., Dict], instance: T, cls: Type[T]) -> None:
        """
        Args:
            func : Wrapped method that returns a dictionary.
            instance : Object instance.
            cls : Object class.
        """
        self.instance, self.cls = instance, cls
        self.__wrapped__ = func
        for attr in self.sig_attrs:
            self.__dict__[attr] = getattr(func, attr)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Return an object from the dictionary yielded by
        the wrapped method.

        Args:
            args : Positional arguments.
            kwargs : Keyword arguments.

        Returns:
            A new object instance.
        """
        dct = {}
        dct.update(self.__wrapped__(*args, **kwargs))
        for key, val in self.instance.items():
            if key not in dct:
                dct[key] = val
        return self.cls(**dct)

    def inplace_update(self, *args: Any, **kwargs: Any) -> None:
        """Modify the object by the dictionary yielded from
        the wrapped method.

        Args:
            args : Positional arguments.
            kwargs : Keyword arguments.
        """
        dct = self.__wrapped__(*args, **kwargs)
        for key, val in dct.items():
            self.instance.__setattr__(key, val)

class DataContainer:
    """Abstract data container class.

    Attributes:
        attr_set : Set of attributes in the container which are necessary
            to initialize in the constructor.
        init_set : Set of optional data attributes.
    """
    attr_set, init_set = set(), set()
    inits = {}

    def __init__(self, **kwargs: Any) -> None:
        """
        Args:
            kwargs : Values of the attributes specified in `attr_set` and
                `init_set`.

        Raises:
            ValueError : If an attribute specified in `attr_set` has not been
                provided.
        """
        for attr in self.attr_set:
            if kwargs.get(attr, None) is None:
                raise ValueError('Attribute {:s} has not been provided'.format(attr))

        for attr, val in kwargs.items():
            if attr in self.attr_set:
                self.__setattr__(attr, val)

        for attr in self.init_set:
            self.__setattr__(attr, kwargs.get(attr))

        for attr, init_func in self.inits.items():
            if self.get(attr, None) is None:
                self.__setattr__(attr, init_func(self))

    def __iter__(self) -> Iterable:
        return (self.attr_set | self.init_set).__iter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self.attr_set | self.init_set

    def __repr__(self) -> str:
        return {attr: self.__dict__[attr] for attr in self}.__repr__()

    def __str__(self) -> str:
        return {attr: self.__dict__[attr] for attr in self}.__str__()

    def get(self, attr: str, value: Optional[Any]=None) -> Any:
        """Retrieve a dataset, return `value` if the attribute is not found.

        Args:
            attr : Data attribute.
            value : Data which is returned if the attribute is not found.

        Returns:
            Attribute's data stored in the container, `value` if `attr`
            is not found.
        """
        return self.__dict__.get(attr, value)

    def keys(self) -> KeysView:
        """Return the list of attributes stored in the container.

        Returns:
            List of attributes stored in the container.
        """
        return {attr: self.__dict__[attr] for attr in self}.keys()

    def items(self) -> ItemsView:
        """Return (key, value) pairs of the datasets stored in the container.

        Returns:
            (key, value) pairs of the datasets stored in the container.
        """
        return {attr: self.__dict__[attr] for attr in self}.items()

    def values(self) -> ValuesView:
        """Return the attributes' data stored in the container.

        Returns:
            List of data stored in the container.
        """
        return {attr: self.__dict__[attr] for attr in self}.values()
