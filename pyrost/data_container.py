""":class:`DataContainer` class implementation.
"""
from __future__ import annotations
from typing import (Any, Callable, Dict, ItemsView, Iterable,
                    List, Optional, ValuesView, TypeVar, Type)

T = TypeVar('T')

class BoundMethod:
    __func__: Callable[..., Dict]
    __self__: T

class dict_to_object:
    """Creates a new bound method. Wraps a function implementation of a
    class bound method to return an instance of an object instead of a
    dictionary.

    Attributes:
        finstance : Class bound method.
    """
    def __init__(self, finstance: Callable[..., Dict]) -> None:
        """
        Args:
            finstance : Function object containing implementation of the
                class bound method.
        """
        self.finstance = finstance

    def __get__(self, instance: T, cls: Type[T]):
        return return_obj_method(self.finstance.__get__(instance, cls), instance, cls)

class return_obj_method:
    """Factory class that uses a bound method to return an instance of an
    object instead of a dictionary.

    Attributes:
        instance : Object instance.
        cls : Object class.
        sig_attrs : Signature attributes.
    """
    sig_attrs = {'__annotations__', '__doc__', '__module__',
                 '__name__', '__qualname__'}

    def __init__(self, method: BoundMethod, instance: T, cls: Type[T]) -> None:
        """
        Args:
            method : Wrapped method that returns a dictionary.
            instance : Object instance.
            cls : Object class.
        """
        self.instance, self.cls = instance, cls
        self.__wrapped__ = method
        for attr in self.sig_attrs:
            self.__dict__[attr] = getattr(method, attr)

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
                raise ValueError(f'Attribute {attr} has not been provided')

        for attr in self.init_set:
            self.__dict__[attr] = None

        for attr in kwargs:
            if attr in self:
                self.__setattr__(attr, kwargs.get(attr))
            else:
                raise ValueError(f'Parameter {attr} is invalid')

        self.init_funcs = {}

    def _init_functions(self, **kwargs: Callable) -> None:
        self.init_funcs.update(**kwargs)

    def _init_attributes(self) -> None:
        for attr, init_func in self.init_funcs.items():
            if self.__dict__.get(attr, None) is None:
                self.__setattr__(attr, init_func())

    def __iter__(self) -> Iterable:
        return (self.attr_set | self.init_set).__iter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self.attr_set | self.init_set

    def __getitem__(self, attr: str) -> Any:
        return self.__dict__.__getitem__(attr)

    def __repr__(self) -> str:
        return {attr: self.__dict__[attr] for attr in self
                if self.__dict__.get(attr) is not None}.__repr__()

    def __str__(self) -> str:
        return {attr: self.__dict__[attr] for attr in self
                if self.__dict__.get(attr) is not None}.__str__()

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

    def contents(self) -> List[str]:
        """Return a list of the attributes stored in the container.

        Returns:
            List of the attributes stored in the container.
        """
        return [attr for attr in self if self.get(attr) is not None]

    def keys(self) -> Iterable[str]:
        """Return a list of the attributes available in the container.

        Returns:
            List of the attributes available in the container.
        """
        return list(self)

    def items(self) -> ItemsView:
        """Return (key, value) pairs of the datasets stored in the container.

        Returns:
            (key, value) pairs of the datasets stored in the container.
        """
        return dict(self).items()

    def values(self) -> ValuesView:
        """Return the attributes' data stored in the container.

        Returns:
            List of data stored in the container.
        """
        return dict(self).values()
