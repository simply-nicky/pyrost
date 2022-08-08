from typing import Optional, Tuple
import numpy as np

def byte_align(array: np.ndarray, n: Optional[int]=None,
               dtype: Optional[np.dtype]=None) -> np.ndarray:
    """Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where `n` is an optional parameter. If `n` is not provided
    then this function will inspect the CPU to determine alignment. If the
    array is aligned then it is returned without further ado.  If it is not
    aligned then a new array is created and the data copied in, but aligned
    on the n-byte boundary.

    `dtype` is an optional argument that forces the resultant array to be
    of that dtype.
    """
    ...

def is_byte_aligned(array: np.ndarray, n: Optional[int]=None) -> bool:
    """Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where `n` is an optional parameter, returning `True` if it is,
    and `False` if it is not. If `n` is not provided then this function will
    inspect the CPU to determine alignment.
    """
    ...

def empty_aligned(shape: Tuple[int, ...], dtype: str='float64', order: str='C',
                  n: Optional[int]=None) -> np.ndarray:
    """Function that returns an empty numpy array that is n-byte aligned,
    where `n` is determined by inspecting the CPU if it is not
    provided.

    The alignment is given by the final optional argument, `n`. If
    `n` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.empty`.
    """
    ...

def zeros_aligned(shape: Tuple[int, ...], dtype: str='float64', order: str='C',
                  n: Optional[int]=None) -> np.ndarray:
    """Function that returns a numpy array of zeros that is n-byte aligned,
    where `n` is determined by inspecting the CPU if it is not provided.

    The alignment is given by the final optional argument, `n`. If
    `n` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.zeros`.
    """
    ...

def ones_aligned(shape: Tuple[int, ...], dtype: str='float64', order: str='C',
                 n: Optional[int]=None) -> np.ndarray:
    """Function that returns a numpy array of ones that is n-byte aligned,
    where `n` is determined by inspecting the CPU if it is not
    provided.

    The alignment is given by the final optional argument, `n`. If
    `n` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.ones`.
    """
    ...

class FFTW:
    """FFTW is a class for computing a variety of discrete Fourier
    transforms of multidimensional, strided arrays using the FFTW
    library. The interface is designed to be somewhat pythonic, with
    the correct transform being inferred from the dtypes of the passed
    arrays.

    The exact scheme may be either directly specified with the
    `direction` parameter or inferred from the dtypes and relative
    shapes of the input arrays. Information on which shapes and dtypes
    imply which transformations is available in the :ref:`FFTW schemes
    <scheme_table>`. If a match is found, the plan corresponding to that
    scheme is created, operating on the arrays that are passed in. If no
    scheme can be created then a `ValueError` is raised.

    The actual transformation is performed by calling the
    :meth:`~pyfftw.FFTW.execute` method.

    The arrays can be updated by calling the
    :meth:`~pyfftw.FFTW.update_arrays` method.

    The created instance of the class is itself callable, and can perform
    the execution of the FFT, both with or without array updates, returning
    the result of the FFT. Unlike calling the :meth:`~pyfftw.FFTW.execute`
    method, calling the class instance will also optionally normalise the
    output as necessary. Additionally, calling with an input array update
    will also coerce that array to be the correct dtype.

    See the documentation on the :meth:`~pyfftw.FFTW.__call__` method
    for more information.
    """

    N                   : int
    simd_aligned        : bool
    input_alignment     : int
    output_alignment    : int
    flags               : Tuple[str, ...]
    input_array         : np.ndarray
    output_array        : np.ndarray
    input_strides       : Tuple[int, ...]
    output_strides      : Tuple[int, ...]
    input_shape         : Tuple[int, ...]
    output_shape        : Tuple[int, ...]
    input_dtype         : np.dtype
    output_dtype        : np.dtype
    direction           : str
    axes                : Tuple[int, ...]
    normalize_idft      : bool
    ortho               : bool

    def __init__(self, input_array: np.ndarray, output_array: np.ndarray,
                 axes: Tuple[int, ...]=(-1,), direction: str='FFTW_FORWARD',
                 flags: Tuple[str, ...]=('FFTW_MEASURE',), threads: int=1,
                 planning_timelimit: Optional[float]=None, normalise_idft: bool=True,
                 ortho: bool=False) -> None:
        r"""**Arguments**:

        * `input_array` and `output_array` should be numpy arrays.
          The contents of these arrays will be destroyed by the planning
          process during initialisation. Information on supported
          dtypes for the arrays is :ref:`given below <scheme_table>`.

        * `axes` describes along which axes the DFT should be taken.
          This should be a valid list of axes. Repeated axes are
          only transformed once. Invalid axes will raise an `IndexError`
          exception. This argument is equivalent to the same
          argument in :func:`numpy.fft.fftn`, except for the fact that
          the behaviour of repeated axes is different (`numpy.fft`
          will happily take the fft of the same axis if it is repeated
          in the `axes` argument). Rudimentary testing has suggested
          this is down to the underlying FFTW library and so unlikely
          to be fixed in these wrappers.

        * The `direction` parameter describes what sort of
          transformation the object should compute. This parameter is
          poorly named for historical reasons: older versions of pyFFTW
          only supported forward and backward transformations, for which
          this name made sense. Since then pyFFTW has been expanded to
          support real to real transforms as well and the name is not
          quite as descriptive.

          `direction` should either be a string, or, in the case of
          multiple real transforms, a list of strings. The two values
          corresponding to the DFT are

          * `'FFTW_FORWARD'`, which is the forward discrete Fourier
            transform, and
          * `'FFTW_BACKWARD'`, which is the backward discrete Fourier
            transform.

          Note that, for the two above options, only the Complex schemes
          allow a free choice for `direction`. The direction *must*
          agree with the the :ref:`table below <scheme_table>` if a Real
          scheme is used, otherwise a `ValueError` is raised.


          Alternatively, if you are interested in one of the real to real
          transforms, then pyFFTW supports four different discrete cosine
          transforms:

          * `'FFTW_REDFT00'`,
          * `'FFTW_REDFT01'`,
          * `'FFTW_REDFT10'`, and
          * `'FFTW_REDFT01'`,

          and four discrete sine transforms:

          * `'FFTW_RODFT00'`,
          * `'FFTW_RODFT01'`,
          * `'FFTW_RODFT10'`, and
          * `'FFTW_RODFT01'`.

          pyFFTW uses the same naming convention for these flags as FFTW:
          the `'REDFT'` part of the name is an acronym for 'real even
          discrete Fourier transform, and, similarly, `'RODFT'` stands
          for 'real odd discrete Fourier transform'. The trailing `'0'`
          is notation for even data (in terms of symmetry) and the
          trailing `'1'` is for odd data.

          Unlike the plain discrete Fourier transform, one may specify a
          different real to real transformation over each axis: for example,

          .. code-block:: none
             a = pyfftw.empty_aligned((128,128,128))
             b = pyfftw.empty_aligned((128,128,128))
             directions = ['FFTW_REDFT00', 'FFTW_RODFT11']
             transform = pyfftw.FFTW(a, b, axes=(0, 2), direction=directions)

          will create a transformation across the first and last axes
          with a discrete cosine transform over the first and a discrete
          sine transform over the last.

          Unfortunately, since this class is ultimately just a wrapper
          for various transforms implemented in FFTW, one cannot combine
          real transformations with real to complex transformations in a
          single object.

        .. _FFTW_flags:

        * `flags` is a list of strings and is a subset of the
          flags that FFTW allows for the planners:

          * `'FFTW_ESTIMATE'`, `'FFTW_MEASURE'`, `'FFTW_PATIENT'` and
            `'FFTW_EXHAUSTIVE'` are supported. These describe the
            increasing amount of effort spent during the planning
            stage to create the fastest possible transform.
            Usually `'FFTW_MEASURE'` is a good compromise. If no flag
            is passed, the default `'FFTW_MEASURE'` is used.
          * `'FFTW_UNALIGNED'` is supported.
            This tells FFTW not to assume anything about the
            alignment of the data and disabling any SIMD capability
            (see below).
          * `'FFTW_DESTROY_INPUT'` is supported.
            This tells FFTW that the input array can be destroyed during
            the transform, sometimes allowing a faster algorithm to be
            used. The default behaviour is, if possible, to preserve the
            input. In the case of the 1D Backwards Real transform, this
            may result in a performance hit. In the case of a backwards
            real transform for greater than one dimension, it is not
            possible to preserve the input, making this flag implicit
            in that case. A little more on this is given
            :ref:`below<scheme_table>`.
          * `'FFTW_WISDOM_ONLY'` is supported.
            This tells FFTW to raise an error if no plan for this transform
            and data type is already in the wisdom. It thus provides a method
            to determine whether planning would require additional effort or the
            cached wisdom can be used. This flag should be combined with the
            various planning-effort flags (`'FFTW_ESTIMATE'`,
            `'FFTW_MEASURE'`, etc.); if so, then an error will be raised if
            wisdom derived from that level of planning effort (or higher) is
            not present. If no planning-effort flag is used, the default of
            `'FFTW_ESTIMATE'` is assumed.
            Note that wisdom is specific to all the parameters, including the
            data alignment. That is, if wisdom was generated with input/output
            arrays with one specific alignment, using `'FFTW_WISDOM_ONLY'`
            to create a plan for arrays with any different alignment will
            cause the `'FFTW_WISDOM_ONLY'` planning to fail. Thus it is
            important to specifically control the data alignment to make the
            best use of `'FFTW_WISDOM_ONLY'`.

          The `FFTW planner flags documentation
          <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_
          has more information about the various flags and their impact.
          Note that only the flags documented here are supported.

        * `threads` tells the wrapper how many threads to use
          when invoking FFTW, with a default of 1. If the number
          of threads is greater than 1, then the GIL is released
          by necessity.

        * `planning_timelimit` is a floating point number that
          indicates to the underlying FFTW planner the maximum number of
          seconds it should spend planning the FFT. This is a rough
          estimate and corresponds to calling of `fftw_set_timelimit()`
          (or an equivalent dependent on type) in the underlying FFTW
          library. If `None` is set, the planner will run indefinitely
          until all the planning modes allowed by the flags have been
          tried. See the `FFTW planner flags page
          <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_
          for more information on this.

        .. _fftw_schemes:

        **Schemes**

        The currently supported full (so not discrete sine or discrete
        cosine) DFT schemes are as follows:

        .. _scheme_table:

        +----------------+-----------------------+------------------------+-----------+
        | Type           | `input_array.dtype` | `output_array.dtype` | Direction |
        +================+=======================+========================+===========+
        | Complex        | `complex64`         | `complex64`          | Both      |
        +----------------+-----------------------+------------------------+-----------+
        | Complex        | `complex128`        | `complex128`         | Both      |
        +----------------+-----------------------+------------------------+-----------+
        | Complex        | `clongdouble`       | `clongdouble`        | Both      |
        +----------------+-----------------------+------------------------+-----------+
        | Real           | `float32`           | `complex64`          | Forwards  |
        +----------------+-----------------------+------------------------+-----------+
        | Real           | `float64`           | `complex128`         | Forwards  |
        +----------------+-----------------------+------------------------+-----------+
        | Real           | `longdouble`        | `clongdouble`        | Forwards  |
        +----------------+-----------------------+------------------------+-----------+
        | Real\ :sup:`1` | `complex64`         | `float32`            | Backwards |
        +----------------+-----------------------+------------------------+-----------+
        | Real\ :sup:`1` | `complex128`        | `float64`            | Backwards |
        +----------------+-----------------------+------------------------+-----------+
        | Real\ :sup:`1` | `clongdouble`       | `longdouble`         | Backwards |
        +----------------+-----------------------+------------------------+-----------+

        \ :sup:`1`  Note that the Backwards Real transform for the case
        in which the dimensionality of the transform is greater than 1
        will destroy the input array. This is inherent to FFTW and the only
        general work-around for this is to copy the array prior to
        performing the transform. In the case where the dimensionality
        of the transform is 1, the default is to preserve the input array.
        This is different from the default in the underlying library, and
        some speed gain may be achieved by allowing the input array to
        be destroyed by passing the `'FFTW_DESTROY_INPUT'`
        :ref:`flag <FFTW_flags>`.

        The discrete sine and discrete cosine transforms are supported
        for all three real types.

        `clongdouble` typically maps directly to `complex256`
        or `complex192`, and `longdouble` to `float128` or
        `float96`, dependent on platform.

        The relative shapes of the arrays should be as follows:

        * For a Complex transform, `output_array.shape == input_array.shape`
        * For a Real transform in the Forwards direction, both the following
          should be true:

          * `output_array.shape[axes][-1] == input_array.shape[axes][-1]//2 + 1`
          * All the other axes should be equal in length.

        * For a Real transform in the Backwards direction, both the following
          should be true:

          * `input_array.shape[axes][-1] == output_array.shape[axes][-1]//2 + 1`
          * All the other axes should be equal in length.

        In the above expressions for the Real transform, the `axes`
        arguments denotes the unique set of axes on which we are taking
        the FFT, in the order passed. It is the last of these axes that
        is subject to the special case shown.

        The shapes for the real transforms corresponds to those
        stipulated by the FFTW library. Further information can be
        found in the FFTW documentation on the `real DFT
        <http://www.fftw.org/fftw3_doc/Guru-Real_002ddata-DFTs.html>`_.

        The actual arrangement in memory is arbitrary and the scheme
        can be planned for any set of strides on either the input
        or the output. The user should not have to worry about this
        and any valid numpy array should work just fine.

        What is calculated is exactly what FFTW calculates.
        Notably, this is an unnormalized transform so should
        be scaled as necessary (fft followed by ifft will scale
        the input by N, the product of the dimensions along which
        the DFT is taken). For further information, see the
        `FFTW documentation
        <http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html>`_.

        The FFTW library benefits greatly from the beginning of each
        DFT axes being aligned on the correct byte boundary, enabling
        SIMD instructions. By default, if the data begins on such a
        boundary, then FFTW will be allowed to try and enable
        SIMD instructions. This means that all future changes to
        the data arrays will be checked for similar alignment. SIMD
        instructions can be explicitly disabled by setting the
        FFTW_UNALIGNED flags, to allow for updates with unaligned
        data.

        :func:`~pyfftw.byte_align` and
        :func:`~pyfftw.empty_aligned` are two methods
        included with this module for producing aligned arrays.

        The optimum alignment for the running platform is provided
        by :data:`pyfftw.simd_alignment`, though a different alignment
        may still result in some performance improvement. For example,
        if the processor supports AVX (requiring 32-byte alignment) as
        well as SSE (requiring 16-byte alignment), then if the array
        is 16-byte aligned, SSE will still be used.

        It's worth noting that just being aligned may not be sufficient
        to create the fastest possible transform. For example, if the
        array is not contiguous (i.e. certain axes are displaced in
        memory), it may be faster to plan a transform for a contiguous
        array, and then rely on the array being copied in before the
        transform (which :class:`pyfftw.FFTW` will handle for you when
        accessed through :meth:`~pyfftw.FFTW.__call__`).
        """
        ...

    def __call__(self, input_array: Optional[np.ndarray]=None,
                 output_array: Optional[np.ndarray]=None,
                 normalise_idft: bool=None, ortho: bool=None) -> np.ndarray:
        """Calling the class instance (optionally) updates the arrays, then
        calls :meth:`~pyfftw.FFTW.execute`, before optionally normalising
        the output and returning the output array.

        It has some built-in helpers to make life simpler for the calling
        functions (as distinct from manually updating the arrays and
        calling :meth:`~pyfftw.FFTW.execute`).

        If `normalise_idft` is `True` (the default), then the output from
        an inverse DFT (i.e. when the direction flag is `'FFTW_BACKWARD'`) is
        scaled by 1/N, where N is the product of the lengths of input array on
        which the FFT is taken. If the direction is `'FFTW_FORWARD'`, this
        flag makes no difference to the output array.

        If `ortho` is `True`, then the output of both forward
        and inverse DFT operations is scaled by 1/sqrt(N), where N is the
        product of the lengths of input array on which the FFT is taken.  This
        ensures that the DFT is a unitary operation, meaning that it satisfies
        Parseval's theorem (the sum of the squared values of the transform
        output is equal to the sum of the squared values of the input).  In
        other words, the energy of the signal is preserved.

        If either `normalise_idft` or `ortho` are `True`, then
        ifft(fft(A)) = A.

        When `input_array` is something other than None, then the passed in
        array is coerced to be the same dtype as the input array used when the
        class was instantiated, the byte-alignment of the passed in array is
        made consistent with the expected byte-alignment and the striding is
        made consistent with the expected striding. All this may, but not
        necessarily, require a copy to be made.

        As noted in the :ref:`scheme table<scheme_table>`, if the FFTW
        instance describes a backwards real transform of more than one
        dimension, the contents of the input array will be destroyed. It is
        up to the calling function to make a copy if it is necessary to
        maintain the input array.

        `output_array` is always used as-is if possible. If the dtype, the
        alignment or the striding is incorrect for the FFTW object, then a
        `ValueError` is raised.

        The coerced input array and the output array (as appropriate) are
        then passed as arguments to
        :meth:`~pyfftw.FFTW.update_arrays`, after which
        :meth:`~pyfftw.FFTW.execute` is called, and then normalisation
        is applied to the output array if that is desired.

        Note that it is possible to pass some data structure that can be
        converted to an array, such as a list, so long as it fits the data
        requirements of the class instance, such as array shape.

        Other than the dtype and the alignment of the passed in arrays, the
        rest of the requirements on the arrays mandated by
        :meth:`~pyfftw.FFTW.update_arrays` are enforced.

        A `None` argument to either keyword means that that array is not
        updated.

        The result of the FFT is returned. This is the same array that is used
        internally and will be overwritten again on subsequent calls. If you
        need the data to persist longer than a subsequent call, you should
        copy the returned array.
        """
        ...

    def update_arrays(self, new_input_array: np.ndarray, new_output_array: np.ndarray) -> None:
        """Update the arrays upon which the DFT is taken.

        The new arrays should be of the same dtypes as the originals, the same
        shapes as the originals and should have the same strides between axes.
        If the original data was aligned so as to allow SIMD instructions
        (e.g. by being aligned on a 16-byte boundary), then the new array must
        also be aligned so as to allow SIMD instructions (assuming, of
        course, that the `FFTW_UNALIGNED` flag was not enabled).

        The byte alignment requirement extends to requiring natural
        alignment in the non-SIMD cases as well, but this is much less
        stringent as it simply means avoiding arrays shifted by, say,
        a single byte (which invariably takes some effort to
        achieve!).

        If all these conditions are not met, a `ValueError` will
        be raised and the data will *not* be updated (though the
        object will still be in a sane state).
        """
        ...

    def execute(self) -> None:
        """Execute the planned operation, taking the correct kind of FFT of
        the input array (i.e. :attr:`FFTW.input_array`), and putting the
        result in the output array (i.e. :attr:`FFTW.output_array`).
        """
        ...
