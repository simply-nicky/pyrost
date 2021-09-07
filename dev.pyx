#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
import cython
from libc.stdlib cimport malloc, free
from cpython.ref cimport Py_INCREF
import speckle_tracking as st

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

cdef int extend_mode_to_code(str mode) except -1:
    if mode == 'constant':
        return EXTEND_CONSTANT
    elif mode == 'nearest':
        return EXTEND_NEAREST
    elif mode == 'mirror':
        return EXTEND_MIRROR
    elif mode == 'reflect':
        return EXTEND_REFLECT
    elif mode == 'wrap':
        return EXTEND_WRAP
    else:
        raise RuntimeError('boundary mode not supported')

cdef np.ndarray check_array(np.ndarray array, int type_num):
    if not np.PyArray_IS_C_CONTIGUOUS(array):
        array = np.PyArray_GETCONTIGUOUS(array)
    cdef int tn = np.PyArray_TYPE(array)
    if tn != type_num:
        array = np.PyArray_Cast(array, type_num)
    return array

cdef np.ndarray number_to_array(object num, np.npy_intp rank, int type_num):
    cdef np.npy_intp *dims = [rank,]
    cdef np.ndarray arr = <np.ndarray>np.PyArray_SimpleNew(1, dims, type_num)
    cdef int i
    for i in range(rank):
        arr[i] = num
    return arr

cdef np.ndarray normalize_sequence(object inp, np.npy_intp rank, int type_num):
    # If input is a scalar, create a sequence of length equal to the
    # rank by duplicating the input. If input is a sequence,
    # check if its length is equal to the length of array.
    cdef np.ndarray arr
    cdef int tn
    if np.PyArray_IsAnyScalar(inp):
        arr = number_to_array(inp, rank, type_num)
    elif np.PyArray_Check(inp):
        arr = <np.ndarray>inp
        tn = np.PyArray_TYPE(arr)
        if tn != type_num:
            arr = <np.ndarray>np.PyArray_Cast(arr, type_num)
    elif isinstance(inp, (list, tuple)):
        arr = <np.ndarray>np.PyArray_FROM_OTF(inp, type_num, np.NPY_ARRAY_C_CONTIGUOUS)
    else:
        raise ValueError("Wrong sequence argument type")
    cdef np.npy_intp size = np.PyArray_SIZE(arr)
    if size != rank:
        raise ValueError("Sequence argument must have length equal to input rank")
    return arr

cdef np.ndarray object_to_array(object inp, int type_num):
    # If input is a scalar, create a sequence of length equal to the
    # rank by duplicating the input. If input is a sequence,
    # check if its length is equal to the length of array.
    cdef np.ndarray arr
    cdef int tn
    if np.PyArray_IsAnyScalar(inp):
        arr = number_to_array(inp, 1, type_num)
    elif np.PyArray_Check(inp):
        arr = <np.ndarray>inp
        tn = np.PyArray_TYPE(arr)
        if tn != type_num:
            arr = <np.ndarray>np.PyArray_Cast(arr, type_num)
    elif isinstance(inp, (list, tuple)):
        arr = <np.ndarray>np.PyArray_FROM_OTF(inp, type_num, np.NPY_ARRAY_C_CONTIGUOUS)
    else:
        raise ValueError("Wrong sequence argument type")
    return arr

def median_filter(data: np.ndarray, size: object, mask: np.ndarray=None, mode: str='reflect', cval: cython.double=0.,
                  num_threads: cython.uint=1) -> np.ndarray:
    """Calculate a median along the `axis`.

    Parameters
    ----------
    data : numpy.ndarray
        Intensity frames.
    size : numpy.ndarray
        Gives the shape that is taken from the input array, at every element position, to
        define the input to the filter function. We adjust size to the number of dimensions
        of the input array, so that, if the input array is shape (10,10,10), and size is 2,
        then the actual size used is (2,2,2).
    mask : numpy.ndarray, optional
        Bad pixel mask.
    mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}, optional
        The mode parameter determines how the input array is extended when the filter
        overlaps a border. Default value is 'reflect'. The valid values and their behavior
        is as follows:

        * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
          values beyond the edge with the same constant value, defined by the `cval`
          parameter.
        * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating
          the last pixel.
        * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting
          about the center of the last pixel. This mode is also sometimes referred to as
          whole-sample symmetric.
        * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting
          about the edge of the last pixel. This mode is also sometimes referred to as
          half-sample symmetric.
        * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around
          to the opposite edge.
    cval : float, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    wfield : numpy.ndarray
        Whitefield.
    """
    if not np.PyArray_IS_C_CONTIGUOUS(data):
        data = np.PyArray_GETCONTIGUOUS(data)

    cdef int ndim = data.ndim
    cdef np.npy_intp *dims = data.shape

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)

    cdef np.ndarray fsize = normalize_sequence(size, ndim, np.NPY_INTP)
    cdef unsigned long *_fsize = <unsigned long *>np.PyArray_DATA(fsize)

    cdef unsigned long *_dims = <unsigned long *>dims
    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef int _mode = extend_mode_to_code(mode)
    cdef void *_cval = <void *>&cval

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 8, _fsize, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _mode, _cval, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _mode, _cval, compare_uint, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    if fail:
        raise RuntimeError('C backend exited with error.')  
    return out

cpdef np.ndarray dot(np.ndarray inp1, np.ndarray inp2, int axis1, int axis2, unsigned int num_threads=1):
    cdef int ndim1 = inp1.ndim
    axis1 = axis1 if axis1 >= 0 else ndim1 + axis1
    axis1 = axis1 if axis1 <= ndim1 - 1 else ndim1 - 1
    if axis1 != ndim1 - 1:
        inp1 = np.PyArray_SwapAxes(inp1, axis1, ndim1 - 1)
        axis1 = ndim1 - 1

    cdef int ndim2 = inp2.ndim
    axis2 = axis2 if axis2 >= 0 else ndim2 + axis2
    axis2 = axis2 if axis2 <= ndim2 - 1 else ndim2 - 1
    if axis2 != ndim2 - 1:
        inp2 = np.PyArray_SwapAxes(inp2, axis2, ndim2 - 1)
        axis2 = ndim2 - 1

    if not np.PyArray_IS_C_CONTIGUOUS(inp1):
        inp1 = np.PyArray_GETCONTIGUOUS(inp1)
    if not np.PyArray_IS_C_CONTIGUOUS(inp2):
        inp2 = np.PyArray_GETCONTIGUOUS(inp2)

    cdef np.npy_intp *dims1 = inp1.shape
    cdef np.npy_intp *dims2 = inp2.shape
    cdef unsigned long *_dims1 = <unsigned long *>dims1
    cdef unsigned long *_dims2 = <unsigned long *>dims2
    if dims1[axis1] != dims2[axis2]:
        raise ValueError(f"incompatible shapes: inp1.shape[axis1] ({dims1[axis1]:d}) "\
                         f"!= inp2.shape[axis2] ({dims2[axis2]:d}).")
    if np.PyArray_TYPE(inp1) != np.PyArray_TYPE(inp2):
        raise ValueError("Incompatible data types.")

    cdef int ondim = ndim1 + ndim2 - 2 if ndim1 + ndim2 > 2 else 1
    cdef np.npy_intp *odims = <np.npy_intp *>malloc(ondim * sizeof(np.npy_intp))
    if odims is NULL:
        raise MemoryError('not enough memory')

    odims[0] = 1
    cdef int i
    for i in range(ndim1 - 1):
        odims[i] = inp1.shape[i]
    for i in range(ndim2 - 1):
        odims[i + ndim1 - 1] = inp2.shape[i]

    cdef int type_num = np.PyArray_TYPE(inp1)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ondim, odims, type_num)
    
    cdef void *_out = np.PyArray_DATA(out)
    cdef void *_inp1 = np.PyArray_DATA(inp1)
    cdef void *_inp2 = np.PyArray_DATA(inp2)

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = dot_c(_out, _inp1, ndim1, _dims1, axis1, _inp2, ndim2,
                         _dims2, axis2, 8, dot_double, num_threads)
        elif type_num == np.NPY_INT64:
            fail = dot_c(_out, _inp1, ndim1, _dims1, axis1, _inp2, ndim2,
                         _dims2, axis2, 8, dot_long, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(inp1.dtype))

    free(odims)
    return out

cdef void init_dims(np.PyArray_Dims *permute, np.PyArray_Dims *reshape, np.npy_intp *shape, np.ndarray axes, int ndim):
    cdef int i, ii = 0
    cdef np.npy_intp nax = axes.size
    permute.len = ndim
    reshape.len = ndim - nax + 1
    permute.ptr = <np.npy_intp *>malloc(permute.len * sizeof(np.npy_intp))
    reshape.ptr = <np.npy_intp *>malloc(reshape.len * sizeof(np.npy_intp))
    cdef bint matched
    for i in range(ndim):
        matched = 0
        for j in range(nax):
            matched |= (axes[j] == i)
        if not matched:
            permute.ptr[ii] = i
            reshape.ptr[ii] = shape[i]
            ii += 1
    if ndim != ii + nax:
        raise ValueError("axes sequence must match to the shape of the array.")
    reshape.ptr[ii] = 1
    for i in range(0, nax):
        permute.ptr[i + ii] = axes[i]
        reshape.ptr[ii] *= shape[axes[i]]

cdef void free_dims(np.PyArray_Dims *dims) nogil:
    free(dims.ptr)
    free(dims)

def tensordot(inp1: np.ndarray, inp2: np.ndarray, axes1: object, axes2: object, num_threads: cython.uint=1) -> np.ndarray:
    cdef np.ndarray ax1 = object_to_array(axes1, np.NPY_INT64)
    cdef np.ndarray ax2 = object_to_array(axes2, np.NPY_INT64)

    if ax1.size != ax2.size:
        raise ValueError("len(axes1) and len(axes2) must be equal.")
    if ax1.size > inp1.ndim or ax1.size > inp2.ndim:
        raise ValueError("Too many axes provided.")
    
    cdef np.PyArray_Dims *perm1 = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
    cdef np.PyArray_Dims *rshp1 = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
    init_dims(perm1, rshp1, inp1.shape, ax1, inp1.ndim)
    
    cdef np.ndarray ninp1 = np.PyArray_Newshape(np.PyArray_Transpose(inp1, perm1), rshp1, np.NPY_CORDER)
    free_dims(perm1)
    free_dims(rshp1)

    cdef np.PyArray_Dims *perm2 = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
    cdef np.PyArray_Dims *rshp2 = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
    init_dims(perm2, rshp2, inp2.shape, ax2, inp2.ndim)
    
    cdef np.ndarray ninp2 = np.PyArray_Newshape(np.PyArray_Transpose(inp2, perm2), rshp2, np.NPY_CORDER)
    free_dims(perm2)
    free_dims(rshp2)

    cdef np.ndarray out = dot(ninp1, ninp2, ninp1.ndim - 1, ninp2.ndim - 1, num_threads)
    return out

cdef class ArrayWrapper:
    """A wrapper class for a C data structure. """
    cdef void* _data

    def __cinit__(self):
        self._data = NULL

    def __dealloc__(self):
        if not self._data is NULL:
            free(self._data)
            self._data = NULL

    @staticmethod
    cdef ArrayWrapper from_ptr(void *data):
        """Factory function to create a new wrapper from a C pointer."""
        cdef ArrayWrapper wrapper = ArrayWrapper.__new__(ArrayWrapper)
        wrapper._data = data
        return wrapper

    cdef np.ndarray to_ndarray(self, int ndim, np.npy_intp *dims, int type_num):
        """Get a NumPy array from a wrapper."""
        cdef np.ndarray ndarray = np.PyArray_SimpleNewFromData(ndim, dims, type_num, self._data)

        # without this, data would be cleaned up right away
        Py_INCREF(self)
        np.PyArray_SetBaseObject(ndarray, self)
        return ndarray

def make_reference(I_n: np.ndarray, W: np.ndarray, u: np.ndarray, di: np.ndarray, dj: np.ndarray,
                   ls: cython.double, num_threads: cython.uint=1):
    if not np.PyArray_IS_C_CONTIGUOUS(I_n):
        I_n = np.PyArray_GETCONTIGUOUS(I_n)
    if not np.PyArray_IS_C_CONTIGUOUS(W):
        W = np.PyArray_GETCONTIGUOUS(W)
    if not np.PyArray_IS_C_CONTIGUOUS(u):
        u = np.PyArray_GETCONTIGUOUS(u)
    if not np.PyArray_IS_C_CONTIGUOUS(di):
        di = np.PyArray_GETCONTIGUOUS(di)
    if not np.PyArray_IS_C_CONTIGUOUS(dj):
        dj = np.PyArray_GETCONTIGUOUS(dj)

    cdef unsigned long *_dims = <unsigned long *>I_n.shape
    cdef double *_I = <double *>np.PyArray_DATA(I_n)
    cdef double *_W = <double *>np.PyArray_DATA(W)
    cdef double *_u = <double *>np.PyArray_DATA(u)
    cdef double *_di = <double *>np.PyArray_DATA(di)
    cdef double *_dj = <double *>np.PyArray_DATA(dj)
    cdef double *_I0
    cdef int _X0, _Y0
    cdef double _n0, _m0

    cdef int fail = 0
    with nogil:
        fail = make_reference_nfft(&_I0, &_X0, &_Y0, &_n0, &_m0,
                                   _I, _W, _u, _dims, _di, _dj, ls, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')

    cdef np.npy_intp *I0_dims = [_Y0, _X0]
    cdef np.ndarray I0_arr = ArrayWrapper.from_ptr(<void *>_I0).to_ndarray(2, I0_dims, np.NPY_FLOAT64)

    return I0_arr, _n0, _m0
        

def st_update(I_n, dij, basis, x_ps, y_ps, z, df, search_window, n_iter=5,
              filter=None, update_translations=False, verbose=False):
    """
    Andrew's speckle tracking update algorithm
    
    I_n - measured data
    W - whitefield
    basis - detector plane basis vectors
    x_ps, y_ps - x and y pixel sizes
    z - distance between the sample and the detector
    df - defocus distance
    wl - wavelength
    sw_max - pixel mapping search window size
    n_iter - number of iterations
    """
    M = np.ones((I_n.shape[1], I_n.shape[2]), dtype=bool)
    W = st.make_whitefield(I_n, M, verbose=verbose)
    u, dij_pix, res = st.generate_pixel_map(W.shape, dij, basis,
                                            x_ps, y_ps, z,
                                            df, verbose=verbose)
    I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=verbose)

    es = []
    for i in range(n_iter):

        # calculate errors
        error_total = st.calc_error(I_n, M, W, dij_pix, I0, u, n0, m0, subpixel=True, verbose=verbose)[0]

        # store total error
        es.append(error_total)

        # update pixel map
        u = st.update_pixel_map(I_n, M, W, I0, u, n0, m0, dij_pix,
                                search_window=search_window, subpixel=True,
                                fill_bad_pix=False, integrate=False,
                                quadratic_refinement=False, verbose=verbose,
                                filter=filter)[0]

        # make reference image
        I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=verbose)

        # update translations
        if update_translations:
            dij_pix = st.update_translations(I_n, M, W, I0, u, n0, m0, dij_pix)[0]

    return {'u':u, 'I0':I0, 'errors':es, 'n0': n0, 'm0': m0}
