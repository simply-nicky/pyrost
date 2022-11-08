#ifndef MEDIAN_H
#define MEDIAN_H
#include "include.h"
#include "array.h"

/*---------------------------------------------------------------------------
    struct footprint:
        ndim        - number of dimensions
        npts        - number of points in a footprint
        counter     - number of y values in buffer *data
        offsets     - coordinate offset relative to the central coordinate
        coordinates - buffer of the current list of coordinates
                      coordinates[i] = offsets[i] + current_coordinate
        item_size   - size of values in buffer *data
        data        - buffer of values
---------------------------------------------------------------------------*/

typedef struct footprint_s
{
    int ndim;
    int npts;
    int counter;
    int *offsets;
    int *coordinates;
    size_t item_size;
    void *data;
} footprint_s;
typedef struct footprint_s *footprint;

footprint init_footprint(int ndim, size_t item_size, size_t *fsize, unsigned char *fmask);
void free_footprint(footprint fpt);

void update_footprint(footprint fpt, int *coord, array arr, array mask, EXTEND_MODE mode, void *cval);

int median(void *out, void *data, unsigned char *mask, int ndim, const size_t *dims, size_t item_size,
    int axis, int (*compar)(const void*, const void*), unsigned threads);

/*-------------------------------------------------------------------------------*/
/** Calculate a multidimensional median filter.

    @param out          Buffer of output stack of images of shape dims.
    @param inp          Buffer of input stack of images of shape dims.
    @param mask         Output mask. Set out[...] to 0 if if mask[...] = 0.
    @param imask        Input mask. Omit inp[...] during the calculation of a median
                        if imask[...] = 0.
    @param ndim         Number of dimensions of inp and out.
    @param item_size    Size of a single element in bytes.
    @param fsize        Shape of filter footprint.
    @param fmask        Filter footprint. out[i, j, ...] = median(inp[ii, jj, ...] *
                        footprint[i - ii, j - jj, ...]).
    @param mode         The mode parameter determines how the input array is extended
                        when the filter overlaps a border. The valid values and their
                        behavior is as follows:

                        - EXTEND_CONSTAND:  (k k k k | a b c d | k k k k)
                        - EXTEND_NEAREST:   (a a a a | a b c d | d d d d)
                        - EXTEND_MIRROR:    (c d c b | a b c d | c b a b)
                        - EXTEND_REFLECT:   (d c b a | a b c d | d c b a)
                        - EXTEND_WRAP:      (a b c d | a b c d | a b c d)
    @param cval         Constant value to fill in the case of EXTEND_CONSTANT.
    @param compar       Comparing function compar(a, b). Returns positive number if a > b,
                        0 if a == b, and negative number if a < b.
    @param threads      Number of threads used during the calculation.

    @return             Returns 0 if it finished normally, 1 otherwise.
 */
int median_filter(void *out, void *inp, unsigned char *mask, unsigned char *imask, int ndim, const size_t *dims,
    size_t item_size, size_t *fsize, unsigned char *fmask, EXTEND_MODE mode, void *cval, int (*compar)(const void*, const void*),
    unsigned threads);

#endif