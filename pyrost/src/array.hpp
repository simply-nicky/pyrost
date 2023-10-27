#ifndef ARRAY_
#define ARRAY_
#include "include.hpp"

namespace cbclib {

template <class InputIt>
size_t get_size(InputIt first, InputIt last)
{
    return std::reduce(first, last, size_t(1), std::multiplies<size_t>());
}

namespace detail{

template <typename T, typename U>
constexpr auto modulo(T a, U b) -> decltype(a % b)
{
    return (a % b + b) % b;
}

template <typename T, typename U>
constexpr auto remainder(T a, U b) -> decltype(modulo(a, b))
{
    return (a - modulo(a, b)) / b;
}

template <typename T, typename U, typename V>
constexpr std::make_signed_t<T> mirror(T a, U min, V max)
{
    using F = std::make_signed_t<T>;
    F val = std::minus<F>()(a, min);
    F period = std::minus<F>()(max, min) - 1;
    if (modulo(remainder(val, period), 2)) return period - modulo(val, period) + min;
    else return modulo(val, period) + min;
}

template <typename T, typename U, typename V>
constexpr std::make_signed_t<T> reflect(T a, U min, V max)
{
    using F = std::make_signed_t<T>;
    F val = std::minus<F>()(a, min);
    F period = std::minus<F>()(max, min);
    if (modulo(remainder(val, period), 2)) return period - 1 - modulo(val, period) + min;
    else return modulo(val, period) + min;
}

template <typename T, typename U, typename V>
constexpr std::make_signed_t<T> wrap(T a, U min, V max)
{
    using F = std::make_signed_t<T>;
    F val = std::minus<F>()(a, min);
    F period = std::minus<F>()(max, min);
    return modulo(val, period) + min;
}

template <typename InputIt1, typename InputIt2>
auto ravel_index_impl(InputIt1 cfirst, InputIt1 clast, InputIt2 sfirst)
{
    using value_t = decltype(+*std::declval<InputIt1 &>());
    value_t index = value_t();
    for(; cfirst != clast; cfirst++, ++sfirst) index += *cfirst * *sfirst;
    return index;
}

template <typename InputIt, typename OutputIt, typename T>
OutputIt unravel_index_impl(InputIt sfirst, InputIt slast, T index, OutputIt cfirst)
{
    for(; sfirst != slast; ++sfirst)
    {
        auto stride = index / *sfirst;
        index -= stride * *sfirst;
        *cfirst++ = stride;
    }
    return cfirst;
}

class shape_handler
{
public:
    size_t ndim;
    size_t size;
    std::vector<size_t> shape;

    using ShapeContainer = detail::any_container<size_t>;

    shape_handler(size_t ndim, size_t size, ShapeContainer shape, ShapeContainer strides)
        : ndim(ndim), size(size), shape(std::move(shape)), strides(std::move(strides)) {}

    shape_handler(ShapeContainer shape) : ndim(std::distance(shape->begin(), shape->end()))
    {
        this->size = get_size(shape->begin(), shape->end());
        size_t stride = this->size;
        for (auto length : *shape)
        {
            stride /= length;
            this->strides.push_back(stride);
            this->shape.push_back(length);
        }
    }

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator<CoordIter>::value>>
    bool is_inbound(CoordIter first, CoordIter last) const
    {
        bool flag = true;
        for (size_t i = 0; first != last; ++first, ++i)
        {
            flag &= *first >= 0 && *first < static_cast<decltype(+*std::declval<CoordIter &>())>(this->shape[i]);
        }
        return flag;
    }

    template <typename Container>
    bool is_inbound(const Container & coord) const
    {
        return is_inbound(coord.begin(), coord.end());
    }

    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    bool is_inbound(const std::initializer_list<T> & coord) const
    {
        return is_inbound(coord.begin(), coord.end());
    }

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator<CoordIter>::value>>
    auto ravel_index(CoordIter first, CoordIter last) const
    {
        return ravel_index_impl(first, last, this->strides.begin());
    }

    template <typename Container>
    auto ravel_index(const Container & coord) const
    {
        return ravel_index_impl(coord.begin(), coord.end(), this->strides.begin());
    }

    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    auto ravel_index(const std::initializer_list<T> & coord) const
    {
        return ravel_index_impl(coord.begin(), coord.end(), this->strides.begin());
    }

    template <
        typename CoordIter,
        typename = std::enable_if_t<
            std::is_integral_v<typename CoordIter::value_type> || 
            std::is_same_v<typename CoordIter::iterator_category, std::output_iterator_tag>
        >
    >
    CoordIter unravel_index(CoordIter first, size_t index) const
    {
        return unravel_index_impl(this->strides.begin(), this->strides.end(), index, first);
    }

protected:
    std::vector<size_t> strides;
};

}

template <typename T>
class strided_iterator
{
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

    strided_iterator(T * ptr, size_t stride) : ptr(ptr), stride(stride) {}

    operator bool() const
    {
        if (this->ptr) return true;
        return false;
    }

    bool operator==(const strided_iterator<T> & iter) const {return this->ptr == iter.ptr;}
    bool operator!=(const strided_iterator<T> & iter) const {return this->ptr != iter.ptr;}
    bool operator<=(const strided_iterator<T> & iter) const {return this->ptr <= iter.ptr;}
    bool operator>=(const strided_iterator<T> & iter) const {return this->ptr >= iter.ptr;}
    bool operator<(const strided_iterator<T> & iter) const {return this->ptr < iter.ptr;}
    bool operator>(const strided_iterator<T> & iter) const {return this->ptr > iter.ptr;}

    strided_iterator<T> & operator+=(const difference_type & step) {this->ptr += step * this->stride; return *this;}
    strided_iterator<T> & operator-=(const difference_type & step) {this->ptr -= step * this->stride; return *this;}
    strided_iterator<T> & operator++() {this->ptr += this->stride; return *this;}
    strided_iterator<T> & operator--() {this->ptr -= this->stride; return *this;}
    strided_iterator<T> operator++(int) {strided_iterator<T> temp = *this; ++(*this); return temp;}
    strided_iterator<T> operator--(int) {strided_iterator<T> temp = *this; --(*this); return temp;}
    strided_iterator<T> operator+(const difference_type & step) const
    {
        return strided_iterator<T>(this->ptr + step * this->stride, this->stride);
    }
    strided_iterator<T> operator-(const difference_type & step) const
    {
        return strided_iterator<T>(this->ptr - step * this->stride, this->stride);
    }

    difference_type operator-(const strided_iterator<T> & iter) const {return (this->ptr - iter.ptr) / this->stride;}

    T & operator[] (size_t index) const {return this->ptr[index * this->stride];}
    T & operator*() const {return *(this->ptr);}
    T * operator->() const {return this->ptr;}
    
private:
    T * ptr;
    size_t stride;
};

template <typename T>
class array : public detail::shape_handler
{
public:
    T * ptr;

    using iterator = T *;
    using const_iterator = const T *;

    array(size_t ndim, size_t size, ShapeContainer shape, ShapeContainer strides, T * ptr)
        : shape_handler(ndim, size, std::move(shape), std::move(strides)), ptr(ptr) {}

    array(shape_handler handler, T * ptr) : shape_handler(std::move(handler)), ptr(ptr) {}

    array(ShapeContainer shape, T * ptr) : shape_handler(std::move(shape)), ptr(ptr) {}

    array(const py::buffer_info & buf) : array(buf.shape, static_cast<T *>(buf.ptr)) {}

    T & operator[] (size_t index) {return this->ptr[index];}
    const T & operator[] (size_t index) const {return this->ptr[index];}
    iterator begin() {return this->ptr;}
    iterator end() {return this->ptr + this->size;}
    const_iterator begin() const {return this->ptr;}
    const_iterator end() const {return this->ptr + this->size;}

    array<T> slice(size_t index, ShapeContainer axes) const
    {
        std::sort(axes->begin(), axes->end());

        std::vector<size_t> other_shape, shape, strides;
        for (size_t i = 0; i < this->ndim; i++)
        {
            if (std::find(axes->begin(), axes->end(), i) == axes->end()) other_shape.push_back(this->shape[i]);
        }
        std::transform(axes->begin(), axes->end(), std::back_inserter(shape), [this](size_t axis){return this->shape[axis];});
        std::transform(axes->begin(), axes->end(), std::back_inserter(strides), [this](size_t axis){return this->strides[axis];});

        std::vector<size_t> coord;
        shape_handler(std::move(other_shape)).unravel_index(std::back_inserter(coord), index);
        for (auto axis : *axes) coord.insert(std::next(coord.begin(), axis), 0);

        index = this->ravel_index(coord.begin(), coord.end());

        auto ndim = shape.size();
        auto size = get_size(shape.begin(), shape.end());

        return array<T>(ndim, size, std::move(shape), std::move(strides), this->ptr + index);
    }

    strided_iterator<T> line_begin(size_t axis, size_t index)
    {
        check_index(axis, index);
        size_t size = this->shape[axis] * this->strides[axis];
        iterator ptr = this->ptr + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<T>(ptr, this->strides[axis]);
    }

    strided_iterator<const T> line_begin(size_t axis, size_t index) const
    {
        check_index(axis, index);
        size_t size = this->shape[axis] * this->strides[axis];
        const_iterator ptr = this->ptr + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<const T>(ptr, this->strides[axis]);
    }

    strided_iterator<T> line_end(size_t axis, size_t index)
    {
        check_index(axis, index);
        size_t size = this->shape[axis] * this->strides[axis];
        iterator ptr = this->ptr + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<T>(ptr + size, this->strides[axis]);
    }

    strided_iterator<const T> line_end(size_t axis, size_t index) const
    {
        check_index(axis, index);
        size_t size = this->shape[axis] * this->strides[axis];
        const_iterator ptr = this->ptr + size * (index / this->strides[axis]) + index % this->strides[axis];
        return strided_iterator<const T>(ptr + size, this->strides[axis]);
    }

protected:
    void check_index(size_t axis, size_t index) const
    {
        if (axis >= this->ndim || index >= (this->size / this->shape[axis]))
            throw std::out_of_range("index " + std::to_string(index) + " is out of bound for axis "
                                    + std::to_string(axis));
    }
};

template <typename T>
class vector_array : public array<T>
{
private:
    std::vector<T> buffer;

public:
    vector_array(typename array<T>::ShapeContainer shape) : array<T>(std::move(shape), nullptr)
    {
        this->buffer = std::vector<T>(this->size, T());
        this->ptr = this->buffer.data();
    }
};

/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangular iterator ---------------------------*/
/*----------------------------------------------------------------------------*/

class rect_iterator : public detail::shape_handler
{
public:
    std::vector<size_t> coord;
    size_t index;

    rect_iterator(ShapeContainer shape) : shape_handler(std::move(shape)), index(0)
    {
        this->unravel_index(std::back_inserter(this->coord), this->index);
    }

    rect_iterator & operator++()
    {
        this->index++;
        this->unravel_index(this->coord.begin(), this->index);
        return *this;
    }

    rect_iterator operator++(int)
    {
        rect_iterator temp = *this;
        this->index++;
        this->unravel_index(this->coord.begin(), this->index);
        return temp;
    }

    bool is_end() const {return this->index >= this->size; }
};

/*----------------------------------------------------------------------------*/
/*------------------------------ Binary search -------------------------------*/
/*----------------------------------------------------------------------------*/
// Array search
enum class side
{
    left = 0,
    right = 1
};

/* find idx \el [0, npts], so that base[idx - 1] < key <= base[idx] */
template <class ForwardIt, typename T, class Compare>
ForwardIt searchsorted(const T & value, ForwardIt first, ForwardIt last, side s, Compare comp)
{
    auto npts = std::distance(first, last);
    auto extreme = std::next(first, npts - 1);
    if (comp(value, *first)) return first;
    if (!comp(value, *extreme)) return extreme;

    ForwardIt out;
    switch (s)
    {
        case side::left:
            out = std::lower_bound(first, last, value, comp);
            break;

        case side::right:
            out = std::next(first, std::distance(first, std::upper_bound(first, last, value, comp)) - 1);
            break;

        default:
            throw std::invalid_argument("searchsorted: invalid side argument.");
    }
    return out;
}

/*----------------------------------------------------------------------------*/
/*------------------------------- Wirth select -------------------------------*/
/*----------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------
    Function :  kth_smallest()
    In       :  array of elements, n elements in the array, rank k 
    Out      :  one element
    Job      :  find the kth smallest element in the array
    Notice   :  Buffer must be of size n

    Reference:
        Author: Wirth, Niklaus
        Title: Algorithms + data structures = programs
        Publisher: Englewood Cliffs: Prentice-Hall, 1976 Physical description: 366 p.
        Series: Prentice-Hall Series in Automatic Computation
---------------------------------------------------------------------------*/
template <class RandomIt, class Compare>
RandomIt wirthselect(RandomIt first, RandomIt last, typename std::iterator_traits<RandomIt>::difference_type k, Compare comp)
{
    auto l = first;
    auto m = std::prev(last);
    auto key = std::next(first, k);
    while (l < m)
    {
        auto value = *key;
        auto i = l;
        auto j = m;

        do
        {
            while (comp(*i, value)) i++;
            while (comp(value, *j)) j--;
            if (i <= j) std::swap(*(i++), *(j--));
        } while (i <= j);
        if (j < key) l = i;
        if (key < i) m = j;
    }
    
    return key;
}

template <class RandomIt, class Compare>
RandomIt wirthmedian(RandomIt first, RandomIt last, Compare comp)
{
    auto n = std::distance(first, last);
    return wirthselect(first, last, (n & 1) ? n / 2 : n / 2 - 1, comp);
}

/*----------------------------------------------------------------------------*/
/*--------------------------- Extend line modes ------------------------------*/
/*----------------------------------------------------------------------------*/
/*
    constant: kkkkkkkk|abcd|kkkkkkkk
    nearest:  aaaaaaaa|abcd|dddddddd
    mirror:   cbabcdcb|abcd|cbabcdcb
    reflect:  abcddcba|abcd|dcbaabcd
    wrap:     abcdabcd|abcd|abcdabcd
*/
enum class extend
{
    constant = 0,
    nearest = 1,
    mirror = 2,
    reflect = 3,
    wrap = 4
};

static std::unordered_map<std::string, extend> const modes = {{"constant", extend::constant},
                                                              {"nearest", extend::nearest},
                                                              {"mirror", extend::mirror},
                                                              {"reflect", extend::reflect},
                                                              {"wrap", extend::wrap}};

}

#endif