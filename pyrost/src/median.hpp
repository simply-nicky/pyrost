#ifndef MEDIAN_
#define MEDIAN_
#include "array.hpp"

namespace cbclib {

namespace detail{

template <typename InputIt1, typename InputIt2, typename InputIt3, typename OutputIt>
OutputIt mirror(InputIt1 first, InputIt1 last, OutputIt d_first, InputIt2 min, InputIt3 max)
{
    for (; first != last; ++first, ++d_first, ++min, ++max)
    {
        *d_first = mirror(*first, *min, *max);
    }
    return d_first;
}

template <typename InputIt1, typename InputIt2, typename InputIt3, typename OutputIt>
OutputIt reflect(InputIt1 first, InputIt1 last, OutputIt d_first, InputIt2 min, InputIt3 max)
{
    for (; first != last; ++first, ++d_first, ++min, ++max)
    {
        *d_first = reflect(*first, *min, *max);
    }
    return d_first;
}

template <typename InputIt1, typename InputIt2, typename InputIt3, typename OutputIt>
OutputIt wrap(InputIt1 first, InputIt1 last, OutputIt d_first, InputIt2 min, InputIt3 max)
{
    for (; first != last; ++first, ++d_first, ++min, ++max)
    {
        *d_first = wrap(*first, *min, *max);
    }
    return d_first;
}

}

template <typename Container, typename T, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
std::optional<T> extend_point(const Container & coord, const array<T> & arr, const array<bool> & mask, extend mode, const T & cval)
{
    using I = typename Container::value_type;

    /* kkkkkkkk|abcd|kkkkkkkk */
    if (mode == extend::constant) return std::optional<T>(cval);

    std::vector<I> close;
    std::vector<I> min (arr.ndim, I());

    switch (mode)
    {
        /* aaaaaaaa|abcd|dddddddd */
        case extend::nearest:

            for (size_t n = 0; n < arr.ndim; n++)
            {
                if (coord[n] >= static_cast<I>(arr.shape[n])) close.push_back(arr.shape[n] - 1);
                else if (coord[n] < I()) close.push_back(I());
                else close.push_back(coord[n]);
            }

            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case extend::mirror:

            detail::mirror(coord.begin(), coord.end(), std::back_inserter(close), min.begin(), arr.shape.begin());

            break;

        /* abcddcba|abcd|dcbaabcd */
        case extend::reflect:

            detail::reflect(coord.begin(), coord.end(), std::back_inserter(close), min.begin(), arr.shape.begin());

            break;

        /* abcdabcd|abcd|abcdabcd */
        case extend::wrap:

            detail::wrap(coord.begin(), coord.end(), std::back_inserter(close), min.begin(), arr.shape.begin());

            break;

        default:
            throw std::invalid_argument("Invalid extend argument: " + std::to_string(static_cast<int>(mode)));
    }

    size_t index = arr.ravel_index(close.begin(), close.end());

    if (mask[index]) return std::optional<T>(arr[index]);
    else return std::nullopt;
}

template <typename T>
struct footprint
{
    size_t ndim;
    size_t npts;
    std::vector<std::vector<long>> offsets;
    std::vector<std::vector<long>> coords;
    std::vector<T> data;

    footprint(size_t ndim, size_t npts, std::vector<std::vector<long>> offsets, std::vector<std::vector<long>> coords)
        : ndim(ndim), npts(npts), offsets(std::move(offsets)), coords(std::move(coords)) {}

    footprint(const array<bool> & fmask) : ndim(fmask.ndim)
    {
        auto fiter = fmask.begin();
        for (size_t i = 0; fiter != fmask.end(); ++fiter, ++i)
        {
            if (*fiter)
            {
                std::vector<long> coord;
                fmask.unravel_index(std::back_inserter(coord), i);
                auto & offset = this->offsets.emplace_back();
                std::transform(coord.begin(), coord.end(), fmask.shape.begin(), std::back_inserter(offset),
                               [](long crd, size_t dim){return crd - dim / 2;});
            }
        }

        this->npts = this->offsets.size();
        this->coords = std::vector<std::vector<long>>(npts, std::vector<long>(ndim));
        if (this->npts == 0) throw std::runtime_error("zero number of points in a footprint.");
    }

    template <typename Container, typename = std::enable_if_t<std::is_convertible_v<typename Container::value_type, long>>>
    footprint & update(const Container & coord, const array<T> & arr, const array<bool> & mask, extend mode, const T & cval)
    {
        this->data.clear();

        for (size_t i = 0; i < this->npts; i++)
        {
            bool extend = false;

            for (size_t n = 0; n < this->ndim; n++)
            {
                this->coords[i][n] = coord[n] + this->offsets[i][n];
                extend |= (this->coords[i][n] >= static_cast<long>(arr.shape[n])) || (this->coords[i][n] < 0);
            }

            if (extend)
            {
                auto val = extend_point(this->coords[i], arr, mask, mode, cval);
                if (val) this->data.push_back(val.value());
            }
            else
            {
                size_t index = arr.ravel_index(this->coords[i].begin(), this->coords[i].end());
                if (mask[index]) this->data.push_back(arr[index]);
            }
        }

        return *this;
    }
};

template <typename T, typename U>
py::array_t<T> median(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                      std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                      U axis, unsigned threads);

template <typename T, typename U>
py::array_t<T> median_filter(py::array_t<T, py::array::c_style | py::array::forcecast> inp, std::optional<U> size,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> footprint,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> inp_mask,
                             std::string mode, const T & cval, unsigned threads);

template <typename T, typename U>
auto robust_mean(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                 std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                 U axis, double r0, double r1, int n_iter, double lm, unsigned threads) -> py::array_t<std::common_type_t<T, float>>;

}

#endif