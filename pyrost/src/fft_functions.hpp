#ifndef FFT_FUNCTIONS_
#define FFT_FUNCTIONS_
#include "array.hpp"

namespace cbclib {

struct fft
{
    enum
    {
        forward = FFTW_FORWARD,
        backward = FFTW_BACKWARD
    };

    enum
    {
        estimate = FFTW_ESTIMATE,
        measure = FFTW_MEASURE,
        patient = FFTW_PATIENT,
        exhaustive = FFTW_EXHAUSTIVE
    };
};

namespace detail {

// Plan makers

struct fftwf_plan_deleter
{
    void operator()(fftwf_plan plan) const
    {
        if (plan)
        {
            fftwf_destroy_plan(plan);
        }
    }
};

using fftwf_plan_ptr = std::unique_ptr<fftwf_plan_s, fftwf_plan_deleter>;

fftwf_plan_ptr make_fftw_forward(int rank, const int *ns, std::complex<float> * inp, std::complex<float> * out, int flags)
{
    auto plan = fftwf_plan_dft(rank, ns, reinterpret_cast<fftwf_complex *>(inp), reinterpret_cast<fftwf_complex *>(out), fft::forward, flags);
    return std::unique_ptr<fftwf_plan_s, fftwf_plan_deleter>(plan);
}

fftwf_plan_ptr make_fftw_backward(int rank, const int *ns, std::complex<float> * inp, std::complex<float> * out, int flags)
{
    auto plan = fftwf_plan_dft(rank, ns, reinterpret_cast<fftwf_complex *>(inp), reinterpret_cast<fftwf_complex *>(out), fft::backward, flags);
    return std::unique_ptr<fftwf_plan_s, fftwf_plan_deleter>(plan);
}

fftwf_plan_ptr make_fftw_forward(int rank, const int *ns, float * inp, std::complex<float> * out, int flags)
{
    auto plan = fftwf_plan_dft_r2c(rank, ns, inp, reinterpret_cast<fftwf_complex *>(out), flags);
    return std::unique_ptr<fftwf_plan_s, fftwf_plan_deleter>(plan);
}

fftwf_plan_ptr make_fftw_backward(int rank, const int *ns, std::complex<float> * inp, float * out, int flags)
{
    auto plan = fftwf_plan_dft_c2r(rank, ns, reinterpret_cast<fftwf_complex *>(inp), out, flags);
    return std::unique_ptr<fftwf_plan_s, fftwf_plan_deleter>(plan);
}

struct fftw_plan_deleter
{
    void operator()(fftw_plan plan) const
    {
        if (plan)
        {
            fftw_destroy_plan(plan);
        }
    }
};

using fftw_plan_ptr = std::unique_ptr<fftw_plan_s, fftw_plan_deleter>;

fftw_plan_ptr make_fftw_forward(int rank, const int *ns, std::complex<double> * inp, std::complex<double> * out, int flags)
{
    auto plan = fftw_plan_dft(rank, ns, reinterpret_cast<fftw_complex *>(inp), reinterpret_cast<fftw_complex *>(out), fft::forward, flags);
    return std::unique_ptr<fftw_plan_s, fftw_plan_deleter>(plan);
}

fftw_plan_ptr make_fftw_backward(int rank, const int *ns, std::complex<double> * inp, std::complex<double> * out, int flags)
{
    auto plan = fftw_plan_dft(rank, ns, reinterpret_cast<fftw_complex *>(inp), reinterpret_cast<fftw_complex *>(out), fft::backward, flags);
    return std::unique_ptr<fftw_plan_s, fftw_plan_deleter>(plan);
}

fftw_plan_ptr make_fftw_forward(int rank, const int *ns, double * inp, std::complex<double> * out, int flags)
{
    auto plan = fftw_plan_dft_r2c(rank, ns, inp, reinterpret_cast<fftw_complex *>(out), flags);
    return std::unique_ptr<fftw_plan_s, fftw_plan_deleter>(plan);
}

fftw_plan_ptr make_fftw_backward(int rank, const int *ns, std::complex<double> * inp, double * out, int flags)
{
    auto plan = fftw_plan_dft_c2r(rank, ns, reinterpret_cast<fftw_complex *>(inp), out, flags);
    return std::unique_ptr<fftw_plan_s, fftw_plan_deleter>(plan);
}

struct fftwl_plan_deleter
{
    void operator()(fftwl_plan plan) const
    {
        if (plan)
        {
            fftwl_destroy_plan(plan);
        }
    }
};

using fftwl_plan_ptr = std::unique_ptr<fftwl_plan_s, fftwl_plan_deleter>;

fftwl_plan_ptr make_fftw_forward(int rank, const int *ns, std::complex<long double> * inp, std::complex<long double> * out, int flags)
{
    auto plan = fftwl_plan_dft(rank, ns, reinterpret_cast<fftwl_complex *>(inp), reinterpret_cast<fftwl_complex *>(out), fft::forward, flags);
    return std::unique_ptr<fftwl_plan_s, fftwl_plan_deleter>(plan);
}

fftwl_plan_ptr make_fftw_backward(int rank, const int *ns, std::complex<long double> * inp, std::complex<long double> * out, int flags)
{
    auto plan = fftwl_plan_dft(rank, ns, reinterpret_cast<fftwl_complex *>(inp), reinterpret_cast<fftwl_complex *>(out), fft::backward, flags);
    return std::unique_ptr<fftwl_plan_s, fftwl_plan_deleter>(plan);
}

fftwl_plan_ptr make_fftw_forward(int rank, const int *ns, long double * inp, std::complex<long double> * out, int flags)
{
    auto plan = fftwl_plan_dft_r2c(rank, ns, inp, reinterpret_cast<fftwl_complex *>(out), flags);
    return std::unique_ptr<fftwl_plan_s, fftwl_plan_deleter>(plan);
}

fftwl_plan_ptr make_fftw_backward(int rank, const int *ns, std::complex<long double> * inp, long double * out, int flags)
{
    auto plan = fftwl_plan_dft_c2r(rank, ns, reinterpret_cast<fftwl_complex *>(inp), out, flags);
    return std::unique_ptr<fftwl_plan_s, fftwl_plan_deleter>(plan);
}

// Execute wrappers

void fftw_execute_impl(fftwf_plan_ptr & plan, std::complex<float> * inp, std::complex<float> * out)
{
    fftwf_execute_dft(plan.get(), reinterpret_cast<fftwf_complex *>(inp), reinterpret_cast<fftwf_complex *>(out));
}

void fftw_execute_impl(fftw_plan_ptr & plan, std::complex<double> * inp, std::complex<double> * out)
{
    fftw_execute_dft(plan.get(), reinterpret_cast<fftw_complex *>(inp), reinterpret_cast<fftw_complex *>(out));
}

void fftw_execute_impl(fftwl_plan_ptr & plan, std::complex<long double> * inp, std::complex<long double> * out)
{
    fftwl_execute_dft(plan.get(), reinterpret_cast<fftwl_complex *>(inp), reinterpret_cast<fftwl_complex *>(out));
}

void fftw_execute_impl(fftwf_plan_ptr & plan, float * inp, std::complex<float> * out)
{
    fftwf_execute_dft_r2c(plan.get(), inp, reinterpret_cast<fftwf_complex *>(out));
}

void fftw_execute_impl(fftw_plan_ptr & plan, double * inp, std::complex<double> * out)
{
    fftw_execute_dft_r2c(plan.get(), inp, reinterpret_cast<fftw_complex *>(out));
}

void fftw_execute_impl(fftwl_plan_ptr & plan, long double * inp, std::complex<long double> * out)
{
    fftwl_execute_dft_r2c(plan.get(), inp, reinterpret_cast<fftwl_complex *>(out));
}

void fftw_execute_impl(fftwf_plan_ptr & plan, std::complex<float> * inp, float * out)
{
    fftwf_execute_dft_c2r(plan.get(), reinterpret_cast<fftwf_complex *>(inp), out);
}

void fftw_execute_impl(fftw_plan_ptr & plan, std::complex<double> * inp, double * out)
{
    fftw_execute_dft_c2r(plan.get(), reinterpret_cast<fftw_complex *>(inp), out);
}

void fftw_execute_impl(fftwl_plan_ptr & plan, std::complex<long double> * inp, long double * out)
{
    fftwl_execute_dft_c2r(plan.get(), reinterpret_cast<fftwl_complex *>(inp), out);
}

static const std::array<size_t, 585> LPRE =
{
    18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 50, 52, 54, 55, 56, 60, 63, 64,
    65, 66, 70, 72, 75, 77, 78, 80, 81, 84, 88, 90, 91, 96, 98, 99, 100, 104, 105, 108, 110, 112, 117, 120, 125, 126, 128,
    130, 132, 135, 140, 144, 147, 150, 154, 156, 160, 162, 165, 168, 175, 176, 180, 182, 189, 192, 195, 196, 198, 200, 208,
    210, 216, 220, 224, 225, 231, 234, 240, 243, 245, 250, 252, 256, 260, 264, 270, 273, 275, 280, 288, 294, 297, 300, 308,
    312, 315, 320, 324, 325, 330, 336, 343, 350, 351, 352, 360, 364, 375, 378, 384, 385, 390, 392, 396, 400, 405, 416, 420,
    432, 440, 441, 448, 450, 455, 462, 468, 480, 486, 490, 495, 500, 504, 512, 520, 525, 528, 539, 540, 546, 550, 560, 567,
    576, 585, 588, 594, 600, 616, 624, 625, 630, 637, 640, 648, 650, 660, 672, 675, 686, 693, 700, 702, 704, 720, 728, 729,
    735, 750, 756, 768, 770, 780, 784, 792, 800, 810, 819, 825, 832, 840, 864, 875, 880, 882, 891, 896, 900, 910, 924, 936,
    945, 960, 972, 975, 980, 990, 1000, 1008, 1024, 1029, 1040, 1050, 1053, 1056, 1078, 1080, 1092, 1100, 1120, 1125, 1134,
    1152, 1155, 1170, 1176, 1188, 1200, 1215, 1225, 1232, 1248, 1250, 1260, 1274, 1280, 1296, 1300, 1320, 1323, 1344, 1350,
    1365, 1372, 1375, 1386, 1400, 1404, 1408, 1440, 1456, 1458, 1470, 1485, 1500, 1512, 1536, 1540, 1560, 1568, 1575, 1584,
    1600, 1617, 1620, 1625, 1638, 1650, 1664, 1680, 1701, 1715, 1728, 1750, 1755, 1760, 1764, 1782, 1792, 1800, 1820, 1848,
    1872, 1875, 1890, 1911, 1920, 1925, 1944, 1950, 1960, 1980, 2000, 2016, 2025, 2048, 2058, 2079, 2080, 2100, 2106, 2112,
    2156, 2160, 2184, 2187, 2200, 2205, 2240, 2250, 2268, 2275, 2304, 2310, 2340, 2352, 2376, 2400, 2401, 2430, 2450, 2457,
    2464, 2475, 2496, 2500, 2520, 2548, 2560, 2592, 2600, 2625, 2640, 2646, 2673, 2688, 2695, 2700, 2730, 2744, 2750, 2772,
    2800, 2808, 2816, 2835, 2880, 2912, 2916, 2925, 2940, 2970, 3000, 3024, 3072, 3080, 3087, 3120, 3125, 3136, 3150, 3159,
    3168, 3185, 3200, 3234, 3240, 3250, 3276, 3300, 3328, 3360, 3375, 3402, 3430, 3456, 3465, 3500, 3510, 3520, 3528, 3564,
    3584, 3600, 3640, 3645, 3675, 3696, 3744, 3750, 3773, 3780, 3822, 3840, 3850, 3888, 3900, 3920, 3960, 3969, 4000, 4032,
    4050, 4095, 4096, 4116, 4125, 4158, 4160, 4200, 4212, 4224, 4312, 4320, 4368, 4374, 4375, 4400, 4410, 4455, 4459, 4480,
    4500, 4536, 4550, 4608, 4620, 4680, 4704, 4725, 4752, 4800, 4802, 4851, 4860, 4875, 4900, 4914, 4928, 4950, 4992, 5000,
    5040, 5096, 5103, 5120, 5145, 5184, 5200, 5250, 5265, 5280, 5292, 5346, 5376, 5390, 5400, 5460, 5488, 5500, 5544, 5600,
    5616, 5625, 5632, 5670, 5733, 5760, 5775, 5824, 5832, 5850, 5880, 5940, 6000, 6048, 6075, 6125, 6144, 6160, 6174, 6237,
    6240, 6250, 6272, 6300, 6318, 6336, 6370, 6400, 6468, 6480, 6500, 6552, 6561, 6600, 6615, 6656, 6720, 6750, 6804, 6825,
    6860, 6875, 6912, 6930, 7000, 7020, 7040, 7056, 7128, 7168, 7200, 7203, 7280, 7290, 7350, 7371, 7392, 7425, 7488, 7500,
    7546, 7560, 7644, 7680, 7700, 7776, 7800, 7840, 7875, 7920, 7938, 8000, 8019, 8064, 8085, 8100, 8125, 8190, 8192, 8232,
    8250, 8316, 8320, 8400, 8424, 8448, 8505, 8575, 8624, 8640, 8736, 8748, 8750, 8775, 8800, 8820, 8910, 8918, 8960, 9000,
    9072, 9100, 9216, 9240, 9261, 9360, 9375, 9408, 9450, 9477, 9504, 9555, 9600, 9604, 9625, 9702, 9720, 9750, 9800, 9828,
    9856, 9900, 9984, 10000
};

size_t find_match(size_t target, size_t p7_11_13)
{
    size_t p5_7_11_13, p3_5_7_11_13;
    size_t match = 2 * target;
    while (p7_11_13 < target)
    {
        p5_7_11_13 = p7_11_13;
        while (p5_7_11_13 < target)
        {
            p3_5_7_11_13 = p5_7_11_13;
            while (p3_5_7_11_13 < target)
            {
                while (p3_5_7_11_13 < target) p3_5_7_11_13 *= 2;

                if (p3_5_7_11_13 == target) return p3_5_7_11_13;
                if (p3_5_7_11_13 < match) match = p3_5_7_11_13;

                while (!(p3_5_7_11_13 & 1)) p3_5_7_11_13 >>= 1;

                p3_5_7_11_13 *= 3;
                if (p3_5_7_11_13 == target) return p3_5_7_11_13;
            }
            if (p3_5_7_11_13 < match) match = p3_5_7_11_13;

            p5_7_11_13 *= 5;
            if (p5_7_11_13 == target) return p5_7_11_13;
        }
        if (p5_7_11_13 < match) match = p5_7_11_13;

        p7_11_13 *= 7;
        if (p7_11_13 == target) return p7_11_13;
    }
    if (p7_11_13 < match) return p7_11_13;
    return match;
}

std::vector<size_t> fftw_buffer_shape(std::vector<size_t> && shape, std::true_type)
{
    return std::move(shape);
}

std::vector<size_t> fftw_buffer_shape(std::vector<size_t> && shape, std::false_type)
{
    shape[shape.size() - 1] = 2 * (shape[shape.size() - 1] / 2 + 1);
    return std::move(shape);
}

template <typename OutputIt, typename T>
OutputIt gaussian_zero(OutputIt first, size_t size, T sigma)
{
    T sum = T();
    auto radius = (size - 1) / 2;
    for (size_t i = 0; i < size; ++i)
    {
        auto gauss = std::exp(-std::pow(std::minus<long>()(i, radius), 2) / (2 * sigma * sigma));
        sum += gauss;
    }

    for (size_t i = 0; i < size; ++i, ++first)
    {
        auto gauss = std::exp(-std::pow(std::minus<long>()(i, radius), 2) / (2 * sigma * sigma));
        *first = gauss / sum;
    }

    return first;
}

template <typename OutputIt, typename T>
OutputIt gaussian_order(OutputIt first, size_t size, T sigma, unsigned order)
{
    std::vector<T> q0 (order + 1, T());
    q0[0] = T(1.0);
    std::vector<T> q1 (order + 1, T());

    for (size_t k = 0; k < order; k++)
    {
        for (size_t i = 0; i <= order; i++)
        {
            T qval = T();
            for (size_t j = 0; j <= order; j++)
            {
                auto idx = j + (order + 1) * i;
                if ((idx % (order + 2)) == 1) qval += q0[j] * (idx / (order + 2) + 1);
                if ((idx % (order + 2)) == (order + 1)) qval -= q0[j] / (sigma * sigma); 
            }
            q1[i] = qval;
        }
        std::copy(q1.begin(), q1.end(), q0.begin());
    }

    T sum = T();
    auto radius = (size - 1) / 2;
    for (size_t i = 0; i < size; ++i)
    {
        sum += std::exp(-std::pow(std::minus<long>()(i, radius), 2) / (2 * sigma * sigma));
    }

    for (size_t i = 0; i < size; ++i, ++first)
    {
        auto gauss = std::exp(-std::pow(std::minus<long>()(i, radius), 2) / (2 * sigma * sigma));

        T factor = T();
        for (size_t j = 0; j <= order; j++) factor += std::pow(std::minus<long>()(i, radius), j) * q1[j];
        *first = factor * gauss / sum;
    }

    return first;
}

}

template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
struct remove_complex
{
    using type = T;
};

template <typename T>
struct remove_complex<std::complex<T>>
{
    using type = T;
};

template <typename T>
using remove_complex_t = typename remove_complex<T>::type;

template <
    class Container, typename From, typename To,
    typename = std::enable_if_t<std::is_convertible_v<typename Container::value_type, int>>
>
auto make_forward_plan(const Container & shape, From * inp, To * out, int flags = fft::estimate)
{
    int rank = shape.size();
    auto ns = std::vector<int>(shape.begin(), shape.end());
    return detail::make_fftw_forward(rank, ns.data(), inp, out, flags);
}

template <
    class Container, typename From, typename To,
    typename = std::enable_if_t<std::is_convertible_v<typename Container::value_type, int>>
>
auto make_backward_plan(const Container & shape, From * inp, To * out, int flags = fft::estimate)
{
    int rank = shape.size();
    auto ns = std::vector<int>(shape.begin(), shape.end());
    return detail::make_fftw_backward(rank, ns.data(), inp, out, flags);
}

template <typename Plan, typename From, typename To>
void fftw_execute(Plan & plan, From * inp, To * out)
{
    detail::fftw_execute_impl(plan, inp, out);
}

template <typename T>
std::vector<size_t> fftw_buffer_shape(typename detail::any_container<size_t> shape)
{
    return detail::fftw_buffer_shape(std::move(shape), typename is_complex<T>::type ());
}

template <typename T, typename U, class Container>
void write_buffer(array<T> & buffer, const Container & fshape, array<U> data)
{
    auto find_origin = [](size_t flen, size_t n)
    {
        return std::minus<long>()(flen, n) - std::minus<long>()(flen, n) / 2;
    };

    std::vector<long> origin;
    std::transform(fshape.begin(), fshape.end(), data.shape.begin(), std::back_inserter(origin), find_origin);

    std::vector<long> coord (buffer.ndim, 0);
    for (auto riter = rect_iterator(fshape); !riter.is_end(); ++riter)
    {
        std::transform(riter.coord.begin(), riter.coord.end(), origin.begin(), coord.begin(), std::minus<long>());
        auto bindex = buffer.ravel_index(riter.coord.begin(), riter.coord.end());

        if (data.is_inbound(coord.begin(), coord.end()))
        {
            auto index = data.ravel_index(coord.begin(), coord.end());
            buffer[bindex] = data[index];
        }
        else buffer[bindex] = T();
    }
}

template <typename T, typename U, class Container>
void read_buffer(const array<T> & buffer, const Container & fshape, array<U> data)
{
    auto find_origin = [](size_t flen, size_t n){return std::minus<long>()(flen, n / 2);};
    std::vector<long> origin;
    std::transform(fshape.begin(), fshape.end(), data.shape.begin(), std::back_inserter(origin), find_origin);

    std::vector<long> coord (buffer.ndim, 0);
    for (auto riter = rect_iterator(data.shape); !riter.is_end(); ++riter)
    {
        std::transform(origin.begin(), origin.end(), riter.coord.begin(), coord.begin(), std::plus<long>());
        std::transform(coord.begin(), coord.end(), fshape.begin(), coord.begin(), detail::modulo<long, size_t>);

        auto index = buffer.ravel_index(coord.begin(), coord.end());
        data[riter.index] = buffer[index];
    }
}

template <class OutputIt, typename T>
OutputIt gauss_kernel(OutputIt first, size_t size, T sigma, unsigned order)
{
    if (order) return detail::gaussian_order(first, size, sigma, order);
    return detail::gaussian_zero(first, size, sigma);
}

template <typename T, typename InputIt>
void write_line(std::vector<T> & buffer, size_t flen, InputIt first, InputIt last, extend mode)
{
    auto n = std::distance(first, last);
    auto origin = std::minus<long>()(flen, n) - std::minus<long>()(flen, n) / 2;

    for (size_t i = 0; i < buffer.size(); ++i)
    {
        auto idx = std::minus<long>()(i, origin);
        if (idx < 0 || idx >= n)
        {
            switch (mode)
            {
                case extend::constant:

                    buffer[i] = T();
                    break;
                
                case extend::nearest:

                    buffer[i] = (idx < 0) ? *first : *std::prev(last);
                    break;

                case extend::mirror:

                    buffer[i] = *std::next(first, detail::mirror(idx, 0, n));
                    break;

                case extend::reflect:

                    buffer[i] = *std::next(first, detail::reflect(idx, 0, n));
                    break;

                case extend::wrap:

                    buffer[i] = *std::next(first, detail::wrap(idx, 0, n));
                    break;

                default:
                    throw std::invalid_argument("Invalid extend argument: " + std::to_string(static_cast<int>(mode)));
            }
        }
        else
        {
            buffer[i] = *std::next(first, idx);
        }
    }
}

template <typename T, typename OutputIt>
void read_line(const std::vector<T> & buffer, size_t flen, OutputIt first, OutputIt last)
{
    auto n = std::distance(first, last);
    auto origin = std::minus<long>()(flen, n / 2);
    for (size_t i = 0; first != last; ++first, ++i)
    {
        *first = buffer[detail::modulo(std::plus<long>()(i, origin), flen)];
    }
}

size_t next_fast_len(size_t target);

template <typename Inp, typename Krn, typename Seq>
auto fft_convolve(py::array_t<Inp> inp, py::array_t<Krn> kernel, std::optional<Seq> axis, unsigned threads);

template <typename T>
py::array_t<T> gaussian_kernel(T sigma, unsigned order, T truncate);

template <typename T, typename U>
py::array_t<T> gaussian_kernel_vec(std::vector<T> sigma, U order, T truncate);

template <typename T, typename U, typename V>
py::array_t<T> gaussian_filter(py::array_t<T> inp, U sigma, V order, remove_complex_t<T> truncate, std::string mode, unsigned threads);

template <typename T, typename U>
py::array_t<T> gaussian_gradient_magnitude(py::array_t<T> inp, U sigma, std::string mode, remove_complex_t<T> truncate, unsigned threads);

}

#endif