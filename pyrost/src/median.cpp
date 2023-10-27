#include "median.hpp"

namespace cbclib {

template <typename T>
void check_mask(const py::array_t<T, py::array::c_style | py::array::forcecast> & inp,
                std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> & mask)
{
    py::buffer_info ibuf = inp.request();
    if (!mask)
    {
        mask = py::array_t<bool>(ibuf.shape);
        PyArray_FILLWBYTE(mask.value().ptr(), 1);
    }
    py::buffer_info mbuf = mask.value().request();
    if (!std::equal(mbuf.shape.begin(), mbuf.shape.end(), ibuf.shape.begin()))
    {
        std::ostringstream oss1, oss2;
        std::copy(mbuf.shape.begin(), mbuf.shape.end(), std::experimental::make_ostream_joiner(oss1, ", "));
        std::copy(ibuf.shape.begin(), ibuf.shape.end(), std::experimental::make_ostream_joiner(oss2, ", "));
        throw std::invalid_argument("mask and inp arrays must have identical shapes: {" + oss1.str() +
                                    "}, {" + oss2.str() + "}");
    }
}

template <typename T, typename U>
py::array_t<T> median(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                      std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                      U axis, unsigned threads)
{
    assert(PyArray_API);

    check_mask(inp, mask);

    sequence<long> seq (axis);
    seq = seq.unwrap(inp.ndim());
    inp = seq.swap_axes(inp);
    mask = seq.swap_axes(mask.value());

    auto ibuf = inp.request();
    auto ax = ibuf.ndim - seq.size();
    auto out_shape = std::vector<py::ssize_t>(ibuf.shape.begin(), std::next(ibuf.shape.begin(), ax));
    auto out = py::array_t<T>(out_shape);

    if (!out.size()) return out;

    auto new_shape = out_shape;
    new_shape.push_back(ibuf.size / out.size());
    inp = inp.reshape(new_shape);
    mask = mask.value().reshape(new_shape);

    auto oarr = array<T>(out.request());
    auto iarr = array<T>(inp.request());
    auto marr = array<bool>(mask.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > oarr.size) ? oarr.size : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<T> buffer;
        std::vector<size_t> idxs (iarr.shape[ax], 0);
        std::iota(idxs.begin(), idxs.end(), 0);

        #pragma omp for
        for (size_t i = 0; i < oarr.size; i++)
        {
            e.run([&]
            {
                buffer.clear();
                auto miter = marr.line_begin(ax, i);
                auto iiter = iarr.line_begin(ax, i);

                for (auto idx : idxs) if (miter[idx]) buffer.push_back(iiter[idx]);

                if (buffer.size()) oarr[i] = *wirthmedian(buffer.begin(), buffer.end(), std::less<T>());
                else oarr[i] = T();
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename U>
py::array_t<T> filter_preprocessor(py::array_t<T, py::array::c_style | py::array::forcecast> & inp, std::optional<U> size,
                                   std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> & fprint,
                                   std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> & mask,
                                   std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> & inp_mask)
{
    check_mask(inp, mask);
    if (!inp_mask) inp_mask = mask.value();

    if (!size && !fprint)
        throw std::invalid_argument("size or fprint must be provided");

    auto ibuf = inp.request();
    if (!fprint)
    {
        fprint = py::array_t<bool>(sequence<size_t>(size.value(), ibuf.ndim));
        PyArray_FILLWBYTE(fprint.value().ptr(), 1);
    }
    py::buffer_info fbuf = fprint.value().request();
    if (fbuf.ndim != ibuf.ndim)
        throw std::invalid_argument("fprint must have the same number of dimensions (" + std::to_string(fbuf.ndim) + 
                                    ") as the input (" + std::to_string(ibuf.ndim) + ")");

    return py::array_t<T>(ibuf.shape);
}

template <typename T, typename U>
py::array_t<T> median_filter(py::array_t<T, py::array::c_style | py::array::forcecast> inp, std::optional<U> size,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> fprint,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                             std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> inp_mask,
                             std::string mode, const T & cval, unsigned threads)
{
    assert(PyArray_API);

    auto it = modes.find(mode);
    if (it == modes.end())
        throw std::invalid_argument("invalid mode argument: " + mode);
    auto m = it->second;

    auto out = filter_preprocessor(inp, size, fprint, mask, inp_mask);

    if (!out.size()) return out;

    auto oarr = array<T>(out.request());
    auto iarr = array<T>(inp.request());
    auto marr = array<bool>(mask.value().request());
    auto imarr = array<bool>(inp_mask.value().request());
    auto farr = array<bool>(fprint.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        footprint<T> fpt (farr);
        std::vector<long> coord (iarr.ndim, 0);

        #pragma omp for schedule(guided)
        for (size_t i = 0; i < iarr.size; i++)
        {
            e.run([&]
            {
                if (marr[i])
                {
                    iarr.unravel_index(coord.begin(), i);
                    fpt.update(coord, iarr, imarr, m, cval);

                    if (fpt.data.size()) oarr[i] = *wirthmedian(fpt.data.begin(), fpt.data.end(), std::less<T>());
                }
                else oarr[i] = T();
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename U>
auto robust_mean(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                 std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> mask,
                 U axis, double r0, double r1, int n_iter, double lm, unsigned threads) -> py::array_t<std::common_type_t<T, float>>
{
    using D = std::common_type_t<T, float>;
    assert(PyArray_API);

    check_mask(inp, mask);

    sequence<long> seq (axis);
    seq = seq.unwrap(inp.ndim());
    inp = seq.swap_axes(inp);
    mask = seq.swap_axes(mask.value());

    auto ibuf = inp.request();
    auto ax = ibuf.ndim - seq.size();
    auto out_shape = std::vector<py::ssize_t>(ibuf.shape.begin(), std::next(ibuf.shape.begin(), ax));
    auto out = py::array_t<D>(out_shape);

    if (!out.size()) return out;

    auto new_shape = out_shape;
    new_shape.push_back(ibuf.size / out.size());
    inp = inp.reshape(new_shape);
    mask = mask.value().reshape(new_shape);

    auto oarr = array<D>(out.request());
    auto iarr = array<T>(inp.request());
    auto marr = array<bool>(mask.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > oarr.size) ? oarr.size : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<T> buffer;
        std::vector<D> err (iarr.shape[ax]);
        std::vector<size_t> idxs (iarr.shape[ax]);

        size_t j0 = r0 * iarr.shape[ax], j1 = r1 * iarr.shape[ax];
        D mean;

        #pragma omp for
        for (size_t i = 0; i < oarr.size; i++)
        {
            e.run([&]
            {
                auto iiter = iarr.line_begin(ax, i);
                auto miter = marr.line_begin(ax, i);

                auto get_err = [=, &mean](size_t idx){return miter[idx] * (iiter[idx] - mean) * (iiter[idx] - mean);};

                buffer.clear();
                std::iota(idxs.begin(), idxs.end(), 0);
                for (auto idx : idxs) if (miter[idx]) buffer.push_back(iiter[idx]);

                if (buffer.size()) mean = *wirthmedian(buffer.begin(), buffer.end(), std::less<T>());
                else mean = D();


                for (int n = 0; n < n_iter; n++)
                {
                    std::iota(idxs.begin(), idxs.end(), 0);
                    std::transform(idxs.begin(), idxs.end(), err.begin(), get_err);
                    std::sort(idxs.begin(), idxs.end(), [&err](size_t i1, size_t i2){return err[i1] < err[i2];});

                    mean = std::transform_reduce(idxs.begin() + j0, idxs.begin() + j1, D(), std::plus<D>(),
                                                [=](size_t idx){return miter[idx] * iiter[idx];}) / (j1 - j0);
                }

                std::iota(idxs.begin(), idxs.end(), 0);
                std::transform(idxs.begin(), idxs.end(), err.begin(), get_err);
                std::sort(idxs.begin(), idxs.end(), [&err](size_t i1, size_t i2){return err[i1] < err[i2];});

                D cumsum = D(); mean = D(); int count = 0;
                for (size_t j = 0; j < idxs.size(); j++)
                {
                    if (lm * cumsum > j * err[idxs[j]]) {mean += miter[idxs[j]] * iiter[idxs[j]]; count++;}
                    cumsum += err[idxs[j]];
                }
                if (count) oarr[i] = mean / count;
                else oarr[i] = D();
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

}

PYBIND11_MODULE(median, m)
{
    using namespace cbclib;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    m.def("median", &median<double, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<double, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<float, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<float, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<int, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<int, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<long, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<long, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<size_t, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<size_t, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);

    m.def("median_filter", &median_filter<double, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<double, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<float, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<float, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<int, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<int, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<long, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<long, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<size_t, size_t>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);
    m.def("median_filter", &median_filter<size_t, std::vector<size_t>>, py::arg("inp"), py::arg("size") = nullptr, py::arg("footprint") = nullptr, py::arg("mask") = nullptr, py::arg("inp_mask") = nullptr, py::arg("mode") = "reflect", py::arg("cval") = 0.0, py::arg("num_threads") = 1);

    m.def("robust_mean", &robust_mean<double, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<double, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<float, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<float, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<int, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<int, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<long, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<long, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<size_t, int>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<size_t, std::vector<int>>, py::arg("inp"), py::arg("mask") = nullptr, py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}