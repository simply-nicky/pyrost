#include "fft_functions.hpp"

namespace cbclib {

size_t next_fast_len(size_t target)
{
    if (target <= 16) return target;
    if (!(target & (target - 1))) return target;
    if (target <= detail::LPRE[detail::LPRE.size() - 1])
    {
        return *searchsorted(target, detail::LPRE.cbegin(), detail::LPRE.cend(), side::left, std::less<size_t>());
    }
    size_t match, best_match = 2 * target;

    match = detail::find_match(target, 1);
    if (match < best_match) best_match = match;
    match = detail::find_match(target, 11);
    if (match < best_match) best_match = match;
    match = detail::find_match(target, 13);
    if (match < best_match) best_match = match;
    return best_match;
}

template <typename Inp, typename Krn, typename Seq>
auto fft_convolve(py::array_t<Inp> inp, py::array_t<Krn> kernel, std::optional<Seq> axis, unsigned threads)
{
    using Out = std::common_type_t<Inp, Krn>;
    assert(PyArray_API);

    sequence<long> seq;
    if (!axis)
    {
        if (inp.ndim() != kernel.ndim())
            throw std::invalid_argument("inp and kernel have different numbers of dimensions: " +
                                        std::to_string(inp.ndim()) + " and " + std::to_string(kernel.ndim()));

        std::vector<long> vec (inp.ndim(), 0);
        std::iota(vec.begin(), vec.end(), 0);
        seq = std::move(vec);
    }
    else seq = axis.value();

    if (seq.size() != static_cast<size_t>(kernel.ndim()))
        throw std::invalid_argument("Wrong number of axes (" + std::to_string(seq.size()) +
                                    "), must be " + std::to_string(kernel.ndim()));

    seq = seq.unwrap(inp.ndim());
    inp = seq.swap_axes(inp);

    auto iarr = array<Inp>(inp.request());
    auto karr = array<Krn>(kernel.request());
    auto out = py::array_t<Out>(iarr.shape);
    auto oarr = array<Out>(out.request());

    auto ax = iarr.ndim - seq.size();
    std::vector<size_t> axes (seq.size());
    std::iota(axes.begin(), axes.end(), ax);
    auto repeats = get_size(iarr.shape.begin(), std::next(iarr.shape.begin(), ax));
    threads = (threads > repeats) ? repeats : threads;

    std::vector<size_t> fshape;
    std::transform(karr.shape.begin(), karr.shape.end(), std::next(iarr.shape.begin(), ax),
                   std::back_inserter(fshape), [](size_t nk, size_t ni){return next_fast_len(nk + ni);});
    auto bshape = fftw_buffer_shape<Out>(fshape);

    Out factor = 1.0 / get_size(fshape.begin(), fshape.end());

    thread_exception e;

    py::gil_scoped_release release;

    vector_array<Out> kbuffer (bshape);
    write_buffer(kbuffer, fshape, std::move(karr));
    auto kbuf_inp = kbuffer.ptr;
    auto kbuf_out = reinterpret_cast<std::complex<remove_complex_t<Out>> *>(kbuffer.ptr);

    auto fwd_plan = make_forward_plan(fshape, kbuf_inp, kbuf_out);
    auto bwd_plan = make_backward_plan(fshape, kbuf_out, kbuf_inp);

    fftw_execute(fwd_plan, kbuf_inp, kbuf_out);

    #pragma omp parallel num_threads(threads)
    {
        vector_array<Out> ibuffer (bshape);

        auto ibuf_inp = ibuffer.ptr;
        auto ibuf_out = reinterpret_cast<std::complex<remove_complex_t<Out>> *>(ibuffer.ptr);
        auto buf_size = is_complex_v<Out> ? ibuffer.size : ibuffer.size / 2;

        #pragma omp for
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                write_buffer(ibuffer, fshape, iarr.slice(i, axes));
                fftw_execute(fwd_plan, ibuf_inp, ibuf_out);
                for (size_t j = 0; j < buf_size; j++) ibuf_out[j] *= kbuf_out[j] * factor;
                fftw_execute(bwd_plan, ibuf_out, ibuf_inp);
                read_buffer(ibuffer, fshape, oarr.slice(i, axes));
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return seq.swap_axes_back(out);
}

template <typename T>
py::array_t<T> gaussian_kernel(T sigma, unsigned order, T truncate)
{
    auto size = 2 * size_t(sigma * truncate) + 1;
    auto out = py::array_t<T>({py::ssize_t(size)});

    gauss_kernel(array<T>(out.request()).begin(), size, sigma, order);

    return out;
}

template <typename T, typename U>
py::array_t<T> gaussian_kernel_vec(std::vector<T> sigma, U order, T truncate)
{
    sequence<unsigned> orders (order, sigma.size());

    std::vector<std::vector<T>> gaussians;
    std::vector<py::ssize_t> shape;
    for (size_t i = 0; i < sigma.size(); i++)
    {
        auto & gauss = gaussians.emplace_back();
        auto size = 2 * size_t(sigma[i] * truncate) + 1;
        gauss_kernel(std::back_inserter(gauss), size, sigma[i], orders[i]);
        shape.push_back(size);
    }

    auto out = py::array_t<T>(shape);
    auto oarr = array<T>(out.request());

    for (auto riter = rect_iterator(oarr.shape); !riter.is_end(); ++riter)
    {
        T val = T(1.0);
        for (size_t i = 0; i < oarr.ndim; i++) val *= gaussians[i][riter.coord[i]];
        oarr[riter.index] = val;
    }

    return out;
}

template <typename T, typename F>
void gauss_filter(array<T> & out, array<T> input, const std::vector<F> & sigmas, const std::vector<unsigned> & orders, F truncate, extend mode, unsigned threads)
{
    thread_exception e;

    py::gil_scoped_release release;    

    for (size_t axis = 0; axis < input.ndim; axis++)
    {
        if (!isclose(sigmas[axis], F()))
        {
            auto repeats = input.size / input.shape[axis];
            auto t = (threads > repeats) ? repeats : threads;

            std::array<size_t, 1> fshape = {next_fast_len(2 * size_t(sigmas[axis] * truncate) + 1 + input.shape[axis])};
            size_t buf_size = fftw_buffer_shape<T>(fshape)[0];
            T factor = 1.0 / fshape[0];

            std::vector<T> kbuffer;
            gauss_kernel(std::back_inserter(kbuffer), buf_size, sigmas[axis], orders[axis]);
            auto kbuf_inp = kbuffer.data();
            auto kbuf_out = reinterpret_cast<std::complex<F> *>(kbuffer.data());

            auto fwd_plan = make_forward_plan(fshape, kbuf_inp, kbuf_out);
            auto bwd_plan = make_backward_plan(fshape, kbuf_out, kbuf_inp);

            fftw_execute(fwd_plan, kbuf_inp, kbuf_out);

            #pragma omp parallel num_threads(t)
            {
                std::vector<T> ibuffer (buf_size, T());

                auto ibuf_inp = ibuffer.data();
                auto ibuf_out = reinterpret_cast<std::complex<F> *>(ibuffer.data());
                auto buf_size = is_complex_v<T> ? ibuffer.size() : ibuffer.size() / 2;

                #pragma omp for
                for (size_t i = 0; i < repeats; i++)
                {
                    e.run([&]
                    {
                        write_line(ibuffer, fshape[0], input.line_begin(axis, i), input.line_end(axis, i), mode);
                        fftw_execute(fwd_plan, ibuf_inp, ibuf_out);
                        for (size_t j = 0; j < buf_size; j++) ibuf_out[j] *= kbuf_out[j] * factor;
                        fftw_execute(bwd_plan, ibuf_out, ibuf_inp);
                        read_line(ibuffer, fshape[0], out.line_begin(axis, i), out.line_end(axis, i));
                    });
                }
            }

            input = out;
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();
}

template <typename T, typename U, typename V>
py::array_t<T> gaussian_filter(py::array_t<T> inp, U sigma, V order, remove_complex_t<T> truncate, std::string mode, unsigned threads)
{
    using F = remove_complex_t<T>;
    assert(PyArray_API);

    auto it = modes.find(mode);
    if (it == modes.end())
        throw std::invalid_argument("invalid mode argument: " + mode);
    auto m = it->second;

    auto iarr = array<T>(inp.request());
    auto out = py::array_t<T>(iarr.shape);
    auto oarr = array<T>(out.request());

    sequence<F> sigmas (sigma, iarr.ndim);
    sequence<unsigned> orders (order, iarr.ndim);

    gauss_filter<T, F>(oarr, std::move(iarr), *sigmas, *orders, truncate, m, threads);

    return out;
}

template <typename T, typename U>
py::array_t<T> gaussian_gradient_magnitude(py::array_t<T> inp, U sigma, std::string mode, remove_complex_t<T> truncate, unsigned threads)
{
    using F = remove_complex_t<T>;
    assert(PyArray_API);

    auto it = modes.find(mode);
    if (it == modes.end())
        throw std::invalid_argument("invalid mode argument: " + mode);
    auto m = it->second;

    auto iarr = array<T>(inp.request());
    auto out = py::array_t<T>(iarr.shape);
    auto oarr = array<T>(out.request());

    sequence<F> sigmas (sigma, iarr.ndim);

    std::vector<unsigned> orders (iarr.ndim, 0);
    orders[0] = 1;
    gauss_filter<T, F>(oarr, iarr, *sigmas, orders, truncate, m, threads);

    if (iarr.ndim > 1)
    {
        std::transform(oarr.begin(), oarr.end(), oarr.begin(), [](T out){return out * out;});
        
        auto buffer = vector_array<T>(iarr.shape);
        auto add_buffer = [](T out, T buf){return out + std::pow(buf, 2);};

        for (size_t axis = 1; axis < iarr.ndim; axis++)
        {
            orders[axis - 1] = 0; orders[axis] = 1;
            gauss_filter<T, F>(buffer, iarr, *sigmas, orders, truncate, m, threads);

            std::transform(oarr.begin(), oarr.end(), buffer.begin(), oarr.begin(), add_buffer);
        }

        std::transform(oarr.begin(), oarr.end(), oarr.begin(), [](T out){return std::sqrt(out);});
    }

    return out;
}

}

PYBIND11_MODULE(fft_functions, m)
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

    m.def("next_fast_len", py::vectorize(next_fast_len), py::arg("target"));

    m.def("fft_convolve", &fft_convolve<float, float, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<float, float, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<double, double, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<double, double, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);

    m.def("fft_convolve", &fft_convolve<std::complex<float>, float, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<float>, float, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<double>, double, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<double>, double, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);

    m.def("fft_convolve", &fft_convolve<float, std::complex<float>, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<float, std::complex<float>, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<double, std::complex<double>, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<double, std::complex<double>, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);

    m.def("fft_convolve", &fft_convolve<std::complex<float>, std::complex<float>, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<float>, std::complex<float>, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<double>, std::complex<double>, int>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);
    m.def("fft_convolve", &fft_convolve<std::complex<double>, std::complex<double>, std::vector<int>>, py::arg("inp"), py::arg("kernel"), py::arg("axis") = std::nullopt, py::arg("num_threads") = 1);

    m.def("gaussian_kernel", &gaussian_kernel<float>, py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0);
    m.def("gaussian_kernel", &gaussian_kernel<double>, py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0);
    m.def("gaussian_kernel", &gaussian_kernel_vec<float, unsigned>, py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0);
    m.def("gaussian_kernel", &gaussian_kernel_vec<float, std::vector<unsigned>>, py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0);
    m.def("gaussian_kernel", &gaussian_kernel_vec<double, unsigned>, py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0);
    m.def("gaussian_kernel", &gaussian_kernel_vec<double, std::vector<unsigned>>, py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0);

    m.def("gaussian_filter", &gaussian_filter<float, float, unsigned>, py::arg("inp"), py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<float, float, std::vector<unsigned>>, py::arg("inp"), py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<float, std::vector<float>, unsigned>, py::arg("inp"), py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<float, std::vector<float>, std::vector<unsigned>>, py::arg("inp"), py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<double, double, unsigned>, py::arg("inp"), py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<double, double, std::vector<unsigned>>, py::arg("inp"), py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<double, std::vector<double>, unsigned>, py::arg("inp"), py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<double, std::vector<double>, std::vector<unsigned>>, py::arg("inp"), py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);

    m.def("gaussian_filter", &gaussian_filter<std::complex<float>, float, unsigned>, py::arg("inp"), py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<std::complex<float>, float, std::vector<unsigned>>, py::arg("inp"), py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<std::complex<float>, std::vector<float>, unsigned>, py::arg("inp"), py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<std::complex<float>, std::vector<float>, std::vector<unsigned>>, py::arg("inp"), py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<std::complex<double>, double, unsigned>, py::arg("inp"), py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<std::complex<double>, double, std::vector<unsigned>>, py::arg("inp"), py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<std::complex<double>, std::vector<double>, unsigned>, py::arg("inp"), py::arg("sigma"), py::arg("order") = 0, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);
    m.def("gaussian_filter", &gaussian_filter<std::complex<double>, std::vector<double>, std::vector<unsigned>>, py::arg("inp"), py::arg("sigma"), py::arg("order") = std::vector<unsigned>{0}, py::arg("truncate") = 4.0, py::arg("mode") = "reflect", py::arg("num_threads") = 1);

    m.def("gaussian_gradient_magnitude", &gaussian_gradient_magnitude<float, float>, py::arg("inp"), py::arg("sigma"), py::arg("mode") = "reflect", py::arg("truncate") = 4.0, py::arg("num_threads") = 1);
    m.def("gaussian_gradient_magnitude", &gaussian_gradient_magnitude<float, std::vector<float>>, py::arg("inp"), py::arg("sigma"), py::arg("mode") = "reflect", py::arg("truncate") = 4.0, py::arg("num_threads") = 1);
    m.def("gaussian_gradient_magnitude", &gaussian_gradient_magnitude<double, double>, py::arg("inp"), py::arg("sigma"), py::arg("mode") = "reflect", py::arg("truncate") = 4.0, py::arg("num_threads") = 1);
    m.def("gaussian_gradient_magnitude", &gaussian_gradient_magnitude<double, std::vector<double>>, py::arg("inp"), py::arg("sigma"), py::arg("mode") = "reflect", py::arg("truncate") = 4.0, py::arg("num_threads") = 1);

    m.def("gaussian_gradient_magnitude", &gaussian_gradient_magnitude<std::complex<float>, float>, py::arg("inp"), py::arg("sigma"), py::arg("mode") = "reflect", py::arg("truncate") = 4.0, py::arg("num_threads") = 1);
    m.def("gaussian_gradient_magnitude", &gaussian_gradient_magnitude<std::complex<float>, std::vector<float>>, py::arg("inp"), py::arg("sigma"), py::arg("mode") = "reflect", py::arg("truncate") = 4.0, py::arg("num_threads") = 1);
    m.def("gaussian_gradient_magnitude", &gaussian_gradient_magnitude<std::complex<double>, double>, py::arg("inp"), py::arg("sigma"), py::arg("mode") = "reflect", py::arg("truncate") = 4.0, py::arg("num_threads") = 1);
    m.def("gaussian_gradient_magnitude", &gaussian_gradient_magnitude<std::complex<double>, std::vector<double>>, py::arg("inp"), py::arg("sigma"), py::arg("mode") = "reflect", py::arg("truncate") = 4.0, py::arg("num_threads") = 1);

}