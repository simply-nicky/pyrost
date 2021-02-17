def pytest_addoption(parser):
    parser.addoption("--path", type=str, default="results/exp/Scan_1986.cxi",
                     help="Path to a CXI file")
    parser.addoption("--roi", type=int, nargs=4, default=(0, 1, 360, 1090),
                     help="Region of interest")
    parser.addoption("--defocus", type=float, default=1.0e-4,
                     help="Defocus distance [m]")

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_dict = vars(metafunc.config.option)
    for attr in ('path', 'roi', 'defocus'):
        option_value = option_dict.get(attr)
        if attr in metafunc.fixturenames and option_value is not None:
            metafunc.parametrize(attr, [option_value])
