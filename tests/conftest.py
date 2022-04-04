def pytest_addoption(parser):
    parser.addoption("--scan_num", type=int, default=3985,
                     help="Scan number")
    parser.addoption("--roi", type=int, nargs=4, default=(270, 310, 350, 1280),
                     help="Region of interest")
    parser.addoption("--good_frames", type=int, nargs=2, default=(0, 90),
                     help="Range of good frames")
    parser.addoption("--defocus", type=float, default=152e-6,
                     help="Defocus distance [m]")
    parser.addoption("--alpha", type=float, default=0.06,
                     help="Defocus distance [m]")

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_dict = vars(metafunc.config.option)
    for attr in ('scan_num', 'roi', 'good_frames', 'defocus', 'alpha'):
        option_value = option_dict.get(attr)
        if attr in metafunc.fixturenames and option_value is not None:
            metafunc.parametrize(attr, [option_value])
