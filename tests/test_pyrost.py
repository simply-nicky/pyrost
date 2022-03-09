import os
import shutil
from datetime import datetime
import pytest
import numpy as np
import pyrost as rst
import pyrost.simulation as st_sim

@pytest.fixture(params=[{'ap_x': 10, 'ap_y': 5, 'defocus': -2e2, 'detx_size': 300,
                         'dety_size': 300, 'n_frames': 50, 'pix_size': 8},
                        {'ap_x': 10, 'ap_y': 5, 'defocus': 1e2, 'detx_size': 300,
                         'dety_size': 300, 'n_frames': 50, 'pix_size': 8}],
                scope='session')
def st_params(request):
    """Return a default instance of simulation parameters.
    """
    return st_sim.STParams.import_default(**request.param)

@pytest.fixture(scope='session')
def st_converter(st_params):
    sim_obj = st_sim.STSim(st_params)
    ptych = sim_obj.ptychograph()
    return st_sim.STConverter(sim_obj, ptych)

@pytest.fixture(scope='session')
def temp_dir():
    now = datetime.now()
    path = now.strftime("temp_%m_%d_%H%M%S")
    os.mkdir(path)
    yield path
    shutil.rmtree(path)

@pytest.fixture(scope='function')
def ini_path(temp_dir):
    """Return a path to the experimental speckle tracking data.
    """
    path = os.path.join(temp_dir, 'test.ini')
    yield path
    os.remove(path)

@pytest.fixture()
def attributes():
    return ['basis_vectors', 'data', 'distance', 'translations', 'wavelength',
            'x_pixel_size', 'y_pixel_size']

@pytest.fixture()
def crop(roi):
    return rst.Crop([[roi[0], roi[2]], [roi[1], roi[3]]])

@pytest.mark.st_sim
def test_st_params(st_params, ini_path):
    assert not os.path.isfile(ini_path)
    ini_parser = st_params.export_ini()
    with open(ini_path, 'w') as ini_file:
        ini_parser.write(ini_file)
    new_params = st_sim.STParams.import_ini(ini_path)
    assert new_params.export_dict() == st_params.export_dict()

@pytest.mark.st_sim
def test_save_and_load_sim(st_converter, temp_dir):
    assert os.path.isdir(temp_dir)
    out_path = os.path.join(temp_dir, 'sim.cxi')
    st_converter.save(out_path)
    assert os.path.isfile(out_path)
    data = st_converter.export_data(out_path)
    for attr in data.contents():
        if attr in data.init_funcs:
            assert attr in data.files

@pytest.mark.standalone
def test_st_update_sim(st_converter, temp_dir):
    out_path = os.path.join(temp_dir, 'sim.cxi')
    data = st_converter.export_data(out_path)
    st_obj = data.get_st()
    h0 = st_obj.find_hopt()
    st_res = st_obj.train(search_window=(0.0, 10.0, 0.1), h0=h0, blur=8.0)
    data.import_st(st_res)
    fit_obj = data.get_fit(axis=1, center=20)
    fit_obj = fit_obj.remove_linear_term()
    fit = fit_obj.fit(max_order=2)
    alpha = np.abs(st_converter.sim_obj.params.alpha)
    alpha_est = np.abs(fit['c_3'])
    assert np.abs(alpha - alpha_est) < 0.1 * alpha

@pytest.mark.rst
def test_load_exp(path, attributes):
    assert os.path.isfile(path)
    files = rst.CXIStore(input_files=path, output_file=path)
    for attr in files:
        assert attr in attributes

@pytest.mark.rst
def test_defocus_sweep_exp(path, crop, defocus):
    assert os.path.isfile(path)
    files = rst.CXIStore(input_files=path, output_file=path)
    data = rst.STData(files=files, transform=crop)
    defoci = np.linspace(0.5 * defocus, 2.0 * defocus)
    df_est = data.defocus_sweep(defoci_x=defoci, hval=10.0)
    assert np.abs(df_est - defocus) < 0.1 * defocus

@pytest.mark.rst
def test_st_udpate_exp(path, crop, defocus, alpha):
    assert os.path.isfile(path)
    files = rst.CXIStore(input_files=path, output_file=path)
    data = rst.STData(files=files, transform=crop, defocus=defocus)
    st_obj = data.get_st()
    h0 = st_obj.find_hopt()
    st_res = st_obj.train_adapt(search_window=(0.0, 10.0, 0.1), h0=h0, blur=8.0)
    data.import_st(st_res)
    fit = data.get_fit(axis=1).remove_linear_term().fit(max_order=3)
    assert np.abs(fit['c_3'] - alpha) < 0.1 * np.abs(alpha)
