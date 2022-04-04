import os
import shutil
from datetime import datetime
from typing import List, Tuple
import pytest
import numpy as np
import pyrost as rst
from pyrost import simulation as st_sim

@pytest.fixture(params=[{'detx_size': 300, 'dety_size': 300, 'n_frames': 50, 'p0': 1e6,
                         'pix_size': 300, 'bar_size': 0.3, 'bar_rnd': 0.5, 'alpha': 0.05},
                        {'detx_size': 300, 'dety_size': 300, 'n_frames': 50, 'p0': 1e6,
                         'pix_size': 300, 'bar_size': 0.3, 'bar_rnd': 0.5, 'alpha': 0.03}],
                scope='session')
def st_params(request) -> st_sim.STParams:
    """Return a default instance of simulation parameters.
    """
    return st_sim.STParams.import_default(**request.param)

@pytest.fixture(scope='session')
def st_converter(st_params: st_sim.STParams) -> st_sim.STConverter:
    sim_obj = st_sim.STSim(st_params)
    ptych = sim_obj.ptychograph()
    return st_sim.STConverter(sim_obj, ptych)

@pytest.fixture(scope='session')
def temp_dir() -> str:
    now = datetime.now()
    path = now.strftime("temp_%m_%d_%H%M%S")
    os.mkdir(path)
    yield path
    shutil.rmtree(path)

@pytest.fixture(scope='function')
def ini_path(temp_dir: str) -> str:
    """Return a path to the experimental speckle tracking data.
    """
    path = os.path.join(temp_dir, 'test.ini')
    yield path
    os.remove(path)

@pytest.fixture
def attributes():
    return ['files', 'wavelength', 'num_threads', 'x_pixel_size', 'y_pixel_size',
            'whitefield', 'distance', 'good_frames', 'data', 'translations',
            'frames', 'basis_vectors', 'mask']

@pytest.fixture
def crop(roi: Tuple[int, int, int, int]) -> rst.Crop:
    return rst.Crop(roi)

@pytest.fixture
def good_frames_list(good_frames: Tuple[int, int]) -> np.ndarray:
    return np.arange(good_frames[0], good_frames[1])

@pytest.mark.st_sim
def test_st_params(st_params: st_sim.STParams, ini_path: str):
    assert not os.path.isfile(ini_path)
    ini_parser = st_params.export_ini()
    with open(ini_path, 'w') as ini_file:
        ini_parser.write(ini_file)
    new_params = st_sim.STParams.import_ini(ini_path)
    assert new_params.export_dict() == st_params.export_dict()

@pytest.mark.st_sim
def test_save_and_load_sim(st_converter: st_sim.STConverter, temp_dir: str):
    assert os.path.isdir(temp_dir)
    out_path = os.path.join(temp_dir, 'sim.cxi')
    st_converter.save(out_path)
    assert os.path.isfile(out_path)
    data = st_converter.export_data(out_path)
    for attr in data.files:
        assert attr in data.contents()

@pytest.mark.standalone
def test_st_update_sim(st_converter: st_sim.STConverter, temp_dir: str):
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
def test_load_exp(scan_num: int, temp_dir: str, attributes: List[str]):
    out_path = os.path.join(temp_dir, 'sigray.cxi')
    data = rst.cxi_converter_sigray(out_path=out_path, scan_num=scan_num, target='Mo')
    for attr in data.contents():
        assert attr in attributes

@pytest.mark.rst
def test_defocus_sweep_exp(scan_num: int, temp_dir: str, crop: rst.Crop,
                           good_frames_list: np.ndarray, defocus: float):
    out_path = os.path.join(temp_dir, 'sigray.cxi')
    data = rst.cxi_converter_sigray(out_path=out_path, scan_num=scan_num,
                                    target='Mo', transform=crop)
    data = data.mask_frames(good_frames_list)
    data = data.integrate_data()
    defoci = np.linspace(0.5 * defocus, 2.0 * defocus)
    sweep_scan = data.defocus_sweep(defoci_x=defoci)
    df_est = defoci[np.argmax(sweep_scan)]
    assert np.abs(df_est - defocus) < 0.1 * defocus

@pytest.mark.rst
def test_st_udpate_exp(scan_num: int, temp_dir: str, crop: rst.Crop,
                       good_frames_list: np.ndarray, defocus: float, alpha: float):
    print(crop)
    out_path = os.path.join(temp_dir, 'sigray.cxi')
    data = rst.cxi_converter_sigray(out_path=out_path, scan_num=scan_num,
                                    target='Mo', transform=crop, defocus_x=defocus)
    data = data.mask_frames(good_frames_list)
    data = data.integrate_data()
    st_obj = data.get_st()
    h0 = st_obj.find_hopt()
    st_res = st_obj.train_adapt(search_window=(0.0, 10.0, 0.1), h0=h0, blur=18.0)
    data.import_st(st_res)
    fit = data.get_fit(axis=1).remove_linear_term().fit(max_order=2)
    assert np.abs(fit['c_3'] - alpha) < 0.3 * np.abs(alpha)
