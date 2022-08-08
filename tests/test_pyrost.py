import os
import shutil
from datetime import datetime
from typing import List, Tuple
import pytest
import numpy as np
import pyrost as rst
from pyrost import simulation as st_sim

@pytest.fixture(params=[{'detx_size': 300, 'dety_size': 300, 'n_frames': 50, 'p0': 1e6,
                         'pix_size': 300, 'bar_size': 0.3, 'bar_rnd': 0.5, 'alpha': 0.05,
                         'num_threads': 4},
                        {'detx_size': 300, 'dety_size': 300, 'n_frames': 50, 'p0': 1e6,
                         'pix_size': 300, 'bar_size': 0.3, 'bar_rnd': 0.5, 'alpha': 0.03,
                         'num_threads': 4}],
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
def crop(roi: Tuple[int, int, int, int]) -> rst.Crop:
    return rst.Crop(roi)

@pytest.fixture
def kamzik_converter(scan_num: int) -> rst.KamzikConverter:
    converter = rst.KamzikConverter()
    converter = converter.read_logs(f'/gpfs/cfel/group/cxi/labs/MLL-Sigray/scan-logs/Scan_{scan_num:d}.log')
    return converter

@pytest.fixture
def input_file(scan_num: int) -> rst.CXIStore:
    data_dir = f'/gpfs/cfel/group/cxi/labs/MLL-Sigray/scan-frames/Scan_{scan_num:d}'
    data_files = sorted([os.path.join(data_dir, path) for path in os.listdir(data_dir)
                         if path.endswith('Lambda.nxs')])
    return rst.CXIStore(data_files)

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
    data = st_converter.export_data(out_path).load()
    for attr in data.input_file:
        assert attr in data.contents()

@pytest.mark.standalone
def test_st_update_sim(st_converter: st_sim.STConverter, temp_dir: str):
    out_path = os.path.join(temp_dir, 'sim.cxi')
    data = st_converter.export_data(out_path).load()
    st_obj = data.get_st()
    h0 = st_obj.find_hopt(verbose=True)
    st_res = st_obj.train(search_window=(0.0, 10.0, 0.1), h0=h0, blur=8.0,
                          f_tol=-1.0, n_iter=10, verbose=True)
    data.import_st(st_res)
    fit_obj = data.get_fit(axis=1, center=20)
    fit_obj = fit_obj.remove_linear_term()
    fit = fit_obj.fit(max_order=2)
    assert np.sum(np.abs(fit['rel_err'])) > 0.0

@pytest.mark.rst
def test_load_exp(kamzik_converter: rst.KamzikConverter, input_file: rst.CXIStore):
    log_data = kamzik_converter.cxi_get(['basis_vectors', 'log_translations'])
    data = rst.STData(input_file, **log_data)
    data = data.load('data')
    for attr in ('y_pixel_size', 'translations', 'basis_vectors', 'x_pixel_size',
                 'good_frames', 'num_threads', 'input_file'):
        assert attr in data.contents()

@pytest.mark.rst
def test_kamzik_converter(kamzik_converter: rst.KamzikConverter):
    keys = kamzik_converter.cxi_keys()
    assert 'sim_translations' in keys and 'log_translations' in keys

@pytest.mark.rst
def test_defocus_sweep_exp(kamzik_converter: rst.KamzikConverter, input_file: rst.CXIStore, crop: rst.Crop,
                           good_frames_list: np.ndarray, defocus: float, distance: float, wavelength: float):
    log_data = kamzik_converter.cxi_get(['basis_vectors', 'log_translations'])
    data = rst.STData(input_file, **log_data, distance=distance, wavelength=wavelength, transform=crop)
    data = data.load('data').mask_frames(good_frames_list).update_mask(vmax=100000).integrate_data()

    defoci = np.linspace(0.5 * defocus, 2.0 * defocus, 50)
    sweep_scan = data.defocus_sweep(defoci_x=defoci, size=50)
    assert np.all(np.asarray(sweep_scan) > 0.0)

@pytest.mark.rst
def test_st_udpate_exp(kamzik_converter: rst.KamzikConverter, input_file: rst.CXIStore, crop: rst.Crop,
                       good_frames_list: np.ndarray, defocus: float, distance: float, wavelength: float):
    log_data = kamzik_converter.cxi_get(['basis_vectors', 'log_translations'])
    data = rst.STData(input_file, **log_data, distance=distance, wavelength=wavelength, transform=crop)
    data = data.load('data').mask_frames(good_frames_list).update_mask(vmax=100000).integrate_data()
    data = data.update_defocus(defocus)

    st_obj = data.get_st()
    h0 = st_obj.find_hopt()
    st_res = st_obj.train_adapt(search_window=(0.0, 10.0, 0.1), h0=h0, blur=8.0)
    data.import_st(st_res)
    fit = data.get_fit(axis=1).remove_linear_term().fit(max_order=3)
    assert np.sum(np.abs(fit['rel_err'])) > 0.0
