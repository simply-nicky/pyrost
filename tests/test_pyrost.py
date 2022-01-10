import os
import shutil
from datetime import datetime
import pytest
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
def sim_obj(st_params: st_sim.STParams):
    sim_obj = st_sim.STSim(st_params)
    return sim_obj

@pytest.fixture(scope='session')
def ptych(sim_obj: st_sim.STSim):
    data = sim_obj.ptychograph()
    return data

@pytest.fixture(params=['float32', 'float64'])
def loader(request):
    """
    Return the default loader.
    """
    protocol = rst.CXIProtocol.import_default(float_precision=request.param)
    return rst.CXILoader.import_default(protocol=protocol)

@pytest.fixture(params=['float32', 'float64'])
def converter(request):
    """
    Return the default loader.
    """
    protocol = rst.CXIProtocol.import_default(float_precision=request.param)
    return st_sim.STConverter(protocol=protocol)

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

@pytest.mark.st_sim
def test_st_params(st_params, ini_path):
    assert not os.path.isfile(ini_path)
    ini_parser = st_params.export_ini()
    with open(ini_path, 'w') as ini_file:
        ini_parser.write(ini_file)
    new_params = st_sim.STParams.import_ini(ini_path)
    assert new_params.export_dict() == st_params.export_dict()

@pytest.mark.st_sim
def test_ptych(ptych, st_params):
    assert len(ptych.shape) == 3
    assert ptych.shape[0] == st_params.n_frames

@pytest.mark.rst
def test_save_and_load_sim(converter, loader, ptych, sim_obj, temp_dir):
    assert os.path.isdir(temp_dir)
    converter.save_sim(ptych, sim_obj, temp_dir)
    cxi_path = os.path.join(temp_dir, 'data.cxi')
    assert os.path.isfile(cxi_path)
    data_dict = loader.load_to_dict(data_files=cxi_path)
    for attr in rst.STData.attr_set:
        if attr != 'protocol':
            assert not data_dict[attr] is None

@pytest.mark.rst
def test_st_update_sim(converter, ptych, sim_obj):
    st_data = converter.export_data(ptych, sim_obj)
    assert st_data.data.dtype == converter.protocol.get_dtype('data')
    st_obj = st_data.get_st()
    pixel_map0 = st_obj.pixel_map.copy()
    st_obj.iter_update(sw_x=10, h0=15, verbose=True, n_iter=10)
    assert (st_obj.pixel_map == pixel_map0).all()
    assert st_obj.pixel_map.dtype == converter.protocol.get_dtype('pixel_map')

@pytest.mark.rst
def test_load_exp(path, roi, defocus, loader):
    assert os.path.isfile(path)
    data_dict = loader.load_to_dict(data_files=path, roi=roi, defocus=defocus)
    for attr in rst.STData.attr_set:
        if attr != 'protocol':
            assert not data_dict[attr] is None

@pytest.mark.rst
def test_st_update_exp(path, roi, defocus, loader):
    assert os.path.isfile(path)
    data = loader.load(data_files=path, roi=roi, defocus=defocus)
    assert data.data.dtype == loader.get_dtype('data')
    st_obj = data.get_st()
    pixel_map0 = st_obj.pixel_map.copy()
    st_obj.iter_update_gd(sw_x=10, h0=30, blur=8.0, verbose=True, n_iter=10)
    assert (st_obj.pixel_map == pixel_map0).all()
    assert st_obj.pixel_map.dtype == loader.get_dtype('pixel_map')

@pytest.mark.standalone
def test_full(converter, ptych, sim_obj):
    data = converter.export_data(ptych, sim_obj)
    assert data.data.dtype == converter.protocol.get_dtype('data')
    st_obj = data.get_st()
    st_res = st_obj.iter_update_gd(sw_x=10, h0=15, blur=2.0, verbose=True, n_iter=10)
    data.update_phase(st_res)
    fit = data.fit_phase(axis=1)
    assert (st_obj.pixel_map != st_res.pixel_map).any()
    assert st_res.pixel_map.dtype == converter.protocol.get_dtype('pixel_map')
    assert not fit is None
