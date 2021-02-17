import os
import shutil
from datetime import datetime
import pytest
import pyrost as rst
import pyrost.simulation as st_sim

@pytest.fixture(params=[{'det_dist': 5e5, 'n_frames': 10, 'ap_x': 4,
                         'ap_y': 1, 'focus': 3e3, 'defocus': 2e2},
                        {'det_dist': 4.5e5, 'n_frames': 5, 'ap_x': 3,
                         'ap_y': 1.5, 'focus': 2e3, 'defocus': 1e2}],
                scope='session')
def st_params(request):
    """Return a default instance of simulation parameters.
    """
    return st_sim.parameters(**request.param)

@pytest.fixture(scope='session')
def sim_obj(st_params):
    sim_obj = st_sim.STSim(st_params)
    return sim_obj

@pytest.fixture(scope='session')
def ptych(sim_obj):
    data = sim_obj.ptychograph()
    return data

@pytest.fixture(params=['float32', 'float64'])
def loader(request):
    """
    Return the default loader.
    """
    protocol = rst.cxi_protocol(float_precision=request.param)
    return rst.loader(protocol=protocol)

@pytest.fixture(params=['float32', 'float64'])
def converter(request):
    """
    Return the default loader.
    """
    return st_sim.converter(float_precision=request.param)

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
def test_load_exp(path, roi, defocus, loader):
    assert os.path.isfile(path)
    data_dict = loader.load_dict(path=path, roi=roi, defocus=defocus)
    for attr in rst.STData.attr_set:
        assert not data_dict[attr] is None

@pytest.mark.rst
def test_save_and_load_sim(converter, loader, ptych, sim_obj, temp_dir):
    assert os.path.isdir(temp_dir)
    converter.save_sim(ptych, sim_obj, temp_dir)
    cxi_path = os.path.join(temp_dir, 'data.cxi')
    assert os.path.isfile(cxi_path)
    data_dict = loader.load_dict(cxi_path)
    for attr in rst.STData.attr_set:
        assert not data_dict[attr] is None

@pytest.mark.rst
def test_st_update_sim(converter, ptych , sim_obj):
    st_data = converter.export_data(ptych, sim_obj)
    assert st_data.data.dtype == converter.protocol.known_types['float']
    st_obj = st_data.get_st()
    pixel_map0 = st_obj.pixel_map.copy()
    st_obj.iter_update(sw_fs=10, ls_pm=2.5, ls_ri=15,
                       verbose=True, n_iter=10)
    assert (st_obj.pixel_map == pixel_map0).all()
    assert st_obj.pixel_map.dtype == converter.protocol.known_types['float']

@pytest.mark.standalone
def test_full(converter, ptych, sim_obj):
    data = converter.export_data(ptych, sim_obj)
    assert data.data.dtype == converter.protocol.known_types['float']
    st_obj = data.get_st()
    st_res = st_obj.iter_update(sw_fs=10, ls_pm=2.5, ls_ri=15,
                                verbose=True, n_iter=10)
    data = data.update_phase(st_res)
    fit = data.fit_phase(axis=1)
    assert (st_obj.pixel_map != st_res.pixel_map).any()
    assert st_res.pixel_map.dtype == converter.protocol.known_types['float']
    assert not fit is None
