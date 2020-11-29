import os
import pytest
import pyrost as rst
import pyrost.simulation as st_sim
import numpy as np

@pytest.fixture(params=[{'det_dist': 5e5, 'n_frames': 10, 'ap_x': 4,
                         'ap_y': 1, 'focus': 3e3, 'defocus': 2e2},
                        {'det_dist': 4.5e5, 'n_frames': 5, 'ap_x': 3,
                         'ap_y': 1.5, 'focus': 2e3, 'defocus': 1e2}])
def st_params(request):
    """Return a default instance of simulation parameters.
    """
    return st_sim.parameters(**request.param)

@pytest.fixture(params=['results/test', 'results/test_ideal'])
def sim_data(request):
    """Return the data path and all the necessary parameters
    of the simulated speckle tracking 1d scan.
    """
    return request.param

@pytest.fixture(params=[{'scan_num': 1986, 'roi': (0, 1, 360, 1090),
                         'defocus': 1.0e-4},
                        {'scan_num': 1740, 'roi': (0, 1, 350, 1065),
                         'defocus': 1.5e-4}])
def exp_data(request):
    """Return the data path and all the necessary parameters
    of the experimental speckle tracking 1d scan.
    """
    params = {key: request.param[key] for key in request.param.keys() if key != 'scan_num'}
    params['path'] = 'results/exp/Scan_{:d}.cxi'.format(request.param['scan_num'])
    return params

@pytest.fixture(params=[{'name': 'diatom.cxi', 'good_frames': np.arange(1, 121),
                         'defocus': 2.23e-3, 'roi': (70, 420, 50, 460)}])
def exp_data_2d(request):
    """Return the data path and all the necessary parameters
    of the experimental speckle tracking 2d scan.
    """
    params = {key: request.param[key] for key in request.param.keys() if key != 'name'}
    params['path'] = os.path.join('results/exp', request.param['name'])
    return params

@pytest.fixture(params=['float32', 'float64'])
def loader(request):
    """
    Return the default loader.
    """
    return rst.loader(request.param)

@pytest.fixture(params=['float32', 'float64'])
def converter(request):
    """
    Return the default loader.
    """
    return st_sim.converter(float_precision=request.param)

@pytest.fixture(scope='function')
def ini_path():
    """Return a path to the experimental speckle tracking data.
    """
    path = 'test.ini'
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
def test_st_sim(st_params):
    with st_sim.STSim(st_params) as sim_obj:
        ptych = sim_obj.ptychograph()
    assert len(ptych.shape) == 3
    assert ptych.shape[0] == st_params.n_frames

@pytest.mark.rst
def test_loader_exp(exp_data, loader):
    assert os.path.isfile(exp_data['path'])
    data_dict = loader.load_dict(**exp_data)
    for attr in rst.STData.attr_set:
        assert not data_dict[attr] is None

@pytest.mark.rst
def test_loader_sim(sim_data, loader):
    assert os.path.isdir(sim_data)
    data_path = os.path.join(sim_data, 'data.cxi')
    assert os.path.isfile(data_path)
    data_dict = loader.load_dict(data_path)
    for attr in rst.STData.attr_set:
        assert not data_dict[attr] is None

@pytest.mark.rst
def test_iter_update(sim_data, loader):
    assert os.path.isdir(sim_data)
    data_path = os.path.join(sim_data, 'data.cxi')
    assert os.path.isfile(data_path)
    st_data = loader.load(data_path, roi=(0, 1, 400, 1450))
    assert st_data.data.dtype == loader.protocol.known_types['float']
    st_obj = st_data.get_st()
    pixel_map0 = st_obj.pixel_map.copy()
    st_obj.iter_update(sw_ss=0, sw_fs=150, ls_pm=2.5, ls_ri=15,
                       verbose=True, n_iter=5)
    assert (st_obj.pixel_map == pixel_map0).all()
    assert st_obj.pixel_map.dtype == loader.protocol.known_types['float']

@pytest.mark.rst
def test_data_process_routines(exp_data_2d, loader):
    assert os.path.isfile(exp_data_2d['path'])
    data = loader.load(**exp_data_2d)
    data = data.make_mask(method='eiger-bad')
    assert (data.get('whitefield') <= 65535).all()

@pytest.mark.standalone
def test_full(st_params, converter):
    with st_sim.STSim(st_params) as sim_obj:
        ptych = sim_obj.ptychograph()
        data = converter.export_data(ptych, st_params)
    assert data.data.dtype == converter.protocol.known_types['float']
    st_obj = data.get_st()
    st_res = st_obj.iter_update(sw_fs=20, ls_pm=3, ls_ri=5,
                                verbose=True, n_iter=10, return_errors=False)
    assert (st_obj.pixel_map != st_res.pixel_map).any()
    assert st_res.pixel_map.dtype == converter.protocol.known_types['float']