import os.path
import pytest
import robust_speckle_tracking as rst
import robust_speckle_tracking.simulation as st_sim

@pytest.fixture(params=[{'defoc': 1e2, 'det_dist': 5e5, 'n_frames': 10, 'ap_x': 30},
                        {'defoc': 5e1, 'det_dist': 1e6, 'n_frames': 50, 'ap_x': 20}])
def st_params(request):
    """
    Return a default simulation parameters instance
    """
    return st_sim.parameters(**request.param)

@pytest.fixture(params=['results/test', 'results/test_ideal'])
def sim_data(request):
    """
    Return a path to the simulated speckle tracking data
    """
    return request.param

@pytest.fixture(params=[{'scan_num': 1986, 'roi': (0, 1, 360, 1090), 'defocus': 1.0e-4},
                        {'scan_num': 1740, 'roi': (0, 1, 350, 1065), 'defocus': 1.5e-4}])
def exp_data(request):
    """
    Return a path to the experimental speckle tracking data
    """
    params = {}
    params['path'] = 'results/exp/Scan_{:d}.cxi'.format(request.param['scan_num'])
    params['defocus'] = request.param['defocus']
    params['roi'] = request.param['roi']
    return params

@pytest.fixture(scope='function')
def ini_path():
    """
    Return a path to the experimental speckle tracking data
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
    sim = st_sim.STSim(st_params)
    ptych = sim.ptychograph()
    assert len(ptych.shape) == 3
    assert ptych.shape[0] == st_params.n_frames

@pytest.mark.rst
def test_loader_exp(exp_data):
    assert os.path.isfile(exp_data['path'])
    data_dict = rst.loader()._load(**exp_data)
    for attr in rst.STData.attr_dict:
        assert not data_dict[attr] is None

@pytest.mark.rst
def test_loader_sim(sim_data):
    assert os.path.isdir(sim_data)
    protocol_path = os.path.join(sim_data, 'protocol.ini')
    assert os.path.isfile(protocol_path)
    data_path = os.path.join(sim_data, 'data.cxi')
    assert os.path.isfile(data_path)
    protocol = rst.Protocol.import_ini(protocol_path)
    loader = rst.STLoader(protocol=protocol)
    data_dict = loader._load(data_path)
    for attr in rst.STData.attr_dict:
        assert not data_dict[attr] is None

@pytest.mark.rst
def test_iter_update(sim_data):
    assert os.path.isdir(sim_data)
    protocol_path = os.path.join(sim_data, 'protocol.ini')
    assert os.path.isfile(protocol_path)
    data_path = os.path.join(sim_data, 'data.cxi')
    assert os.path.isfile(data_path)
    protocol = rst.Protocol.import_ini(protocol_path)
    loader = rst.STLoader(protocol=protocol)
    st_data = loader.load(data_path)
    st_obj = st_data.get_last_st()
    pixel_map0 = st_obj.pixel_map.copy()
    st_obj.iter_update(150, ls_pm=2.5, ls_ri=15, verbose=True, n_iter=5)
    assert (st_obj.pixel_map == pixel_map0).all()
