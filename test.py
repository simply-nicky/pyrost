import st_sim
import numpy as np

def inv_geomspace(x0, x1, n):
    y = np.linspace(x0, x1, n)
    return (x1 - x0) / (np.log(x1) - np.log(x0)) * (np.log(y) - np.log(x0)) + x0

def main():
    out_dir = 'results/{var:s}/{var:s}_{val:d}'
    params = st_sim.defaults()
    p_scan = 3e7
    params['det_dist'] = 2e6
    params['alpha'] = -0.01
    params['bar_size'] = 0.1
    params['attenuation'] = 0.3
    params['n_frames'] = 200
    params['verbose'] = True
    params['p0'] = 2e5
    variable = 'bar_size'
    values = np.linspace(10, 500, 20, dtype=np.int)
    for val in values:
        params[variable] = val
        params['p0'] = p_scan / val
        scan = st_sim.STSim(**params)
        scan.ptych_cxi().save(out_dir.format(var=variable, val=val))
        scan.close()

if __name__ == "__main__":
    main()
