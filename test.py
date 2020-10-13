import st_sim
import numpy as np

def inv_geomspace(x0, x1, n):
    y = np.linspace(x0, x1, n)
    return (x1 - x0) / (np.log(x1) - np.log(x0)) * (np.log(y) - np.log(x0)) + x0

def main():
    out_dir = 'results/{var:s}_p{p_scan:.0e}/{var:s}_{val:d}'
    p_scan = 3e6
    sample_span = 9.2
    prm = st_sim.parameters(bar_size=0.5, bar_sigma=0.25, bar_atn=0.2,
                            bulk_atn=0.2, th_s=8e-5,
                            offset=2., defoc=150, alpha=0.06,
                            ap_x=32, x0=0.7, random_dev=0.8)
    xmax = 0.8 * prm.ap_x / prm.focus * prm.defoc
    bsteps = st_sim.bin.barcode_steps(x0=-xmax + prm.offset, br_dx=prm.bar_size, rd=prm.random_dev,
                                      x1=xmax + sample_span - prm.offset)
    variable = 'n_frames'
    nf_arr = np.concatenate(([2, 5], np.linspace(10, 100, 10, dtype=np.int)))
    st_converter = st_sim.STConverter()
    for n_frames in nf_arr:
        prm.n_frames = n_frames
        prm.p0 = p_scan / n_frames
        prm.step_size = sample_span / n_frames
        sim = st_sim.STSim(prm, bsteps=bsteps)
        data = sim.ptychograph()
        st_converter.save_sim(data, sim, out_dir.format(var=variable,
                                                        p_scan=p_scan,
                                                        val=n_frames))
        sim.close()

if __name__ == "__main__":
    main()
