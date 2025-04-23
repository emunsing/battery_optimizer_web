from cylp.cy import CyClpSimplex
import time
import numpy as np
import gc

import pathlib


def run_model_n_times(fname, n=10):
    fname = str(fname)

    load_times = []
    compute_times = []

    for i in range(n):
        print(f"Starting {i}")
        load_start = time.time()
        solver = CyClpSimplex()
        solver.readMps(fname)
        load_times.append(time.time() - load_start)
        
        compute_start = time.time()
        solver.primal()
        print("objective:", solver.objectiveValue)
        compute_times.append(time.time() - load_start)
        del solver
        gc.collect()
    
    return np.mean(compute_times)


def run_models():
    output_root = pathlib.Path('mps_files')
    input_fnames = [
        'tmp_full_noindex.csv',
        'tmp_half_noindex.csv',
        'tmp_quarter_noindex.csv',
        'tmp_fivehundred_rows_noindex.csv',
        'tmp_onethousand_noindex.csv',
    ]

    compute_times = {}

    for f in input_fnames:
        compute_times[f] = run_model_n_times(output_root / (pathlib.Path(f).stem + '.mps'))
    print("compute times")
    print(compute_times)
    return compute_times

def test_basis():
    from cylp.cy import CyClpSimplex
    solver = CyClpSimplex()
    solver.readMps("mps_files/tmp_full_noindex_warmstart.mps")
    initial_start = time.time()
    solver.primal()
    initial_end = time.time()
    solver.readMps("mps_files/tmp_full_noindex.mps")
    second_start = time.time()
    solver.primal()
    second_end = time.time()

    print(f"Solved initial solve in {initial_end - initial_start:0.03f}s, second solve in {second_end - second_start:0.03f}s")


if __name__ == "__main__":
    test_basis()