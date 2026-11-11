"""Run a reduced end-to-end software smoke test for Experiment A.

This executes every regressor family with small meshes and data sizes. The
results are only a software check and must NOT be used as research findings.

Run from the project directory:
    python smoke_test.py
"""
from pathlib import Path

import experiment_a as study

# Keep this separate from the default research configuration.
study.NX = 20
study.NY = 16
study.SAMPLE_SIZES = [16, 32]
study.N_REPEATS = 2
study.FINAL_N = 32
study.N_VALIDATION = 24
study.N_TEST = 24
study.GPR_RESTARTS = 0
study.ANN_MAX_ITER = 250
study.OUTPUT_DIR = Path("experiment_a_smoke_results")

if __name__ == "__main__":
    study.main()
