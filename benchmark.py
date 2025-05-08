import numpy as np
import time
from trilat import trilat, trilat_A
from previous import trilat_linear, trilat_adachi

SOLVERS = [
    ('linear', trilat_linear),
    ('adachi', trilat_adachi),
    ('proposed_A', trilat_A),
    ('proposed', trilat)
]

def run_once(s, d2, solver):
    return solver(s, d2)


def benchmark(m=10, dim=3, reps=5):
    # prepare random data
    x_true = np.random.randn(dim)
    s = np.random.randn(dim, m)
    d2 = np.sum((s - x_true[:,None])**2, axis=0)

    results = {}
    for name, fn in SOLVERS:
        # warm up
        fn(s, d2)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(s, d2)
            times.append(time.perf_counter() - t0)
        results[name] = np.median(times)
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--reps', type=int, default=10)
    args = parser.parse_args()

    res = benchmark(args.m, args.dim, args.reps)
    print(f"Benchmark m={args.m}, dim={args.dim}")
    for name, t in res.items():
        print(f"{name:12s}: {t*1e3:.3f} ms")
