"""TASK 2 — Multi-Core Scaling vs. Amdahl's Law.

Array summation split into chunks across worker threads.
numpy releases the GIL inside np.sum, so threads really run in parallel.
"""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from common import STUDENT_ID, start_log, header, logical_cores

N = 20_000_000
P = 0.95          # parallel fraction for Amdahl's Law
REPEATS = 5       # best-of-N timing to reduce noise


def amdahl(p_cores, P=P):
    return 1.0 / ((1.0 - P) + P / p_cores)


def parallel_sum(data, workers, pool):
    chunks = np.array_split(data, workers)            # views, no copy
    partial = list(pool.map(np.sum, chunks))
    return float(sum(partial))


def main():
    start_log("task2_output.txt")
    header("TASK 2 — Multi-Core Scaling vs. Amdahl's Law")

    data = np.random.default_rng(STUDENT_ID).random(N)
    reference = float(np.sum(data))

    max_cores = logical_cores()
    counts = []
    for c in [1, 2, 4, max_cores, max_cores * 2]:
        if c not in counts:
            counts.append(c)

    results = []
    for w in counts:
        with ThreadPoolExecutor(max_workers=w) as pool:
            parallel_sum(data, w, pool)               # warm-up (threads start)
            best = float("inf")
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                s = parallel_sum(data, w, pool)
                best = min(best, time.perf_counter() - t0)
        assert abs(s - reference) < 1e-6 * abs(reference), "wrong sum!"
        results.append((w, best))

    t1 = results[0][1]
    print(f"Array: {N:,} float64 | P (parallel fraction) = {P} | best of {REPEATS} runs\n")
    print(f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}")
    print("-" * 57)
    for w, t in results:
        print(f"{w:>6} | {t * 1000:>14.3f} | {t1 / t:>8.2f} | {amdahl(w):>18.2f}")
    print(f"\nAmdahl limit for infinite cores: 1/(1-P) = {1 / (1 - P):.1f}x")
    print("Note: summation is memory-bound, so real speedup saturates below Amdahl's bound "
          "once RAM bandwidth is exhausted; workers > cores add only scheduling overhead.")


if __name__ == "__main__":
    main()
