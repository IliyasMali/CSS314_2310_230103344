"""PRACTICE TASK 2 — Pseudo-Random Generator State Bottleneck.

Monte Carlo estimate of pi with 40,000,000 points and 4 threads.
Points are drawn in batches (numpy), the hit test runs outside the lock.
  [1] Shared PRNG     : ONE generator for all threads, every draw guarded by a mutex
                        -> generator state is a serial bottleneck.
  [2] Independent PRNG: each thread owns a generator, seed = Student_ID*10 + thread_index
                        -> no shared state, no locking, threads run in parallel.
"""
import math
import time
import threading
import numpy as np
# ---------------- helpers (student ID, hardware info, output log) ----------------
import os
import sys
import platform
import subprocess

STUDENT_ID = 230103344


class Tee:
    """Duplicates everything printed to the console into taskN_output.txt (UTF-8)."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.console = sys.stdout

    def write(self, s):
        self.console.write(s)
        self.file.write(s)

    def flush(self):
        self.console.flush()
        self.file.flush()


def start_log(filename):
    """Call ONLY inside main(); output file is saved next to this script."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    sys.stdout = Tee(path)


def cpu_model():
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def logical_cores():
    return os.cpu_count() or 1


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return logical_cores()  # fallback if psutil is not installed


def header(task_title):
    print("=" * 64)
    print(task_title)
    print("=" * 64)
    print(f"Student ID     : {STUDENT_ID}")
    print(f"OS             : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform       : {platform.platform()}")
    print(f"CPU model      : {cpu_model()}")
    print(f"Logical cores  : {logical_cores()}  (os.cpu_count())")
    print(f"Physical cores : {physical_cores()}")
    print(f"Python         : {platform.python_version()}")
    print("-" * 64)
# ---------------------------------------------------------------------------------
TOTAL = 40_000_000
THREADS = 4
BATCH = 100_000                       # points drawn per lock acquisition / per call


def count_hits(xy):
    x, y = xy[0], xy[1]
    return int(np.count_nonzero(x * x + y * y <= 1.0))


def shared_prng():
    rng = np.random.default_rng(STUDENT_ID)          # single shared instance
    lock = threading.Lock()
    hits = [0] * THREADS
    per = TOTAL // THREADS

    def worker(k):
        done = 0
        while done < per:
            n = min(BATCH, per - done)
            with lock:                               # mutex around the shared state
                xy = rng.random((2, n))
            hits[k] += count_hits(xy)
            done += n

    return _run(worker, hits)


def independent_prng():
    hits = [0] * THREADS
    per = TOTAL // THREADS

    def worker(k):
        rng = np.random.default_rng(STUDENT_ID * 10 + k)  # private generator
        done = 0
        while done < per:
            n = min(BATCH, per - done)
            xy = rng.random((2, n))                  # no lock needed
            hits[k] += count_hits(xy)
            done += n

    return _run(worker, hits)


def _run(worker, hits):
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    pi = 4.0 * sum(hits) / TOTAL
    return pi, abs(pi - math.pi), elapsed


def main():
    start_log("task2_output.txt")
    header("PRACTICE TASK 2 — Pseudo-Random Generator State Bottleneck")
    print(f"Samples: {TOTAL:,} | Threads: {THREADS} | batch = {BATCH:,} points per draw\n")

    pi_s, err_s, t_s = shared_prng()
    pi_i, err_i, t_i = independent_prng()

    print("Shared PRNG (1 generator + mutex):")
    print(f"  Computed pi    : {pi_s:.8f}")
    print(f"  Absolute error : {err_s:.8f}   (math.pi = {math.pi:.8f})")
    print(f"  Wall time      : {t_s * 1000:.2f} ms\n")
    print("Independent PRNG (seed = Student_ID*10 + thread_index):")
    print(f"  Seeds          : {[STUDENT_ID * 10 + k for k in range(THREADS)]}")
    print(f"  Computed pi    : {pi_i:.8f}")
    print(f"  Absolute error : {err_i:.8f}   (math.pi = {math.pi:.8f})")
    print(f"  Wall time      : {t_i * 1000:.2f} ms\n")
    print(f"Speedup from eliminating generator state locking: {t_s / t_i:.2f}x")


if __name__ == "__main__":
    main()
