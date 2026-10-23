"""PRACTICE TASK 3 — Prefix Scan (Step Efficiency vs. Work Efficiency).

Inclusive prefix sum of 2,000,000 integers in [1, 10].
  Serial  : output[i] = output[i-1] + input[i]   (np.cumsum = this exact loop in C, 1 thread)
  Parallel: 2-pass chunked scan over worker threads
            pass 1 - each thread scans its own chunk and reports the chunk total
            reduce - exclusive scan of chunk totals -> offset for every chunk
            pass 2 - each thread adds its offset to its chunk
"""
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
N = 2_000_000
REPEATS = 5


def serial_scan(a):
    return np.cumsum(a)


def serial_scan_loop(a):
    """Literal formula, used once as a reference for the textbook definition."""
    out = [0] * len(a)
    acc = 0
    for i, v in enumerate(a.tolist()):
        acc += v
        out[i] = acc
    return out


def parallel_scan(a, workers, pool):
    out = np.empty_like(a)
    bounds = np.linspace(0, len(a), workers + 1, dtype=int)
    chunks = [(int(bounds[k]), int(bounds[k + 1])) for k in range(workers)]

    def local_scan(lo_hi):                       # pass 1
        lo, hi = lo_hi
        np.cumsum(a[lo:hi], out=out[lo:hi])
        return out[hi - 1] if hi > lo else 0

    totals = list(pool.map(local_scan, chunks))
    offsets = np.concatenate(([0], np.cumsum(totals)[:-1]))   # offset reduction

    def apply_offset(args):                      # pass 2
        (lo, hi), off = args
        if off:
            out[lo:hi] += off

    list(pool.map(apply_offset, zip(chunks, offsets)))
    return out


def main():
    start_log("task3_output.txt")
    header("PRACTICE TASK 3 — Prefix Scan (Step Efficiency vs. Work Efficiency)")

    workers = physical_cores()
    a = np.random.default_rng(STUDENT_ID).integers(1, 11, size=N, dtype=np.int64)
    print(f"Array: {N:,} int64 values in [1, 10] | parallel workers = {workers}\n")

    t0 = time.perf_counter()
    loop_ref = serial_scan_loop(a)
    t_loop = time.perf_counter() - t0

    t_ser = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        ser = serial_scan(a)
        t_ser = min(t_ser, time.perf_counter() - t0)

    t_par = float("inf")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parallel_scan(a, workers, pool)          # warm-up
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            par = parallel_scan(a, workers, pool)
            t_par = min(t_par, time.perf_counter() - t0)

    try:
        assert par[-1] == ser[-1], "last element mismatch"
        assert ser[-1] == loop_ref[-1], "numpy scan differs from loop formula"
        assert np.array_equal(par, ser), "full arrays differ"
        status = "PASS"
    except AssertionError as e:
        status = f"FAIL ({e})"

    print(f"(reference) pure-Python loop of the formula: {t_loop * 1000:.2f} ms\n")
    print(f"{'Serial Time (ms)':>17} | {'Parallel Time (ms)':>18} | {'Speedup':>8} | Verification Status")
    print("-" * 72)
    print(f"{t_ser * 1000:>17.3f} | {t_par * 1000:>18.3f} | {t_ser / t_par:>8.2f} | {status}")
    print(f"\nserial_output[-1]   = {int(ser[-1]):,}")
    print(f"parallel_output[-1] = {int(par[-1]):,}")
    print("\nNote: the 2-pass scan does ~2N additions (work-inefficient vs N serial) "
          "but splits them over the cores, so speedup is bounded by ~workers/2 and by "
          "memory bandwidth.")

    if not status.startswith("PASS"):
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
