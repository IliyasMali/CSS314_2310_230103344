"""PRACTICE TASK 1 — Row-Major vs. Column-Major Parallel Cache Traversal.

8000 x 8000 float64 matrix (C order = row-major in memory, 512 MB).
  Mode A (Row-Major) : each worker owns a slice of ROWS and sums row by row
                       -> contiguous memory, every 64-byte cache line fully used.
  Mode B (Column-Major / Strided): each worker owns a slice of COLUMNS and sums
                       column by column -> consecutive elements are 64,000 bytes
                       apart, every access touches a new cache line.
numpy releases the GIL inside np.sum, so the threads run truly in parallel.
"""
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
SIZE = 8000
REPEATS = 2


def sum_rows(M, lo, hi):
    s = 0.0
    for i in range(lo, hi):
        s += np.sum(M[i, :])          # contiguous row
    return s


def sum_cols(M, lo, hi):
    s = 0.0
    for j in range(lo, hi):
        s += np.sum(M[:, j])          # strided column (stride = 8000 * 8 bytes)
    return s


def run(fn, M, workers, pool):
    bounds = np.linspace(0, SIZE, workers + 1, dtype=int)
    futs = [pool.submit(fn, M, int(bounds[k]), int(bounds[k + 1])) for k in range(workers)]
    return sum(f.result() for f in futs)


def bench(fn, M, workers, pool):
    best, total = float("inf"), None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        total = run(fn, M, workers, pool)
        best = min(best, time.perf_counter() - t0)
    return best, total


def main():
    start_log("task1_output.txt")
    header("PRACTICE TASK 1 — Row-Major vs. Column-Major Parallel Cache Traversal")

    workers = physical_cores()
    M = np.random.default_rng(STUDENT_ID).random((SIZE, SIZE))
    print(f"Student ID: {STUDENT_ID} | Logical cores: {logical_cores()} | "
          f"Worker threads (physical cores): {workers}")
    print(f"Matrix: {SIZE} x {SIZE} float64 ({M.nbytes / 1e6:.0f} MB), C-order (row-major)\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        t_a, s_a = bench(sum_rows, M, workers, pool)
        t_b, s_b = bench(sum_cols, M, workers, pool)

    assert abs(s_a - s_b) < 1e-6 * abs(s_a), "sums differ!"
    print(f"Mode A (Row-Major, linear)      : {t_a * 1000:10.2f} ms   sum = {s_a:.4f}")
    print(f"Mode B (Column-Major, strided)  : {t_b * 1000:10.2f} ms   sum = {s_b:.4f}")
    print(f"\nCache penalty factor (B / A)    : {t_b / t_a:.2f}x")
    print("\nConclusion: Mode A reads 8 consecutive doubles per 64-byte cache line and the "
          "hardware prefetcher streams ahead. Mode B jumps 64,000 bytes per element, so "
          "each load touches a new cache line and a new 4 KB page (TLB miss).")
    print(f"Here the strided traversal cost {t_b / t_a:.2f}x the linear one. Part of the "
          "penalty is hidden because the 8000 lines touched by one column (512 KB) can "
          "stay in L2 and get reused by the next column; the larger the L2/SLC, the "
          "smaller the gap.")


if __name__ == "__main__":
    main()
