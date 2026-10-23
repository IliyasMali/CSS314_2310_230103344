"""PRACTICE TASK 5 — Thread Oversubscription Threshold.

10,000 small CPU-bound tasks: each task hashes a 1 KB string 500 times
(hashlib.pbkdf2_hmac with SHA-256, 500 iterations). pbkdf2_hmac releases the
GIL while hashing, so Python threads really execute on several cores.
The batch is executed with different ThreadPool sizes; the pool is created
inside the timed region, so thread creation + context switching are included.
"""
import time
import hashlib
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
TASKS = 10_000
HASH_ROUNDS = 500
PAYLOAD = (str(STUDENT_ID) * 120)[:1024].encode()      # 1 KB string
SALT = str(STUDENT_ID).encode()
TOLERANCE = 1.05                                        # >5% slower than best = degraded


def task(i):
    return hashlib.pbkdf2_hmac("sha256", PAYLOAD + i.to_bytes(4, "little"), SALT, HASH_ROUNDS)


def run(pool_size):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        digests = list(pool.map(task, range(TASKS), chunksize=1))
    elapsed = time.perf_counter() - t0
    assert len(digests) == TASKS
    return elapsed


def main():
    start_log("task5_output.txt")
    header("PRACTICE TASK 5 — Thread Oversubscription Threshold")

    cores = logical_cores()
    sizes = []
    for s in [1, max(1, cores // 2), cores, cores * 2, cores * 4, cores * 16, 256]:
        if s not in sizes:
            sizes.append(s)
    print(f"{TASKS:,} tasks x (hash 1 KB string {HASH_ROUNDS} times, SHA-256) | "
          f"logical cores = {cores}")
    print(f"Pool sizes: {sizes}\n")

    run(cores)                                         # warm-up
    times = {s: run(s) for s in sizes}
    t1 = times[1]
    best_size = min(times, key=times.get)
    best_t = times[best_size]

    # first pool size after the best one where time is worse than the best
    threshold = None
    after_best = sizes[sizes.index(best_size) + 1:]
    for s in after_best:
        if times[s] > best_t * TOLERANCE:
            threshold = s
            break

    print(f"{'Pool Size':>9} | {'Execution Time (ms)':>19} | {'Speedup vs 1 thread':>19} | Status")
    print("-" * 82)
    for s in sizes:
        t = times[s]
        if s == 1:
            status = "baseline"
        elif s == best_size:
            status = "BEST (saturation point)"
        elif s == threshold:
            status = "<-- DEGRADATION STARTS (context-switch overhead)"
        elif sizes.index(s) < sizes.index(best_size):
            status = "scaling up"
        elif t > best_t * TOLERANCE:
            status = "degraded (oversubscribed)"
        else:
            status = "plateau (oversubscribed, within 5% of best)"
        print(f"{s:>9} | {t * 1000:>19.1f} | {t1 / t:>19.2f} | {status}")

    print()
    print(f"Best pool size: {best_size} threads ({best_t * 1000:.1f} ms, "
          f"{t1 / best_t:.2f}x speedup)")
    if threshold is not None:
        print(f"Execution time starts worsening at pool size = {threshold} "
              f"({times[threshold] / best_t:.2f}x slower than best): more runnable threads "
              "than cores -> the OS time-slices them, paying context switches, cache "
              "thrashing and GIL hand-offs.")
    else:
        print("No pool size was more than 5% slower than the best one: on this machine the "
              "OS scheduler absorbed oversubscription without measurable loss.")


if __name__ == "__main__":
    main()
