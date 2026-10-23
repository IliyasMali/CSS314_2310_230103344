"""PRACTICE TASK 4 — IPC Overhead (CPU-only path).

Passing a float64 array of N elements from the parent to a worker process:
  Path A: multiprocessing.Queue -> the array is PICKLED, pushed through an OS
          pipe, and UNPICKLED (full copy) on the other side.
  Path B: multiprocessing.shared_memory -> both processes map the SAME physical
          pages; only the block name (a few bytes) is sent, the worker wraps
          the buffer in an ndarray without copying (zero-copy).
The worker process is started once (spawn cost excluded); each measurement is
a full round trip: send -> worker gets array and reads it -> acknowledgement.
"""
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
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
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 5


def attach(name):
    try:
        return shared_memory.SharedMemory(name=name, track=False)   # Python 3.13+
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def worker(inbox, outbox):
    while True:
        msg = inbox.get()
        if msg is None:
            break
        kind = msg[0]
        if kind == "pickle":
            arr = msg[1]                                  # already unpickled copy
            outbox.put(float(arr[0] + arr[-1]))
        else:                                             # ("shm", name, n)
            _, name, n = msg
            shm = attach(name)
            arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)  # zero-copy view
            val = float(arr[0] + arr[-1])
            del arr
            shm.close()
            outbox.put(val)


def main():
    start_log("task4_output.txt")
    header("PRACTICE TASK 4 — IPC Overhead: Pickle Queue vs. Zero-Copy Shared Memory")
    print("CPU-only path (no CUDA): parent -> worker process round trip, best of "
          f"{REPEATS}\n")

    inbox, outbox = mp.Queue(), mp.Queue()
    proc = mp.Process(target=worker, args=(inbox, outbox))
    proc.start()
    inbox.put(("pickle", np.zeros(1)))                    # warm-up
    outbox.get()

    rng = np.random.default_rng(STUDENT_ID)
    rows = []
    for n in SIZES:
        data = rng.random(n)
        expected = float(data[0] + data[-1])

        # Path A: pickle through a pipe
        t_a = float("inf")
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            inbox.put(("pickle", data))
            got = outbox.get()
            t_a = min(t_a, time.perf_counter() - t0)
        assert got == expected

        # Path B: shared memory (data placed in the shared block before timing)
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        t_setup0 = time.perf_counter()
        view = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        view[:] = data                                    # one-time producer write
        t_setup = time.perf_counter() - t_setup0
        t_b = float("inf")
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            inbox.put(("shm", shm.name, n))
            got = outbox.get()
            t_b = min(t_b, time.perf_counter() - t0)
        assert got == expected
        del view
        shm.close()
        shm.unlink()
        rows.append((n, data.nbytes, t_a, t_b, t_setup))

    inbox.put(None)
    proc.join()

    print(f"{'N':>12} | {'Size (MB)':>9} | {'Path A pickle/pipe (ms)':>23} | "
          f"{'Path B shared mem (ms)':>22} | {'A / B':>8}")
    print("-" * 87)
    for n, nbytes, t_a, t_b, _ in rows:
        print(f"{n:>12,} | {nbytes / 1e6:>9.1f} | {t_a * 1000:>23.3f} | "
              f"{t_b * 1000:>22.3f} | {t_a / t_b:>7.1f}x")
    print("\nOne-time cost to write the data into shared memory (producer side):")
    for n, _, _, _, t_setup in rows:
        print(f"  N = {n:>10,}: {t_setup * 1000:.3f} ms")
    print("\nConclusion: Path A grows linearly with N (serialize + copy through the "
          "kernel pipe + deserialize). Path B stays almost constant because only the "
          "block name crosses the process boundary; the pages are shared, not copied.")


if __name__ == "__main__":
    main()
