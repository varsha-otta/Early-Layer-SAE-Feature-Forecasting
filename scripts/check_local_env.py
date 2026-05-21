"""Verify local Python environment for probe training.

Run after `pip install -r requirements.txt`. Probes are small enough to run on
CPU; this script only checks that the expected libraries import cleanly.
"""
import sys


def check():
    print(f"Python: {sys.version.split()[0]}")
    print()
    failures = []
    expected = ["numpy", "scipy", "sklearn", "torch", "matplotlib", "pandas", "tqdm"]
    for name in expected:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {name:<12} {ver}")
        except ImportError as e:
            failures.append(name)
            print(f"  {name:<12} MISSING ({e})")

    print()
    try:
        import torch
        print(f"  torch CUDA available: {torch.cuda.is_available()} (expected False on this machine)")
        print(f"  torch threads: {torch.get_num_threads()}")
    except ImportError:
        pass

    print()
    if failures:
        print(f"{len(failures)} import failures: {failures}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("Local environment OK. Ready for probe training (Step 4+).")


if __name__ == "__main__":
    check()
