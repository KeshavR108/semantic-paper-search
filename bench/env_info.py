"""Collect the run environment so every result file says where it came from."""

import os
import platform
import subprocess
import sys

import numpy as np


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_model():
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def ram_gb():
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        return round(int(out) / 2**30, 1) if out else None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError):
        return None


def git_commit():
    commit = _run(["git", "rev-parse", "HEAD"])
    if commit and _run(["git", "status", "--porcelain", "--untracked-files=no"]):
        commit += "-dirty"
    return commit


def environment(extra_packages=()):
    env = {
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "ram_gb": ram_gb(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": git_commit(),
    }
    for name in extra_packages:
        try:
            from importlib.metadata import version
            env[name] = version(name)
        except Exception:
            env[name] = None
    return env
