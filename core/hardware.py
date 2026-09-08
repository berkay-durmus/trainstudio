"""Hardware detection and live telemetry.

The GPU of the machine the application runs on is used: CUDA → MPS → CPU.
This module must stay importable without torch installed, so the UI can show an
installation warning.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field, asdict
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DeviceInfo:
    kind: str                       # "cuda" | "mps" | "cpu"
    torch_device: str               # "cuda:0" | "mps" | "cpu"
    name: str                       # "NVIDIA RTX 6000 Ada" | "Apple M1 Pro" | ...
    total_memory_gb: float = 0.0    # VRAM on GPU, system RAM on MPS/CPU
    count: int = 1
    capability: str = ""            # CUDA compute capability, e.g. "8.9"
    supports_amp: bool = False      # AMP is only reliable on CUDA
    supports_channels_last: bool = False
    all_devices: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.kind == "cuda":
            return f"{self.name} · {self.total_memory_gb:.0f} GB VRAM"
        if self.kind == "mps":
            return f"{self.name} · Apple GPU (MPS) · {self.total_memory_gb:.0f} GB unified memory"
        return f"{self.name} · CPU"

    def to_dict(self) -> dict:
        return asdict(self)


def _system_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except Exception:
            return 0.0


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    if platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "CPU"


@lru_cache(maxsize=1)
def detect() -> DeviceInfo:
    """Return the best available device. Cached for the lifetime of the process."""
    ram = _system_ram_gb()
    try:
        import torch
    except ImportError:
        return DeviceInfo(
            kind="cpu", torch_device="cpu", name=_cpu_name(),
            total_memory_gb=ram, all_devices=["cpu"],
        )

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            kind="cuda",
            torch_device="cuda:0",
            name=props.name,
            total_memory_gb=props.total_memory / 1024**3,
            count=n,
            capability=f"{props.major}.{props.minor}",
            supports_amp=True,
            supports_channels_last=True,
            all_devices=[f"cuda:{i}" for i in range(n)] + ["cpu"],
        )

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return DeviceInfo(
            kind="mps",
            torch_device="mps",
            name=_cpu_name(),
            total_memory_gb=ram,
            # autocast on MPS is still partial; we deliberately keep AMP off.
            supports_amp=False,
            supports_channels_last=False,
            all_devices=["mps", "cpu"],
        )

    return DeviceInfo(
        kind="cpu", torch_device="cpu", name=_cpu_name(),
        total_memory_gb=ram, all_devices=["cpu"],
    )


def usable_vram_gb(dev: DeviceInfo | None = None) -> float:
    """Memory budget (GB) used when estimating batch size.

    85% of VRAM on CUDA; 50% of unified memory on MPS (the system shares the same
    pool); 30% of RAM on CPU.
    """
    dev = dev or detect()
    if dev.kind == "cuda":
        return max(1.0, dev.total_memory_gb * 0.85)
    if dev.kind == "mps":
        return max(1.0, dev.total_memory_gb * 0.50)
    return max(1.0, dev.total_memory_gb * 0.30)


def recommended_workers() -> int:
    """DataLoader worker count. Forking is expensive on macOS, so we stay modest."""
    cpus = os.cpu_count() or 4
    if platform.system() == "Darwin":
        return min(4, max(0, cpus // 2))
    return min(8, max(2, cpus // 2))


# ─────────────────────────────────────────────────────────────────────────────
# Live telemetry — used when the trainer periodically emits a "system" event
# ─────────────────────────────────────────────────────────────────────────────

_NVML_STATE: dict = {"tried": False, "handle": None, "mod": None}


def _nvml():
    if not _NVML_STATE["tried"]:
        _NVML_STATE["tried"] = True
        try:
            import pynvml

            pynvml.nvmlInit()
            _NVML_STATE["mod"] = pynvml
            _NVML_STATE["handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            _NVML_STATE["mod"] = None
    return _NVML_STATE["mod"], _NVML_STATE["handle"]


def telemetry() -> dict:
    """Instantaneous CPU/RAM/GPU usage. Never raises."""
    out: dict = {}
    try:
        import psutil

        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        out["ram_used_gb"] = vm.used / 1024**3
        out["ram_pct"] = vm.percent
    except Exception:
        pass

    dev = detect()
    if dev.kind == "cuda":
        mod, handle = _nvml()
        if mod is not None and handle is not None:
            try:
                util = mod.nvmlDeviceGetUtilizationRates(handle)
                mem = mod.nvmlDeviceGetMemoryInfo(handle)
                out["gpu_pct"] = float(util.gpu)
                out["gpu_mem_used_gb"] = mem.used / 1024**3
                out["gpu_mem_total_gb"] = mem.total / 1024**3
            except Exception:
                pass
        if "gpu_mem_used_gb" not in out:
            try:
                import torch

                out["gpu_mem_used_gb"] = torch.cuda.memory_reserved() / 1024**3
                out["gpu_mem_total_gb"] = dev.total_memory_gb
            except Exception:
                pass
    elif dev.kind == "mps":
        try:
            import torch

            out["gpu_mem_used_gb"] = torch.mps.current_allocated_memory() / 1024**3
        except Exception:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────


def environment_snapshot() -> dict:
    """Written to the run directory as env.json — for reproducibility."""
    import sys

    snap = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(_system_ram_gb(), 1),
        "device": detect().to_dict(),
        "packages": {},
    }
    for mod in ("torch", "torchvision", "timm", "segmentation_models_pytorch",
                "transformers", "ultralytics", "monai", "albumentations",
                "numpy", "streamlit"):
        try:
            m = __import__(mod)
            snap["packages"][mod] = getattr(m, "__version__", "?")
        except Exception:
            snap["packages"][mod] = None

    if shutil.which("nvidia-smi"):
        snap["nvidia_smi"] = True
    return snap


def missing_packages() -> list[str]:
    """Required packages that are missing, for the installation warning in the UI."""
    required = {
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "albumentations": "albumentations",
        "cv2": "opencv-python-headless",
        "sklearn": "scikit-learn",
    }
    missing = []
    for mod, pip_name in required.items():
        try:
            __import__(mod)
        except Exception:
            missing.append(pip_name)
    return missing


if __name__ == "__main__":
    import json

    d = detect()
    print(d.label)
    print(json.dumps(d.to_dict(), indent=2))
    print("usable vram budget:", round(usable_vram_gb(), 2), "GB")
    print("workers:", recommended_workers())
    print("telemetry:", telemetry())
