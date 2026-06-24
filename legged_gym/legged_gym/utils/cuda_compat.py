import os
import re
import torch


def _version_tuple(version):
    match = re.match(r"(\d+)\.(\d+)", version)
    if not match:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


def check_cuda_runtime_compat():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    if not torch.cuda.is_available():
        return
    torch_version = _version_tuple(torch.__version__)
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        if capability >= (8, 9) and torch_version <= (1, 10):
            name = torch.cuda.get_device_name(index)
            raise RuntimeError(
                f"Detected {name} with compute capability sm_{capability[0]}{capability[1]}, "
                f"but PyTorch {torch.__version__} was built before RTX 4090/sm_89 support. "
                "Use the hw environment with PyTorch 1.13.1+cu117 or 2.0.1+cu118, "
                "then reinstall isaacgym, rsl_rl and legged_gym in editable mode."
            )
