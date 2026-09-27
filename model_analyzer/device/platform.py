#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Platform abstraction layer. Provides platform-neutral helpers that select
# between the NVIDIA (DCGM/numba.cuda) and Ascend (DCMI) implementations.
#
# The platform can be forced with the MODEL_ANALYZER_PLATFORM environment
# variable ("nvidia" or "ascend"); otherwise it is auto-detected by the
# presence of the Ascend DCMI library.

import os
from functools import lru_cache

from model_analyzer.record.types.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.types.gpu_power_usage import GPUPowerUsage
from model_analyzer.record.types.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.types.gpu_utilization import GPUUtilization

# The device metric Record types supported by the platform monitors
# (DCGMMonitor on NVIDIA, DCMIMonitor on Ascend).
GPU_METRIC_RECORD_TYPES = frozenset(
    {GPUUsedMemory, GPUFreeMemory, GPUUtilization, GPUPowerUsage}
)


@lru_cache(maxsize=1)
def is_ascend_platform() -> bool:
    """
    Returns True when running on the Ascend platform.

    The MODEL_ANALYZER_PLATFORM environment variable takes precedence over
    auto-detection, which checks for the presence of libdcmi.so.
    """

    platform_env = os.environ.get("MODEL_ANALYZER_PLATFORM", "auto").lower()
    if platform_env == "ascend":
        return True
    elif platform_env in ("nvidia", "cuda"):
        return False

    try:
        import model_analyzer.monitor.dcmi.dcmi_structs as dcmi_structs

        return dcmi_structs._dcmiIsLoadable()
    except Exception:
        return False


def accelerator_is_available() -> bool:
    """
    Platform-neutral replacement for numba.cuda.is_available().
    Returns True if at least one accelerator (GPU or NPU) is usable.
    """

    if is_ascend_platform():
        from model_analyzer.device.npu_device_factory import NPUDeviceFactory

        return NPUDeviceFactory.is_available()

    import numba.cuda

    return numba.cuda.is_available()


def get_device_factory():
    """
    Returns the device factory for the current platform:
    NPUDeviceFactory on Ascend, GPUDeviceFactory on NVIDIA.
    """

    if is_ascend_platform():
        from model_analyzer.device.npu_device_factory import NPUDeviceFactory

        return NPUDeviceFactory()

    from model_analyzer.device.gpu_device_factory import GPUDeviceFactory

    return GPUDeviceFactory()


def get_device_info(device, index) -> dict:
    """
    Returns the name and total memory (in bytes) of the given device,
    as used by the MetricsManager analyzer state.

    Parameters
    ----------
    device : Device
        A GPUDevice or NPUDevice
    index : int
        Position of the device in the visible devices list. Used to
        index into numba.cuda devices on the NVIDIA path.
    """

    from model_analyzer.device.npu_device import NPUDevice

    if isinstance(device, NPUDevice):
        return {
            "name": device.device_name(),
            "total_memory": device.total_memory(),
        }

    import numba.cuda

    cuda_device = numba.cuda.list_devices()[index]
    device_info = {"name": str(cuda_device.name, encoding="utf-8")}
    with cuda_device:
        device_info["total_memory"] = (
            numba.cuda.current_context().get_memory_info().total
        )
    return device_info


def device_visibility_env(gpus) -> dict:
    """
    Returns the environment variables that restrict a process to the
    given devices. On NVIDIA this is CUDA_VISIBLE_DEVICES with GPU UUIDs;
    on Ascend this is ASCEND_RT_VISIBLE_DEVICES with logical device ids.
    """

    if is_ascend_platform():
        return {
            "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(gpu.device_id()) for gpu in gpus)
        }

    return {"CUDA_VISIBLE_DEVICES": ",".join(gpu.device_uuid() for gpu in gpus)}
