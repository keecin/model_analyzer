#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ct.lybindings for the Ascend DCMI (Device Management Interface) library
# (libdcmi.so). This mirrors the structure of the vendored DCGM bindings in
# model_analyzer/monitor/dcgm/ so that the Ascend adaptation keeps the same
# architecture as the NVIDIA path.
#
# The DCMI API is defined by the driver header:
#   /usr/local/Ascend/driver/include/dcmi_interface_api.h

import os
import threading
from ctypes import CDLL, Structure, c_char, c_int, c_ubyte, c_uint, c_ulonglong
from typing import Dict, Optional

# Constants from dcmi_interface_api.h
DCMI_MAX_CARD_NUM = 64  # The system supports up to 64 cards
DCMI_MAX_CHIP_NAME_LEN = 32  # Maximum length of chip name
DCMI_MAX_LENTH = 256  # Maximum length of string

# Utilization rate input types (dcmi_interface_api.h)
DCMI_UTILIZATION_RATE_DDR = 1
DCMI_UTILIZATION_RATE_AICORE = 2
DCMI_UTILIZATION_RATE_AICPU = 3
DCMI_UTILIZATION_RATE_CTRLCPU = 4
DCMI_UTILIZATION_RATE_DDR_BANDWIDTH = 5
DCMI_UTILIZATION_RATE_HBM = 6
DCMI_UTILIZATION_RATE_HBM_BANDWIDTH = 10
DCMI_UTILIZATION_RATE_VECTORCORE = 12
DCMI_UTILIZATION_RATE_NPU = 13
DCMI_UTILIZATION_RATE_AICUBE = 14

# DCMI return codes: 0 is success, negative values are errors.
# -8255 means the queried item is not supported on this platform.
DCMI_SUCCESS = 0
DCMI_ERROR_NOT_SUPPORTED = -8255


class DCMIError(Exception):
    """
    Raised when a DCMI API call returns a non-zero code.
    """

    def __init__(self, ret, function=""):
        self.ret = ret
        self.function = function
        super().__init__(f"DCMI API {function} failed with return code: {ret}")


class dcmi_chip_info_v2(Structure):
    _fields_ = [
        ("chip_type", c_ubyte * DCMI_MAX_CHIP_NAME_LEN),
        ("chip_name", c_ubyte * DCMI_MAX_CHIP_NAME_LEN),
        ("chip_ver", c_ubyte * DCMI_MAX_CHIP_NAME_LEN),
        ("aicore_cnt", c_uint),
        ("npu_name", c_ubyte * DCMI_MAX_CHIP_NAME_LEN),
    ]


class dcmi_pcie_info_all(Structure):
    _fields_ = [
        ("venderid", c_uint),
        ("subvenderid", c_uint),
        ("deviceid", c_uint),
        ("subdeviceid", c_uint),
        ("domain", c_int),
        ("bdf_busid", c_uint),
        ("bdf_deviceid", c_uint),
        ("bdf_funcid", c_uint),
        ("reserve", c_ubyte * 32),
    ]


class dcmi_elabel_info(Structure):
    _fields_ = [
        ("product_name", c_char * DCMI_MAX_LENTH),
        ("model", c_char * DCMI_MAX_LENTH),
        ("manufacturer", c_char * DCMI_MAX_LENTH),
        ("manufacturer_date", c_char * DCMI_MAX_LENTH),
        ("serial_number", c_char * DCMI_MAX_LENTH),
    ]


class dcmi_hbm_info(Structure):
    _fields_ = [
        ("memory_size", c_ulonglong),  # unit: MB
        ("freq", c_uint),
        ("memory_usage", c_ulonglong),  # unit: MB
        ("temp", c_int),
        ("bandwith_util_rate", c_uint),
    ]


# Library loading ##
_libLoadLock = threading.Lock()
_dcmiLib: Optional[CDLL] = None
_dcmiGetFunctionPointer_cache: Dict = dict()

# Candidate locations for libdcmi.so. The DCMI library ships with the Ascend
# driver package, not with the CANN toolkit.
_DCMI_LIB_SEARCH_PATHS = [
    "/usr/local/Ascend/driver/lib64/driver/libdcmi.so",
    "/usr/local/Ascend/driver/lib64/libdcmi.so",
    "/usr/lib64/libdcmi.so",
    "/usr/lib/libdcmi.so",
]


def _findLibDcmi(libDcmiPath=None):
    if libDcmiPath:
        if os.path.isfile(libDcmiPath):
            return libDcmiPath
        candidate = os.path.join(libDcmiPath, "libdcmi.so")
        if os.path.isfile(candidate):
            return candidate
    for candidate in _DCMI_LIB_SEARCH_PATHS:
        if os.path.isfile(candidate):
            return candidate
    return None


def _LoadDcmiLibrary(libDcmiPath=None):
    """
    Load the DCMI shared library if it isn't loaded already.
    Raises DCMIError if the library cannot be found or loaded.
    """
    global _dcmiLib

    if _dcmiLib is None:
        _libLoadLock.acquire()
        try:
            if _dcmiLib is None:
                lib_file = _findLibDcmi(libDcmiPath)
                if lib_file is None:
                    raise DCMIError(-1, "libdcmi.so not found")
                _dcmiLib = CDLL(lib_file)
        finally:
            _libLoadLock.release()

    return _dcmiLib


def _dcmiInit(libDcmiPath=None):
    """
    Loads the DCMI shared library. Must be called (once) before using any
    of the functions in dcmi_agent.
    """
    _LoadDcmiLibrary(libDcmiPath)


def _dcmiIsLoadable():
    """
    Returns True if the DCMI library can be loaded on this system.
    Used for platform detection; never raises.
    """
    try:
        _LoadDcmiLibrary()
        return True
    except Exception:
        return False


def _dcmiGetFunctionPointer(name):
    if name in _dcmiGetFunctionPointer_cache:
        return _dcmiGetFunctionPointer_cache[name]

    _libLoadLock.acquire()
    try:
        if _dcmiLib is None:
            raise DCMIError(-1, f"{name}: DCMI library not initialized")
        try:
            _dcmiGetFunctionPointer_cache[name] = getattr(_dcmiLib, name)
            return _dcmiGetFunctionPointer_cache[name]
        except AttributeError:
            raise DCMIError(-1, f"{name}: function not found in libdcmi.so")
    finally:
        _libLoadLock.release()


def _dcmiCheckReturn(ret, function=""):
    if ret != DCMI_SUCCESS:
        raise DCMIError(ret, function)
    return ret
