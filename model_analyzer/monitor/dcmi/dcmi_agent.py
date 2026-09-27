#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Python bindings for the device-query subset of the Ascend DCMI library.
# Mirrors model_analyzer/monitor/dcgm/dcgm_agent.py.
#
# All functions raise DCMIError on non-zero return codes.

from ctypes import byref, c_int, c_uint
from typing import List, NamedTuple, Tuple

import model_analyzer.monitor.dcmi.dcmi_structs as structs


def _bytes_to_str(byte_array):
    return bytes(byte_array).split(b"\0", 1)[0].decode("utf-8", errors="replace")


class ChipInfo(NamedTuple):
    chip_type: str
    chip_name: str
    chip_ver: str
    aicore_cnt: int
    npu_name: str


class PciInfo(NamedTuple):
    domain: int
    bus: int
    device: int
    func: int

    def bus_id(self) -> str:
        """
        Returns the PCI bus id in the same format used by npu-smi,
        e.g. "0000:C1:00.0".
        """
        return f"{self.domain & 0xFFFFFFFF:04X}:{self.bus:02X}:{self.device:02X}.{self.func}"


class ElabelInfo(NamedTuple):
    product_name: str
    model: str
    manufacturer: str
    serial_number: str


class HbmInfo(NamedTuple):
    memory_size: int  # MB
    freq: int
    memory_usage: int  # MB
    temp: int
    bandwidth_util_rate: int


dcmiFP = structs._dcmiGetFunctionPointer


def dcmi_init():
    """
    Initializes the DCMI interface. Must be called once before any other
    DCMI function. Safe to call multiple times.
    """
    fn = dcmiFP("dcmi_init")
    ret = fn()
    structs._dcmiCheckReturn(ret, "dcmi_init")
    return ret


def dcmi_get_card_list() -> List[int]:
    """
    Returns the list of available NPU card ids.
    """
    card_num = c_int(0)
    card_list = (c_int * structs.DCMI_MAX_CARD_NUM)()
    fn = dcmiFP("dcmi_get_card_list")
    ret = fn(byref(card_num), card_list, structs.DCMI_MAX_CARD_NUM)
    structs._dcmiCheckReturn(ret, "dcmi_get_card_list")
    return list(card_list[: card_num.value])


def dcmi_get_device_num_in_card(card_id: int) -> int:
    """
    Returns the number of chips (devices) on the given card.
    """
    device_num = c_int(0)
    fn = dcmiFP("dcmi_get_device_num_in_card")
    ret = fn(card_id, byref(device_num))
    structs._dcmiCheckReturn(ret, "dcmi_get_device_num_in_card")
    return device_num.value


def dcmi_get_device_logic_id(card_id: int, device_id: int) -> int:
    """
    Returns the logical device id for a (card_id, device_id) pair.
    The logical id matches the numbering used by npu-smi and the
    ASCEND_RT_VISIBLE_DEVICES environment variable.
    """
    logic_id = c_int(0)
    fn = dcmiFP("dcmi_get_device_logic_id")
    ret = fn(byref(logic_id), card_id, device_id)
    structs._dcmiCheckReturn(ret, "dcmi_get_device_logic_id")
    return logic_id.value


def dcmi_get_card_id_device_id_from_logicid(logic_id: int) -> Tuple[int, int]:
    """
    Returns the (card_id, device_id) pair for a logical device id.
    """
    card_id, device_id = c_int(0), c_int(0)
    fn = dcmiFP("dcmi_get_card_id_device_id_from_logicid")
    ret = fn(byref(card_id), byref(device_id), logic_id)
    structs._dcmiCheckReturn(ret, "dcmi_get_card_id_device_id_from_logicid")
    return card_id.value, device_id.value


def dcmi_get_device_chip_info_v2(card_id: int, device_id: int) -> ChipInfo:
    """
    Returns chip information (type/name/version/aicore count) for a device.
    """
    chip_info = structs.dcmi_chip_info_v2()
    fn = dcmiFP("dcmi_get_device_chip_info_v2")
    ret = fn(card_id, device_id, byref(chip_info))
    structs._dcmiCheckReturn(ret, "dcmi_get_device_chip_info_v2")
    return ChipInfo(
        chip_type=_bytes_to_str(chip_info.chip_type),
        chip_name=_bytes_to_str(chip_info.chip_name),
        chip_ver=_bytes_to_str(chip_info.chip_ver),
        aicore_cnt=chip_info.aicore_cnt,
        npu_name=_bytes_to_str(chip_info.npu_name),
    )


def dcmi_get_device_pcie_info_v2(card_id: int, device_id: int) -> PciInfo:
    """
    Returns PCIe topology information for a device.
    """
    pcie_info = structs.dcmi_pcie_info_all()
    fn = dcmiFP("dcmi_get_device_pcie_info_v2")
    ret = fn(card_id, device_id, byref(pcie_info))
    structs._dcmiCheckReturn(ret, "dcmi_get_device_pcie_info_v2")
    return PciInfo(
        domain=pcie_info.domain,
        bus=pcie_info.bdf_busid,
        device=pcie_info.bdf_deviceid,
        func=pcie_info.bdf_funcid,
    )


def dcmi_get_card_elabel_v2(card_id: int) -> ElabelInfo:
    """
    Returns the electronic label (product name, serial number, ...) of a card.
    The serial number is used as the stable unique identifier for a device.
    """
    elabel = structs.dcmi_elabel_info()
    fn = dcmiFP("dcmi_get_card_elabel_v2")
    ret = fn(card_id, byref(elabel))
    structs._dcmiCheckReturn(ret, "dcmi_get_card_elabel_v2")
    return ElabelInfo(
        product_name=elabel.product_name.decode("utf-8", errors="replace"),
        model=elabel.model.decode("utf-8", errors="replace"),
        manufacturer=elabel.manufacturer.decode("utf-8", errors="replace"),
        serial_number=elabel.serial_number.decode("utf-8", errors="replace"),
    )


def dcmi_get_device_hbm_info(card_id: int, device_id: int) -> HbmInfo:
    """
    Returns the HBM memory info of a device. Sizes are in MB.
    """
    hbm_info = structs.dcmi_hbm_info()
    fn = dcmiFP("dcmi_get_device_hbm_info")
    ret = fn(card_id, device_id, byref(hbm_info))
    structs._dcmiCheckReturn(ret, "dcmi_get_device_hbm_info")
    return HbmInfo(
        memory_size=hbm_info.memory_size,
        freq=hbm_info.freq,
        memory_usage=hbm_info.memory_usage,
        temp=hbm_info.temp,
        bandwidth_util_rate=hbm_info.bandwith_util_rate,
    )


def dcmi_get_device_utilization_rate(
    card_id: int, device_id: int, input_type: int
) -> int:
    """
    Returns the utilization rate (0-100) of a device for the given
    input type (see DCMI_UTILIZATION_RATE_* constants).
    """
    rate = c_uint(0)
    fn = dcmiFP("dcmi_get_device_utilization_rate")
    ret = fn(card_id, device_id, input_type, byref(rate))
    structs._dcmiCheckReturn(ret, "dcmi_get_device_utilization_rate")
    return rate.value


def dcmi_get_device_power_info(card_id: int, device_id: int) -> int:
    """
    Returns the real-time power of a device. The returned value is in
    units of 0.1 W, i.e. 951 means 95.1 W.
    """
    power = c_int(0)
    fn = dcmiFP("dcmi_get_device_power_info")
    ret = fn(card_id, device_id, byref(power))
    structs._dcmiCheckReturn(ret, "dcmi_get_device_power_info")
    return power.value


def dcmi_get_device_temperature(card_id: int, device_id: int) -> int:
    """
    Returns the temperature of a device in degrees Celsius.
    """
    temperature = c_int(0)
    fn = dcmiFP("dcmi_get_device_temperature")
    ret = fn(card_id, device_id, byref(temperature))
    structs._dcmiCheckReturn(ret, "dcmi_get_device_temperature")
    return temperature.value


def dcmi_get_device_health(card_id: int, device_id: int) -> int:
    """
    Returns the health status of a device. 0 means healthy.
    """
    health = c_uint(0)
    fn = dcmiFP("dcmi_get_device_health")
    ret = fn(card_id, device_id, byref(health))
    structs._dcmiCheckReturn(ret, "dcmi_get_device_health")
    return health.value
