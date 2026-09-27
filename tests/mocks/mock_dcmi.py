#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
from unittest.mock import MagicMock, patch

from model_analyzer.monitor.dcmi.dcmi_agent import (
    ChipInfo,
    ElabelInfo,
    HbmInfo,
    PciInfo,
)

from .mock_base import MockBase

TEST_NPU_COUNT = 2
TEST_CHIP_NAME = "910B3"
TEST_SERIAL_NUMBER = "10256A25980"
TEST_DEVICE_UUID = f"{TEST_SERIAL_NUMBER}0-0"
TEST_PCI_BUS_ID = "0000:C1:00.0"

# Values returned by the mocked DCMI queries
TEST_HBM_TOTAL_MEMORY_MB = 65536
TEST_HBM_USED_MEMORY_MB = 4096
TEST_UTILIZATION_RATE = 42
TEST_POWER_RAW = 951  # 0.1 W units
TEST_POWER_WATTS = TEST_POWER_RAW / 10.0


def _test_chip_info(card_id: int, device_id: int) -> ChipInfo:
    return ChipInfo(
        chip_type="Ascend",
        chip_name=TEST_CHIP_NAME,
        chip_ver="V1",
        aicore_cnt=20,
        npu_name=TEST_CHIP_NAME,
    )


def _test_pcie_info(card_id: int, device_id: int) -> PciInfo:
    return PciInfo(domain=0, bus=0xC1 + card_id, device=0, func=0)


def _test_elabel(card_id: int) -> ElabelInfo:
    return ElabelInfo(
        product_name="IT21HMDA6_N0",
        model="NA",
        manufacturer="NA",
        serial_number=f"{TEST_SERIAL_NUMBER}{card_id}",
    )


def _test_hbm_info(card_id: int, device_id: int) -> HbmInfo:
    return HbmInfo(
        memory_size=TEST_HBM_TOTAL_MEMORY_MB,
        freq=1600,
        memory_usage=TEST_HBM_USED_MEMORY_MB,
        temp=38,
        bandwidth_util_rate=5,
    )


class MockDCMIAgent:
    """
    Mock of model_analyzer.monitor.dcmi.dcmi_agent, backed by simple
    in-memory state for TEST_NPU_COUNT devices (one chip per card).
    """

    @staticmethod
    def dcmi_init():
        pass

    @staticmethod
    def dcmi_get_card_list() -> List[int]:
        return list(range(TEST_NPU_COUNT))

    @staticmethod
    def dcmi_get_device_num_in_card(card_id):
        return 1

    @staticmethod
    def dcmi_get_device_logic_id(card_id, device_id):
        return card_id

    @staticmethod
    def dcmi_get_card_id_device_id_from_logicid(logic_id):
        return logic_id, 0

    @staticmethod
    def dcmi_get_device_chip_info_v2(card_id, device_id):
        return _test_chip_info(card_id, device_id)

    @staticmethod
    def dcmi_get_device_pcie_info_v2(card_id, device_id):
        return _test_pcie_info(card_id, device_id)

    @staticmethod
    def dcmi_get_card_elabel_v2(card_id):
        return _test_elabel(card_id)

    @staticmethod
    def dcmi_get_device_hbm_info(card_id, device_id):
        return _test_hbm_info(card_id, device_id)

    @staticmethod
    def dcmi_get_device_utilization_rate(card_id, device_id, input_type):
        return TEST_UTILIZATION_RATE

    @staticmethod
    def dcmi_get_device_power_info(card_id, device_id):
        return TEST_POWER_RAW

    @staticmethod
    def dcmi_get_device_temperature(card_id, device_id):
        return 39

    @staticmethod
    def dcmi_get_device_health(card_id, device_id):
        return 0


class MockDCMI(MockBase):
    """
    Mocks the DCMI layer (dcmi_structs initialization and the dcmi_agent
    functions) in the modules that use it.
    """

    def __init__(self):
        super().__init__()
        self._fill_patchers()

    def _fill_patchers(self):
        patchers = self._patchers

        structs_imports_path = [
            "model_analyzer.monitor.dcmi.dcmi_monitor",
            "model_analyzer.device.npu_device_factory",
        ]
        for import_path in structs_imports_path:
            patchers.append(patch(f"{import_path}.structs._dcmiInit", MagicMock()))

        dcmi_agent_imports_path = [
            "model_analyzer.monitor.dcmi.dcmi_monitor",
            "model_analyzer.device.npu_device_factory",
        ]
        for import_path in dcmi_agent_imports_path:
            patchers.append(patch(f"{import_path}.dcmi_agent", MockDCMIAgent))
