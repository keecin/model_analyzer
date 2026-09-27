#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from model_analyzer.device.npu_device_factory import NPUDeviceFactory
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

from .common import test_result_collector as trc
from .mocks.mock_dcmi import (
    TEST_CHIP_NAME,
    TEST_HBM_TOTAL_MEMORY_MB,
    TEST_NPU_COUNT,
    TEST_PCI_BUS_ID,
    TEST_SERIAL_NUMBER,
    MockDCMI,
)


class TestNPUDeviceFactory(trc.TestResultCollector):
    def setUp(self):
        self.mock_dcmi = MockDCMI()
        self.mock_dcmi.start()

    def tearDown(self):
        self.mock_dcmi.stop()

    def test_init_all_devices(self):
        factory = NPUDeviceFactory()

        devices = factory.get_visible_npus()
        self.assertEqual(len(devices), TEST_NPU_COUNT)

        for i, device in enumerate(devices):
            self.assertEqual(device.device_id(), i)
            self.assertEqual(device.device_name(), TEST_CHIP_NAME)
            self.assertEqual(device.card_id(), i)
            self.assertEqual(device.chip_id(), 0)
            self.assertEqual(device.device_uuid(), f"{TEST_SERIAL_NUMBER}{i}-0")
            self.assertEqual(
                device.total_memory(), TEST_HBM_TOTAL_MEMORY_MB * 1024 * 1024
            )

        # PCI bus ids are unique and well formatted
        bus_ids = [device.pci_bus_id() for device in devices]
        self.assertEqual(len(set(bus_ids)), TEST_NPU_COUNT)
        self.assertEqual(bus_ids[0], TEST_PCI_BUS_ID)

    def test_get_device_by_logic_id(self):
        factory = NPUDeviceFactory()

        device = factory.get_device_by_logic_id(0)
        self.assertEqual(device.device_uuid(), f"{TEST_SERIAL_NUMBER}0-0")

        with self.assertRaises(TritonModelAnalyzerException):
            factory.get_device_by_logic_id(TEST_NPU_COUNT + 10)

    def test_get_device_by_uuid(self):
        factory = NPUDeviceFactory()

        uuid = f"{TEST_SERIAL_NUMBER}1-0"
        device = factory.get_device_by_uuid(uuid)
        self.assertEqual(device.device_id(), 1)

        with self.assertRaises(TritonModelAnalyzerException):
            factory.get_device_by_uuid("unknown-uuid")

    def test_verify_requested_gpus_all(self):
        factory = NPUDeviceFactory()

        devices = factory.verify_requested_gpus(["all"])
        self.assertEqual(len(devices), TEST_NPU_COUNT)

    def test_verify_requested_gpus_none(self):
        factory = NPUDeviceFactory()

        devices = factory.verify_requested_gpus(["[]"])
        self.assertEqual(len(devices), 0)

    def test_verify_requested_gpus_by_id(self):
        factory = NPUDeviceFactory()

        devices = factory.verify_requested_gpus(["1"])
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].device_id(), 1)

        with self.assertRaises(TritonModelAnalyzerException):
            factory.verify_requested_gpus([str(TEST_NPU_COUNT + 10)])

    def test_verify_requested_gpus_by_uuid(self):
        factory = NPUDeviceFactory()

        devices = factory.verify_requested_gpus([f"{TEST_SERIAL_NUMBER}0-0"])
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].device_id(), 0)

    def test_visible_npus_env_var(self):
        factory = NPUDeviceFactory()

        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "1"}):
            visible = factory.get_visible_npus()
            self.assertEqual(len(visible), 1)
            self.assertEqual(visible[0].device_id(), 1)

            # Requesting a device that is not visible yields an empty list
            devices = factory.verify_requested_gpus(["0"])
            self.assertEqual(len(devices), 0)

        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}):
            visible = factory.get_visible_npus()
            self.assertEqual(len(visible), TEST_NPU_COUNT)

        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": ""}):
            visible = factory.get_visible_npus()
            self.assertEqual(len(visible), TEST_NPU_COUNT)


if __name__ == "__main__":
    unittest.main()
