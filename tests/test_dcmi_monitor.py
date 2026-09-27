#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import unittest

from model_analyzer.device.npu_device import NPUDevice
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.monitor.dcmi.dcmi_monitor import DCMIMonitor
from model_analyzer.record.types.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.types.gpu_power_usage import GPUPowerUsage
from model_analyzer.record.types.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.types.gpu_utilization import GPUUtilization

from .common import test_result_collector as trc
from .mocks.mock_dcmi import (
    TEST_CHIP_NAME,
    TEST_HBM_TOTAL_MEMORY_MB,
    TEST_HBM_USED_MEMORY_MB,
    TEST_POWER_WATTS,
    TEST_SERIAL_NUMBER,
    TEST_UTILIZATION_RATE,
    MockDCMI,
)

TEST_CARD_ID = 0
TEST_CHIP_ID = 0
TEST_LOGIC_ID = 0


class TestDCMIMonitor(trc.TestResultCollector):
    def setUp(self):
        self.mock_dcmi = MockDCMI()
        self.mock_dcmi.start()

        self._gpus = [
            NPUDevice(
                device_name=TEST_CHIP_NAME,
                device_id=TEST_LOGIC_ID,
                pci_bus_id="0000:C1:00.0",
                device_uuid=f"{TEST_SERIAL_NUMBER}0-0",
                card_id=TEST_CARD_ID,
                chip_id=TEST_CHIP_ID,
            )
        ]

    def tearDown(self):
        self.mock_dcmi.stop()

    def test_record_memory(self):
        frequency = 1
        monitoring_time = 1.1
        metrics = [GPUUsedMemory, GPUFreeMemory]
        dcmi_monitor = DCMIMonitor(self._gpus, frequency, metrics)
        dcmi_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = dcmi_monitor.stop_recording_metrics()

        for record in records:
            self.assertIsInstance(record.device_uuid(), str)
            self.assertIsInstance(record.value(), float)
            self.assertIsInstance(record.timestamp(), int)
            if isinstance(record, GPUUsedMemory):
                self.assertEqual(record.value(), TEST_HBM_USED_MEMORY_MB)
            elif isinstance(record, GPUFreeMemory):
                self.assertEqual(
                    record.value(), TEST_HBM_TOTAL_MEMORY_MB - TEST_HBM_USED_MEMORY_MB
                )

        # At least one sample per metric per monitoring iteration
        self.assertEqual(len(records) % len(metrics), 0)
        self.assertGreaterEqual(len(records), 2 * len(metrics))

        # Timestamps are in nanoseconds; samples are taken once per
        # monitoring interval, so two samples are >= 0.9s apart
        self.assertGreaterEqual(records[-1].timestamp() - records[0].timestamp(), 0.9e9)

        with self.assertRaises(TritonModelAnalyzerException):
            dcmi_monitor.stop_recording_metrics()

        dcmi_monitor.destroy()

    def test_record_power(self):
        frequency = 1
        monitoring_time = 1.1
        metrics = [GPUPowerUsage]
        dcmi_monitor = DCMIMonitor(self._gpus, frequency, metrics)
        dcmi_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = dcmi_monitor.stop_recording_metrics()

        for record in records:
            self.assertIsInstance(record.device_uuid(), str)
            self.assertIsInstance(record.value(), float)
            self.assertEqual(record.value(), TEST_POWER_WATTS)
            self.assertIsInstance(record.timestamp(), int)

        self.assertEqual(len(records) % len(metrics), 0)
        self.assertGreater(len(records), 0)
        self.assertGreaterEqual(records[-1].timestamp() - records[0].timestamp(), 0.9e9)

        dcmi_monitor.destroy()

    def test_record_utilization(self):
        frequency = 1
        monitoring_time = 1.1
        metrics = [GPUUtilization]
        dcmi_monitor = DCMIMonitor(self._gpus, frequency, metrics)
        dcmi_monitor.start_recording_metrics()
        time.sleep(monitoring_time)
        records = dcmi_monitor.stop_recording_metrics()

        for record in records:
            self.assertIsInstance(record.device_uuid(), str)
            self.assertIsInstance(record.value(), float)
            self.assertLessEqual(record.value(), 100)
            self.assertEqual(record.value(), TEST_UTILIZATION_RATE)
            self.assertIsInstance(record.timestamp(), int)

        self.assertEqual(len(records) % len(metrics), 0)
        self.assertGreater(len(records), 0)
        self.assertGreaterEqual(records[-1].timestamp() - records[0].timestamp(), 0.9e9)

        dcmi_monitor.destroy()

    def test_unsupported_metric(self):
        frequency = 1
        metrics = ["UndefinedTag"]
        with self.assertRaises(TritonModelAnalyzerException):
            DCMIMonitor(self._gpus, frequency, metrics)


if __name__ == "__main__":
    unittest.main()
