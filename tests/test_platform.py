#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import MagicMock, patch

from model_analyzer.device.npu_device import NPUDevice
from model_analyzer.device.platform import (
    accelerator_is_available,
    device_visibility_env,
    get_device_factory,
    is_ascend_platform,
)

from .common import test_result_collector as trc

TEST_UUID = "10256A259809-0"


def make_test_npu(device_id=0):
    return NPUDevice(
        device_name="910B3",
        device_id=device_id,
        pci_bus_id="0000:C1:00.0",
        device_uuid=f"{TEST_UUID[:-2]}{device_id}-0",
        card_id=device_id,
        chip_id=0,
    )


class TestPlatform(trc.TestResultCollector):
    def setUp(self):
        # Reset the lru_cache so env var changes take effect
        is_ascend_platform.cache_clear()

    def tearDown(self):
        is_ascend_platform.cache_clear()

    def test_is_ascend_platform_env_override(self):
        with patch.dict(os.environ, {"MODEL_ANALYZER_PLATFORM": "ascend"}):
            is_ascend_platform.cache_clear()
            self.assertTrue(is_ascend_platform())

        with patch.dict(os.environ, {"MODEL_ANALYZER_PLATFORM": "nvidia"}):
            is_ascend_platform.cache_clear()
            self.assertFalse(is_ascend_platform())

    def test_accelerator_is_available_ascend(self):
        with (
            patch.dict(os.environ, {"MODEL_ANALYZER_PLATFORM": "ascend"}),
            patch(
                "model_analyzer.device.npu_device_factory.NPUDeviceFactory.is_available",
                MagicMock(return_value=True),
            ),
        ):
            is_ascend_platform.cache_clear()
            self.assertTrue(accelerator_is_available())

    def test_accelerator_is_available_nvidia(self):
        with patch.dict(os.environ, {"MODEL_ANALYZER_PLATFORM": "nvidia"}):
            is_ascend_platform.cache_clear()
            with patch("numba.cuda.is_available", MagicMock(return_value=False)):
                self.assertFalse(accelerator_is_available())

    def test_get_device_factory(self):
        with patch.dict(os.environ, {"MODEL_ANALYZER_PLATFORM": "ascend"}):
            is_ascend_platform.cache_clear()
            with (
                patch(
                    "model_analyzer.device.npu_device_factory.NPUDeviceFactory",
                    MagicMock(),
                ),
            ):
                factory = get_device_factory()
                self.assertIsNotNone(factory)

    def test_device_visibility_env_ascend(self):
        npus = [make_test_npu(0), make_test_npu(1)]

        with patch.dict(os.environ, {"MODEL_ANALYZER_PLATFORM": "ascend"}):
            is_ascend_platform.cache_clear()
            env = device_visibility_env(npus)
            self.assertEqual(env, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})

    def test_device_visibility_env_nvidia(self):
        from model_analyzer.device.gpu_device import GPUDevice

        gpus = [GPUDevice("test", i, "0000:C1:00.0", f"GPU-{i}") for i in range(2)]

        with patch.dict(os.environ, {"MODEL_ANALYZER_PLATFORM": "nvidia"}):
            is_ascend_platform.cache_clear()
            env = device_visibility_env(gpus)
            self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "GPU-0,GPU-1"})

    def test_get_device_info_ascend(self):
        from model_analyzer.device.platform import get_device_info

        npu = make_test_npu(0)
        info = get_device_info(npu, 0)
        self.assertEqual(info["name"], "910B3")
        self.assertEqual(info["total_memory"], npu.total_memory())


if __name__ == "__main__":
    unittest.main()
