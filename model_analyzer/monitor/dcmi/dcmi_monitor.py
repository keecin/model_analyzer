#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NPU monitor based on the Ascend DCMI interface. Mirrors
# model_analyzer/monitor/dcgm/dcgm_monitor.py (DCGMMonitor) so that it is a
# drop-in replacement for GPU metric collection on the Ascend platform.

import logging
import time

import model_analyzer.monitor.dcmi.dcmi_agent as dcmi_agent
import model_analyzer.monitor.dcmi.dcmi_structs as structs
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.monitor.monitor import Monitor
from model_analyzer.record.types.gpu_free_memory import GPUFreeMemory
from model_analyzer.record.types.gpu_power_usage import GPUPowerUsage
from model_analyzer.record.types.gpu_used_memory import GPUUsedMemory
from model_analyzer.record.types.gpu_utilization import GPUUtilization

logger = logging.getLogger(LOGGER_NAME)


class DCMIMonitor(Monitor):
    """
    Use DCMI to monitor NPU metrics
    """

    # Mapping between Model Analyzer Records and the DCMI queries used
    # to fill them.
    model_analyzer_to_dcmi_metric = {
        GPUUsedMemory: "hbm_used",
        GPUFreeMemory: "hbm_free",
        GPUUtilization: "utilization",
        GPUPowerUsage: "power",
    }

    def __init__(self, gpus, frequency, metrics, dcmiPath=None):
        """
        Parameters
        ----------
        gpus : list of NPUDevice
            The NPUs to be monitored
        frequency : int
            Sampling frequency for the metric
        metrics : list
            List of Record types to monitor
        dcmiPath : str (optional)
            Path to the DCMI shared library
        """

        super().__init__(frequency, metrics)
        structs._dcmiInit(dcmiPath)
        dcmi_agent.dcmi_init()

        self._gpus = gpus

        for metric in metrics:
            if metric not in self.model_analyzer_to_dcmi_metric:
                raise TritonModelAnalyzerException(
                    f"{metric} is not supported by Model Analyzer DCMI Monitor"
                )

        # (timestamp_ns, {device_uuid: {metric_type: value}}) samples
        self._samples = []

    def is_monitoring_connected(self) -> bool:
        # The DCMI interface was initialized in the constructor; a
        # successful card list query confirms we can talk to the driver.
        try:
            return len(dcmi_agent.dcmi_get_card_list()) > 0
        except Exception:
            return False

    def _monitoring_iteration(self):
        """
        Samples all requested metrics for all monitored NPUs.
        Called periodically by the background monitoring thread.
        """

        timestamp = time.time_ns()
        sample = {}
        for gpu in self._gpus:
            try:
                values = self._sample_device(gpu)
                if values:
                    sample[gpu.device_uuid()] = values
            except Exception as e:
                logger.warning(
                    f"Failed to sample NPU metrics for device {gpu.device_uuid()}: {e}"
                )

        self._samples.append((timestamp, sample))

    def _sample_device(self, gpu):
        """
        Queries DCMI for one device, returning only the requested metrics.
        """
        card_id, chip_id = gpu.card_id(), gpu.chip_id()
        values = {}

        if GPUUsedMemory in self._metrics or GPUFreeMemory in self._metrics:
            hbm_info = dcmi_agent.dcmi_get_device_hbm_info(card_id, chip_id)
            # DCMI reports HBM sizes in MB, the same unit Model Analyzer
            # records use.
            if GPUUsedMemory in self._metrics:
                values[GPUUsedMemory] = float(hbm_info.memory_usage)
            if GPUFreeMemory in self._metrics:
                values[GPUFreeMemory] = float(
                    hbm_info.memory_size - hbm_info.memory_usage
                )

        if GPUUtilization in self._metrics:
            # AICore utilization matches the utilization reported by
            # npu-smi's summary table and npu-exporter's
            # npu_chip_info_utilization metric.
            values[GPUUtilization] = float(
                dcmi_agent.dcmi_get_device_utilization_rate(
                    card_id, chip_id, structs.DCMI_UTILIZATION_RATE_AICORE
                )
            )

        if GPUPowerUsage in self._metrics:
            # DCMI power is in units of 0.1 W; records use watts.
            raw_power = dcmi_agent.dcmi_get_device_power_info(card_id, chip_id)
            values[GPUPowerUsage] = raw_power / 10.0

        return values

    def _collect_records(self):
        """
        Converts the collected samples into Model Analyzer Records.
        """

        records = []
        for timestamp, sample in self._samples:
            for device_uuid, values in sample.items():
                for metric_type in self._metrics:
                    if metric_type in values:
                        records.append(
                            metric_type(
                                value=values[metric_type],
                                device_uuid=device_uuid,
                                timestamp=timestamp,
                            )
                        )

        self._samples = []
        return records

    def destroy(self):
        """
        Destroy the DCMIMonitor. This function must be called
        in order to appropriately deallocate the resources.
        """

        super().destroy()
