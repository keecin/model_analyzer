#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NPU device discovery via the Ascend DCMI interface. Mirrors
# model_analyzer/device/gpu_device_factory.py (GPUDeviceFactory).

import logging
import os

import model_analyzer.monitor.dcmi.dcmi_agent as dcmi_agent
import model_analyzer.monitor.dcmi.dcmi_structs as structs
from model_analyzer.constants import LOGGER_NAME
from model_analyzer.device.npu_device import NPUDevice
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

logger = logging.getLogger(LOGGER_NAME)


class NPUDeviceFactory:
    """
    Factory class for creating NPUDevices
    """

    def __init__(self):
        self._devices = []
        self._devices_by_logic_id = {}
        self._devices_by_uuid = {}
        self.init_all_devices()

    @staticmethod
    def is_available():
        """
        Returns
        -------
        bool
            True if DCMI can be initialized and at least one NPU is present
        """
        try:
            structs._dcmiInit()
            dcmi_agent.dcmi_init()
            return len(dcmi_agent.dcmi_get_card_list()) > 0
        except Exception:
            return False

    def init_all_devices(self, dcmiPath=None):
        """
        Create NPUDevice objects for all DCMI visible devices.

        Parameters
        ----------
        dcmiPath : str
            Absolute path to the DCMI shared library
        """

        structs._dcmiInit(dcmiPath)
        dcmi_agent.dcmi_init()

        card_ids = dcmi_agent.dcmi_get_card_list()
        for card_id in card_ids:
            device_num = dcmi_agent.dcmi_get_device_num_in_card(card_id)
            # Card serial number is used as the stable device uuid. Not all
            # platforms report it; fall back to a card/chip based identifier.
            serial_number = None
            try:
                elabel = dcmi_agent.dcmi_get_card_elabel_v2(card_id)
                if elabel.serial_number and elabel.serial_number not in ("NA", ""):
                    serial_number = elabel.serial_number
            except structs.DCMIError:
                logger.warning(
                    f"Could not read the electronic label of NPU card {card_id}."
                )

            for chip_id in range(device_num):
                logic_id = dcmi_agent.dcmi_get_device_logic_id(card_id, chip_id)

                device_uuid = (
                    f"{serial_number}-{chip_id}"
                    if serial_number
                    else f"NPU-card{card_id}-chip{chip_id}"
                )

                chip_info = dcmi_agent.dcmi_get_device_chip_info_v2(card_id, chip_id)
                device_name = chip_info.npu_name or chip_info.chip_name

                try:
                    pci_bus_id = dcmi_agent.dcmi_get_device_pcie_info_v2(
                        card_id, chip_id
                    ).bus_id()
                except structs.DCMIError:
                    pci_bus_id = f"card{card_id}-chip{chip_id}"

                try:
                    hbm_info = dcmi_agent.dcmi_get_device_hbm_info(card_id, chip_id)
                    total_memory = hbm_info.memory_size * 1024 * 1024
                except structs.DCMIError:
                    total_memory = 0

                npu_device = NPUDevice(
                    device_name=device_name,
                    device_id=logic_id,
                    pci_bus_id=pci_bus_id,
                    device_uuid=device_uuid,
                    card_id=card_id,
                    chip_id=chip_id,
                    total_memory=total_memory,
                )

                self._devices.append(npu_device)
                self._devices_by_logic_id[logic_id] = npu_device
                self._devices_by_uuid[device_uuid] = npu_device

        # Keep devices ordered by logical id for deterministic output
        self._devices.sort(key=lambda device: device.device_id())

    def get_device_by_logic_id(self, logic_id):
        """
        Get an NPU device using its logical device id, i.e. the numbering
        used by npu-smi and ASCEND_RT_VISIBLE_DEVICES.

        Returns
        -------
        NPUDevice
            The device associated with this logical id.

        Raises
        ------
        TritonModelAnalyzerException
            If the id is out of bound.
        """

        if logic_id not in self._devices_by_logic_id:
            raise TritonModelAnalyzerException(
                f"NPU with logical device id {logic_id} was not found."
            )
        return self._devices_by_logic_id[logic_id]

    def get_device_by_uuid(self, uuid):
        """
        Get an NPU device using its unique identifier (card serial number).

        Raises
        ------
        TritonModelAnalyzerException
            If the uuid does not exist this exception will be raised.
        """

        if uuid not in self._devices_by_uuid:
            raise TritonModelAnalyzerException(f"NPU UUID {uuid} was not found.")
        return self._devices_by_uuid[uuid]

    def get_visible_npus(self):
        """
        Returns
        -------
        list of NPUDevice
            The devices visible to this process, as determined by the
            ASCEND_RT_VISIBLE_DEVICES environment variable. All devices
            are returned when the variable is unset.
        """

        visible_devices_env = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")

        if not visible_devices_env.strip():
            return list(self._devices)

        visible_devices = []
        for token in visible_devices_env.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                logic_id = int(token)
            except ValueError:
                logger.warning(
                    f"Ignoring non-numeric device id '{token}' in ASCEND_RT_VISIBLE_DEVICES."
                )
                continue
            try:
                visible_devices.append(self.get_device_by_logic_id(logic_id))
            except TritonModelAnalyzerException:
                logger.warning(
                    f"Device '{token}' in ASCEND_RT_VISIBLE_DEVICES is not present."
                )
        return visible_devices

    def verify_requested_gpus(self, requested_gpus):
        """
        Creates a list of NPUDevices corresponding to the devices visible
        to this process among the requested ones.

        Parameters
        ----------
        requested_gpus : list of str
            Can either be NPU UUIDs (card serial numbers) or logical device ids

        Returns
        -------
        List of NPUDevices
            list of NPUDevices corresponding to visible devices among requested

        Raises
        ------
        TritonModelAnalyzerException
        """

        visible_npus = self.get_visible_npus()

        if len(requested_gpus) == 1:
            if requested_gpus[0] == "all":
                self._log_npus_used(visible_npus)
                return visible_npus
            elif requested_gpus[0] == "[]":
                logger.info("No NPUs requested")
                return []

        try:
            # Check if each string in the list can be parsed as an int
            requested_logic_ids = list(map(int, requested_gpus))
            requested_devices = []
            for logic_id in requested_logic_ids:
                try:
                    requested_devices.append(self.get_device_by_logic_id(logic_id))
                except TritonModelAnalyzerException:
                    raise TritonModelAnalyzerException(
                        f"Requested NPU with device id : {logic_id}. This NPU is not present."
                    )
        except ValueError:
            # requested_gpus are assumed to be UUIDs (serial numbers)
            requested_devices = [
                self.get_device_by_uuid(uuid) for uuid in requested_gpus
            ]

        # Return the intersection of visible and requested devices.
        available_npus = [
            device for device in visible_npus if device in requested_devices
        ]
        self._log_npus_used(available_npus)

        return available_npus

    def _log_npus_used(self, npus):
        """
        Log the info for the NPUDevices in use
        """

        for npu in npus:
            logger.info(
                f"Using NPU {npu.device_id()} {npu.device_name()} with UUID {npu.device_uuid()}"
            )
