#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from model_analyzer.device.device import Device


class NPUDevice(Device):
    """
    Representing an Ascend NPU device

    The device is uniquely identified by the combination of its DCMI
    card id and chip (device) id within the card. The logical device id
    matches the numbering used by npu-smi and the
    ASCEND_RT_VISIBLE_DEVICES environment variable.
    """

    def __init__(
        self,
        device_name,
        device_id,
        pci_bus_id,
        device_uuid,
        card_id,
        chip_id,
        total_memory=0,
    ):
        """
        Parameters
        ----------
            device_name: str
                Human readable name of the device
            device_id : int
                Logical device id, matches npu-smi numbering
            pci_bus_id : str
                PCI bus id, e.g. "0000:C1:00.0"
            device_uuid : str
                Stable unique identifier (card serial number) for the device
            card_id : int
                DCMI card id of the card this device belongs to
            chip_id : int
                DCMI device (chip) id within the card
            total_memory : int
                Total HBM memory of the device in bytes
        """

        assert type(device_name) is str
        assert type(device_id) is int
        assert type(pci_bus_id) is str
        assert type(device_uuid) is str
        assert type(card_id) is int
        assert type(chip_id) is int

        self._device_name = device_name
        self._device_id = device_id
        self._pci_bus_id = pci_bus_id
        self._device_uuid = device_uuid
        self._card_id = card_id
        self._chip_id = chip_id
        self._total_memory = total_memory

    def device_name(self):
        """
        Returns
        -------
        str
            device name
        """

        return self._device_name

    def device_id(self):
        """
        Returns
        -------
        int
            logical device id of this NPU
        """

        return self._device_id

    def pci_bus_id(self):
        """
        Returns
        -------
        str
            PCI bus id of this NPU
        """

        return self._pci_bus_id

    def device_uuid(self):
        """
        Returns
        -------
        str
            Unique identifier (card serial number) of this NPU
        """

        return self._device_uuid

    def card_id(self):
        """
        Returns
        -------
        int
            DCMI card id of the card this device belongs to
        """

        return self._card_id

    def chip_id(self):
        """
        Returns
        -------
        int
            DCMI device (chip) id within the card
        """

        return self._chip_id

    def total_memory(self):
        """
        Returns
        -------
        int
            Total HBM memory of this device in bytes
        """

        return self._total_memory
