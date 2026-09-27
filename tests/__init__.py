#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# Unit tests default to the NVIDIA platform so that they behave the same on
# every machine (the Ascend DCMI library is auto-detected otherwise and would
# change the code paths under test on Ascend hosts). Tests that exercise the
# Ascend code paths mock the DCMI layer explicitly. Set
# MODEL_ANALYZER_PLATFORM explicitly to override.
os.environ.setdefault("MODEL_ANALYZER_PLATFORM", "nvidia")
