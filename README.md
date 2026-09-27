[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)

# Triton Model Analyzer（Ascend 昇腾适配版）

本仓库 fork 自 [triton-inference-server/model_analyzer](https://github.com/triton-inference-server/model_analyzer)，
并在此基础上完成了华为昇腾（Ascend）NPU 平台的适配，使 model-analyzer 能够对运行在昇腾平台
Triton Inference Server 上的模型进行调优。

## 简介

Triton Model Analyzer 是一个 CLI 工具，用于在给定硬件上为运行于
[Triton Inference Server](https://github.com/triton-inference-server/server/) 的模型
（单模型、多模型、Ensemble、BLS 等）寻找更优的模型配置（max_batch_size、
dynamic_batching、instance_group 等），并生成报告帮助理解不同配置在吞吐、延迟、
显存与算力消耗之间的权衡。

具体使用方法（CLI 参数、YAML 配置、搜索模式、报告等）请参考
[源仓库文档](https://github.com/triton-inference-server/model_analyzer/tree/main/docs)，
本 README 仅说明昇腾适配相关的内容。

## 昇腾平台适配说明

原版实现对 NVIDIA GPU 强依赖：设备枚举依赖 `numba.cuda` + DCGM（vendored ctypes 绑定），
GPU 指标采集依赖 DCGM 或 Triton metrics 端点的 `nv_gpu_*` 指标，设备可见性通过
`CUDA_VISIBLE_DEVICES` 控制。在昇腾平台上这些均不可用，且原版会在
`numba.cuda.is_available()` 为 False 时**静默降级为 cpu_only 模式**，导致无法采集
NPU 利用率/显存/功耗等指标。

本仓库的适配方案与原版 vendored DCGM 的架构完全同构，核心改动如下：

| 适配点 | 说明 |
|---|---|
| DCMI 绑定层 | 新增 `model_analyzer/monitor/dcmi/` 包，对标原版 vendored DCGM：ctypes 直接绑定昇腾驱动 `libdcmi.so`（设备枚举、芯片信息、PCIe、序列号、HBM、利用率、功耗、温度） |
| NPU 指标监控 | 新增 `DCMIMonitor`（对标 `DCGMMonitor`），周期采样并填充 `gpu_used_memory` / `gpu_free_memory` / `gpu_utilization` / `gpu_power_usage` 记录。指标口径与 npu-smi / npu-exporter 一致：AICore 利用率（%）、HBM（MB）、功耗（W） |
| NPU 设备枚举 | 新增 `NPUDevice` / `NPUDeviceFactory`（对标 `GPUDevice` / `GPUDeviceFactory`），通过 DCMI 枚举设备，解析 `ASCEND_RT_VISIBLE_DEVICES`，`--gpus` 参数接受逻辑卡号（npu-smi 编号）或卡序列号 |
| 平台抽象层 | 新增 `model_analyzer/device/platform.py`，运行时自动选择 NVIDIA（DCGM/numba）或 Ascend（DCMI）实现；可用环境变量 `MODEL_ANALYZER_PLATFORM=ascend\|nvidia` 强制指定 |
| 指标采集调度 | `MetricsManager` 在昇腾平台上自动使用 `DCMIMonitor`：local/docker/c_api 模式直接采样本机设备；remote 模式先探测 metrics 端点，无设备指标（无 `nv_gpu_*` / `npu_chip_info_*`）时回退本机 DCMI 采样 |
| 设备可见性 | Triton 启动层适配：local 模式设 `ASCEND_RT_VISIBLE_DEVICES`（逻辑卡号）；docker 模式通过 `ASCEND_VISIBLE_DEVICES` 由 ascend-docker-runtime 挂载 NPU |
| 远程监控 | `RemoteMonitor` 新增支持 npu-exporter（[mind-cluster](https://gitee.com/ascend/mind-cluster)）的 `npu_chip_info_*` 指标族，供部署了 npu-exporter 的 remote 场景使用 |
| numba 解耦 | 所有 `numba.cuda` 引用改为经平台抽象层懒加载，昇腾运行时不依赖 numba（numba 移至 `nvidia` 可选依赖） |
| 依赖清理 | `tritonclient[all]` 改为 `tritonclient[http,grpc]`（`[all]` 会拉入 cuda-python 全家桶）；删除未使用的 cryptography、distro、httplib2、urllib3 |

指标记录、结果表、图表、报告等下游模块完全复用原版实现（设备无关，按 device_uuid 组织数据）。
昇腾平台上 device_uuid 使用 DCMI 读取的**卡序列号**（如 `10256A259809-0`）。

### ge-backend Triton 已知限制

对昇腾 ge-backend Triton（`npu_ge` backend）压测时需要注意：

- 需使用 HTTP 协议（`client_protocol: http`），其 gRPC 实现与 perf_analyzer 不兼容
  （报 `failed to find the requested model version`）
- 其 HTTP 服务不支持二进制 tensor 格式，perf_analyzer 需加
  `--input-tensor-format json --output-tensor-format json`（可通过 YAML 的
  `perf_analyzer_flags` 传入）
- perf_analyzer 需显式指定模型版本（`model-version: 1`，即 `-x 1`）
- ensemble 的动态 shape 输出会被压平（`[-1,-1]` 实际返回一维），导致 perf_analyzer
  校验失败，建议对组成模型分别压测

## 编译与安装

### 编译

编译需在昇腾环境的容器中进行，使用的容器镜像为
[Ascend-SACT/tritonserver-ascend](https://gitcode.com/Ascend-SACT/tritonserver-ascend)
（当前开发与验证所用容器即基于此镜像，内含 CANN、驱动、perf_analyzer 等依赖）。

在仓库根目录执行：

```bash
pip install build
python3 -m build --wheel
```

编译产物位于 `dist/triton_model_analyzer-<version>-py3-none-any.whl`。

### 安装

```bash
# 基础安装（昇腾平台，不含 CUDA 相关依赖）
pip install dist/triton_model_analyzer-*.whl

# 可选依赖
pip install 'dist/triton_model_analyzer-*.whl[nvidia]'        # 仅 NVIDIA GPU 平台需要（numba）
pip install 'dist/triton_model_analyzer-*.whl[perf-analyzer]' # perf_analyzer 二进制，容器镜像已内置则无需
```

安装后即可使用 `model-analyzer` 命令。昇腾平台上通常无需设置
`MODEL_ANALYZER_PLATFORM`（自动检测 `libdcmi.so`）。

## Reporting problems, asking questions

欢迎反馈使用问题。反馈时请尽量提供：复现步骤、model-analyzer 日志（`--verbose`）、
YAML 配置、以及 `npu-smi info` 输出。
