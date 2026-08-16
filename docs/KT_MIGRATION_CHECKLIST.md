# KT 技术迁移清单

本文记录 KTransformers（KT）能力迁移到当前 SGLang 版本的工作状态。

当前目标硬件：Ampere One 192 核 CPU + 2 张 NVIDIA RTX PRO 6000。

当前重点模型：`Qwen3.6-35B-A3B-Q8_0.gguf`。

## 当前基线

- [x] ARM Q8_0 标准 GGML block 布局问题已修复，绕过错误的 ARM IQK x4 快路径。
- [x] `step1_test_arm_cuda_q8.py` 已完成直接 CPUInfer 与 CUDA stream 预检。
- [x] Q8_0 直接 CPU/CUDA 输出均为 finite，1000 次稳定性测试通过。
- [x] BF16 MoE 测试已通过，包含 1000 次运行。
- [x] FP16 MoE 测试已通过，包含 1000 次运行。
- [x] 当前 `sglang` 集成分支已推送到 `springrain/sglang` 的 `main`。
- [ ] `ktransformers` 主仓库的 `third_party/sglang` gitlink 已更新到新的 SGLang commit。
- [ ] `ktransformers` 主仓库的 `.gitmodules` 已切换到实际使用的 SGLang fork。

## P0：Q8 主链路

- [ ] 在 ARM 服务器重新编译并安装与当前 KT 源码匹配的 `kt_kernel` 扩展。
- [ ] 使用当前 SGLang main 完成 Q8_0 GGUF 端到端启动和请求测试。
- [ ] 验证 `kt-num-gpu-experts=0`，覆盖全 CPU 专家。
- [ ] 验证 `kt-num-gpu-experts=64`，覆盖 CPU/GPU 混合专家。
- [ ] 验证 `kt-num-gpu-experts=256`，覆盖全 GPU 专家对照组。
- [ ] 验证 TP1 和 TP2。
- [ ] 验证 batch=1、batch=16 和较大 batch。
- [ ] 分别验证 prefill 和 decode。
- [ ] 验证有无 `CUDA_LAUNCH_BLOCKING=1` 时输出一致。
- [ ] 验证有无 `CUDA_LAUNCH_BLOCKING=1` 时吞吐和首 token 延迟。
- [ ] 验证默认 CUDA graph 配置。
- [ ] 使用 `--disable-cuda-graph` 完成 eager 基线对照。
- [ ] 确认 Q8 权重加载、专家 mask、GPU expert remap 均无错误。
- [ ] 确认 CPU 专家实际执行，避免 GPU 满载但 CPU 长时间空闲。
- [ ] 确认 SGLang 端到端输出无 NaN/Inf，且与全 GPU 对照误差可接受。

## P1：Q8 稳定性和性能

- [ ] 验证 NUMA 节点配置。
- [ ] 验证多 threadpool 配置。
- [ ] 验证 `kt-cpuinfer` 与 threadpool 的线程分配。
- [x] 基础 CUDA stream 提交、同步和 CPU/GPU 合并路径已在 Step 1 验证。
- [ ] 验证 deferred experts 和 `MAX_DEFER`。
- [ ] 验证动态 GPU expert 更新。
- [ ] 验证专家分布记录和 mask telemetry。
- [ ] 验证 EPLB 与 KT 的组合。
- [ ] 验证非 trivial expert mapping、物理专家映射和权重更新。
- [x] 增加 SGLang 端到端 Q8 回归脚本。
- [x] 增加 KT mask/remap、专家选择、动态映射和参数组合单元测试。
- [ ] 增加 Q8 CPU/GPU 混合路径的持续集成或定期实机测试。

### 测试入口

黑盒矩阵脚本位于 test/manual/kt_q8_hybrid_e2e.py，需要在 ARM + CUDA
服务器上执行。默认覆盖 GPU 专家数 0/64/256、TP 1/2、batch 1/16
以及 CUDA_LAUNCH_BLOCKING 开关：

~~~bash
python test/manual/kt_q8_hybrid_e2e.py \
  --model /data/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --trust-remote-code \
  --kt-weight-path /data/ampere/data/Qwen3.6-35B-A3B-Q8_0.gguf \
  --kt-method LLAMAFILE \
  --kt-cpuinfer 128 \
  --kt-threadpool-count 2
~~~

P1 配置可通过 --kt-numa-nodes、--max-deferred-experts-per-token、
--gpu-prefill-token-threshold、--enable-dynamic-expert-update 和
--record-expert-distribution 加入同一矩阵。日志和结果默认写入
test-logs/kt-q8-hybrid/。单元测试可用：

~~~bash
python -m pytest -q test/registered/unit/layers/moe/test_kt_ep_helpers.py
~~~

## P2：Composite KT LoRA

- [ ] 基于当前 SGLang LoRA API 重写 merged adapter 拆分逻辑。
- [ ] 通用识别 expert 参数与 attention/linear 参数，不依赖 Qwen3.5 特判。
- [ ] 将 expert LoRA 交给 KT CPU expert wrapper。
- [ ] 将非 expert LoRA 交给 SGLang 原生 LoRA manager。
- [ ] 支持标准 PEFT adapter 和 safetensors adapter。
- [ ] 增加 TP、batch、CUDA graph 和并发请求约束。
- [ ] 验证 DeepSeek、Qwen、GLM 等标准 MoE 模型。
- [ ] 验证非 MoE 模型不被 KT LoRA 逻辑影响。

## P2：MXFP4 layerwise prefill

- [ ] 基于当前 upstream MXFP4 quant method 重写 layerwise manager。
- [ ] 不再依赖已删除的 `v4_marlin_moe.py` 和旧 DeepSeek 专用类。
- [ ] 实现 GPU 权重 slot、显存预留和懒加载。
- [ ] 实现 MXFP4 Marlin/FlashInfer repack 和恢复。
- [ ] 实现 prefill token threshold 触发逻辑。
- [ ] 实现 OOM 检测、禁用和普通 hybrid fallback。
- [ ] 验证 TP 多 rank 初始化和状态一致性。
- [ ] 验证 layerwise 模式不会破坏普通 decode。
- [ ] 增加 MXFP4 layerwise 单元测试和端到端测试。
- [ ] 单独评估 ARM CPU MXFP4 内核需求，避免将 GPU layerwise 与 ARM CPU 支持混为一谈。

## P3：SM120 MXFP8 MoE（暂缓）

- [ ] 跟踪 upstream SGLang 对 SM120（RTX PRO 6000）原生 W8A8 MXFP8 MoE 的支持状态；PR #33208 仅覆盖 MXFP8 dense GEMM，不包含 MoE。
- [ ] upstream 完成 SM120 MXFP8 MoE 权重处理和 runner 集成后，再评估接入 custom FlashInfer 与 KT GPU experts 路径。
- [ ] 接入时验证 TP、GPU expert remap、prefill/decode、CUDA graph 和数值一致性。

当前暂不修改运行代码，也不将 SM120 MXFP8 dense 支持视为 MXFP8 MoE 已支持。

## P3：通用模型和依赖维护

- [ ] 以通用 `FusedMoE`/`quant_method` 接口适配新模型。
- [ ] 验证 DeepSeek、Qwen、GLM、MiniMax 等标准 MoE 模型。
- [ ] 为自定义 dispatcher、top-k、量化格式建立适配边界。
- [ ] 建立 SGLang、custom_flashinfer、sglang-kernel、kt-kernel 版本矩阵。
- [ ] 明确 `sglang-kt` 与 `kt_kernel` 的安装和 ABI 匹配要求。
- [ ] 清理旧版 `v4_marlin_moe` 等过时代码和兼容分支。
- [ ] 恢复或重写 KT 专用 benchmark、性能扫描和诊断脚本。

## Q8 完成标准

- [ ] ARM 实机上全 CPU、混合、全 GPU 三组结果均 finite。
- [ ] 混合模式与全 GPU 对照输出误差在预设阈值内。
- [ ] 无 `CUDA_LAUNCH_BLOCKING=1` 时没有明显功能或吞吐回退。
- [ ] CPU 和 GPU 均有符合预期的利用率。
- [ ] TP1/TP2、prefill/decode、不同 batch 均通过。
- [ ] 测试命令、日志、模型配置和扩展 `.so` 路径全部记录。
