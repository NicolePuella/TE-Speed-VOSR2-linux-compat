# TE-Speed-VOSR2 Linux 兼容版（Community Linux port）

让 **Linux 上的 ComfyUI** 也能跑 [`TE-Speed-VOSR2`](https://github.com/tl2012tl/TE-Speed-VOSR2)
的 VOSR2 图片/视频超分节点。注册**同名节点**，原工作流**零改动**即可加载。

> 非官方兼容层。作者仓库：<https://github.com/tl2012tl/TE-Speed-VOSR2>；算法/模型：<https://github.com/cswry/VOSR>

## 为什么需要它

作者包 `custom_nodes/TE-Speed-VOSR2/` 把节点层和推理层编译成了 **Windows PE**：

```
nodes.pyd  settings.pyd  inference.pyd  model_store.pyd
```

Linux 上 CPython 只认 `.so`，这四个文件直接加载失败 → 工作流里
`TESpeedVOSR2Loader / Settings / Image / Video` 全是红块（`missing_node_type`）。

但作者包里的 `backend/` 是**纯 Python 的模型实现**（LightningDiT 1.4B、DINOv2-L、
Qwen-Image VAE、分块 VAE、颜色对齐），Linux 上完全可用。本仓库只补上缺失的那一层：
用 Python 重新实现 `.pyd` 的功能，推理流程按上游
[`cswry/VOSR`](https://github.com/cswry/VOSR) 的 `inference_vosr_onestep.py` 复刻
（one-step flow matching + latent 分块 + 每块独立 DINOv2 特征 + 高斯权重融合）。

## 安装

```bash
cd ComfyUI/custom_nodes/

# 1) 先装作者原包（本兼容层需要它的 backend/，这是必需的）
git clone https://github.com/tl2012tl/TE-Speed-VOSR2

# 2) 再装本兼容层
git clone https://github.com/NicolePuella/TE-Speed-VOSR2-linux-compat

# 3) 重启 ComfyUI
```

**模型放哪**（按顺序找，第一个命中的生效）：

1. 环境变量 `VOSR2_MODEL_ROOT`（可用 `:` 分隔多个目录）
2. ComfyUI 的 `models/vosr2/`
3. 环境变量 `VOSR2_EXTRA_MODEL_ROOTS`（可选，追加额外目录）

模型包 = 一个含 `args.json` + `checkpoints/ema_model.safetensors` 的目录，例如
`models/vosr2/VOSR2/`。

无额外 Python 依赖，只用 ComfyUI 自带的 `torch` / `safetensors`。

## 节点与输入（与原包一致）

| 节点 | 输入（顺序即原顺序） | 输出 |
|---|---|---|
| `TESpeedVOSR2Loader` | model_bundle, precision, memory_policy, torch_compile, vae_encode_amp | model |
| `TESpeedVOSR2Settings` | quality_profile, tile_strategy, tile_size, tile_overlap, vae_tile_size, vae_tile_overlap, image_batch, frame_batch, dino_batch, temporal_cache, cache_threshold, cache_refresh, memory_policy, color_alignment | settings |
| `TESpeedVOSR2Image` | model, images, settings, scale, seed | image |
| `TESpeedVOSR2Video` | model, images, settings, scale, seed, frame_batch, temporal_cache, cache_threshold, cache_refresh | image |

节点名 / 输入名 / 输入顺序 / 返回名全部按原工作流里存下来的 `widgets_values_named` 复刻，
所以**原来的工作流文件不用改**，直接把红块换成新拉起的同名节点即可。

## precision 实测（RTX 4080S）

| 值 | 结果 |
|---|---|
| `auto` / `bf16` | **默认**。无 NaN，SageAttention 可用，显存最省 |
| `fp32` | 与上游参考实现完全一致；SageAttention 不支持 fp32 会自动退回 SDPA（日志有提示） |
| `fp16` | **整张输出 NaN**，已强制改走 bf16（别用） |

bf16 与 fp32 输出差异：平均 1/255，个别高频像素最大 83/255（视觉一致）。

## 实测性能（512×512 → 1024×1024, scale=2, tile 512 / vae_tile 1024）

- 引擎加载 ~90s（首次；之后同精度复用缓存），单张推理 ~2s，峰值显存 ~8.8 GiB
- 锐度参考：拉普拉斯方差 **VOSR2 198** vs 双三次 24（确实是超分不是放大）
- 分块推理无可见接缝（latent 128×128，tile 64，overlap 8，3×3 网格）

## 跑完想马上跑别的模型：调一下释放接口

引擎张量不归 ComfyUI 显存管理器调度：加载前会主动 `unload_all_models()` 释放 ComfyUI 缓存的模型，
所以**不要同时跑两个工作流**；VOSR2 跑完会一直占 3~6 GiB，接着跑大模型容易 OOM。

不用重启 ComfyUI，调一下释放接口即可（同时清掉 VOSR2 引擎和 ComfyUI 自己的模型缓存）：

```bash
curl -X POST http://HOST:8188/te_speed_vosr2/free    # GET 同样可用，返回 {"status":"ok"}
```

注意 ComfyUI 自带的 `POST /free {"unload_models": true}` **释放不动引擎占的显存**，别指望它。
再次用 VOSR2 时会重新加载引擎（约 95 秒）。

## 与作者实现的已知差异

- 作者的「speed/manual 档位、显存调度、DiT 瓦片批处理」是**性能优化**；这里只保留**数学等价**的分块与批处理，
  没有复刻他的全部加速策略，所以速度可能比 Windows 版慢一些。
- `torch_compile` 需要 triton，缺失时跳过（与作者一致）。
- `temporal_cache` 只在**未分块**路径生效（相邻帧差异小于阈值时复用上一帧 DINOv2 特征），分块路径忽略。
- `quality_profile` / `tile_strategy` / `memory_policy` 只调节分块与批大小，不改变数学结果。

## 环境

在 Linux + ComfyUI（Python 3.12 / torch 2.x，RTX 4080S）实测通过。
如果在你的环境跑出问题，欢迎开 issue 并附上启动日志里 `[TE-Speed-VOSR2-compat]` 开头的行。

## 什么时候删掉这个目录

作者一旦发布 Linux 版（`.so`），**先删本目录**再用官方包，避免同名类冲突：

```bash
rm -rf custom_nodes/TE-Speed-VOSR2-linux-compat
```

## 致谢

- [tl2012tl/TE-Speed-VOSR2](https://github.com/tl2012tl/TE-Speed-VOSR2) —— 节点定义、`backend/` 模型实现（本仓库原样 import，未改动）
- [cswry/VOSR](https://github.com/cswry/VOSR) —— VOSR2 模型与 one-step 推理流程
