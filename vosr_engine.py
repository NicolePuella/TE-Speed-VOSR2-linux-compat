"""VOSR2 推理引擎 —— TE-Speed-VOSR2 的 Linux 兼容实现（作者后端 + 参考推理流程的 Python 复刻）。

来源与边界
----------
* 模型架构、分块 VAE、颜色对齐：**作者本包自带的 `custom_nodes/TE-Speed-VOSR2/backend/`**
  （纯 Python，原样 import，未改动）。
* 采样与分块推理流程：复刻上游 `cswry/VOSR` 的 `inference_vosr_onestep.py`
  （`tiled_latent_inference` / `VOSR.sample_onestep`）。
* 作者编译成 `.pyd` 的四块（`nodes` / `settings` / `inference` / `model_store`）是 Windows PE，
  Linux 无法加载，本文件用 Python 重新实现了它们的功能（节点层见 `__init__.py`）。

与作者实现的已知差异
----------------------------------------------
* `quality_profile` / `tile_strategy` / `memory_policy` 只是调节分块与批大小，不改变数学结果。
* `torch_compile` 需要 triton；不可用时跳过（与作者行为一致，日志会说明）。
* 视频的 `temporal_cache` 是"相邻帧 DINOv2 特征复用"的近似实现，阈值/刷新按作者控件的语义。
* 引擎张量不受 ComfyUI 显存管理器调度，所以加载前会主动卸载 ComfyUI 已缓存的模型
"""

import gc
import importlib.util
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

AE_FACTOR = 8
DINOV2_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_COMFY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PACK_DIR = os.path.join(_COMFY_ROOT, "custom_nodes", "TE-Speed-VOSR2")

_BACKEND = None
_ENGINES = {}          # (bundle, precision, memory_policy) -> VOSREngine
_LAST_FEA = {}         # 视频 temporal_cache 用


def log(msg):
    print("[TE-Speed-VOSR2-compat] %s" % msg, flush=True)


def comfy_root():
    return _COMFY_ROOT


# --------------------------------------------------------------------------
# 作者 backend 包的加载（按文件路径，不动 sys.path，避免 .pyd 同名模块干扰）
# --------------------------------------------------------------------------
def backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    backend_dir = os.path.join(_PACK_DIR, "backend")
    if not os.path.isdir(backend_dir):
        raise RuntimeError("找不到作者后端目录: %s" % backend_dir)
    spec = importlib.util.spec_from_file_location(
        "te_vosr2_backend", os.path.join(backend_dir, "__init__.py"),
        submodule_search_locations=[backend_dir],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["te_vosr2_backend"] = mod
    spec.loader.exec_module(mod)
    # backend 的 __init__ 是空的，子模块要显式导入；这里收成一个命名空间供下面统一调用
    _BACKEND = SimpleNamespace(
        pkg=mod,
        models=SimpleNamespace(
            lightningdit=_import_backend_sub("models.lightningdit"),
            dinov2=_import_backend_sub("models.dinov2"),
            qwenimage_vae2d=_import_backend_sub("models.qwenimage_vae2d"),
        ),
        tiled_vae=_import_backend_sub("tiled_vae"),
        color=_import_backend_sub("color"),
    )
    return _BACKEND


def _import_backend_sub(name):
    return importlib.import_module("te_vosr2_backend." + name)


# --------------------------------------------------------------------------
# 模型目录发现
# --------------------------------------------------------------------------
def bundle_roots():
    roots = []
    env = os.environ.get("VOSR2_MODEL_ROOT", "")
    for p in env.split(os.pathsep):
        if p.strip():
            roots.append(p.strip())
    try:
        import folder_paths
        try:
            roots += list(folder_paths.get_folder_paths("vosr2"))
        except Exception:
            pass
        try:
            roots.append(os.path.join(folder_paths.models_dir, "vosr2"))
        except Exception:
            pass
    except Exception:
        pass
    # 兜底：额外模型目录，默认留空；可用 VOSR2_EXTRA_MODEL_ROOTS（os.pathsep 分隔）追加
    extra = os.environ.get("VOSR2_EXTRA_MODEL_ROOTS", "")
    for p in extra.split(os.pathsep):
        if p.strip():
            roots.append(p.strip())
    out = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        if r not in out:
            out.append(r)
    return out


def _bundle_weight(bundle_path):
    for d in ("checkpoints", "", "clean_weights"):
        for name in ("ema_model.safetensors", "model.safetensors"):
            p = os.path.join(bundle_path, d, name) if d else os.path.join(bundle_path, name)
            if os.path.isfile(p):
                return p
    return None


def find_bundles():
    """返回 {名字: 目录}，名字是相对模型根目录的路径（例如 "VOSR2"）。"""
    found = {}
    for root in bundle_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            if "args.json" in filenames and _bundle_weight(dirpath):
                name = os.path.relpath(dirpath, root).replace(os.sep, "/")
                found.setdefault(name, dirpath)
    return found


def resolve_bundle(name):
    bundles = find_bundles()
    if not bundles:
        raise RuntimeError(
            "没找到 VOSR2 模型包。搜索过: %s\n"
            "需要目录里有 args.json + checkpoints/ema_model.safetensors。"
            % ", ".join(bundle_roots())
        )
    if name in bundles:
        return name, bundles[name]
    for k in bundles:                      # 兼容只写子目录名的情况
        if k.split("/")[-1] == name:
            return k, bundles[k]
    raise RuntimeError("无效的 VOSR2 模型包名 %r，可用: %s" % (name, sorted(bundles)))


# --------------------------------------------------------------------------
# 引擎
# --------------------------------------------------------------------------
class VOSREngine:
    def __init__(self, name, path, precision="auto", memory_policy="auto",
                 torch_compile=False, vae_encode_amp=False):
        self.name = name
        self.path = path
        self.precision = precision
        self.memory_policy = memory_policy
        self.vae_encode_amp = bool(vae_encode_amp)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args_path = os.path.join(path, "args.json")
        with open(args_path, "r", encoding="utf-8") as f:
            self.args = SimpleNamespace(**json.load(f))
        if getattr(self.args, "auxiliary_time_cond", None) is None:
            self.args.auxiliary_time_cond = True
        self.layer_list = list(getattr(self.args, "layer_dinov2b_list", [17]))

        # 实测（RTX 4080S + 作者 backend）：fp16 下 DiT 会整张输出 NaN（SageAttention / SDPA 都一样），
        # bf16 与 fp32 数值一致（min/max/mean 三位小数内相同）。所以 auto = bf16：省一半显存又不炸。
        self.dtype = {
            "auto": torch.bfloat16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.bfloat16,          # fp16 会出 NaN，这里强制退回 bf16
        }.get(precision, torch.bfloat16)
        if precision == "fp16":
            log("precision=fp16 在本模型上会整张输出 NaN，已自动改用 bf16（想要最高保真用 fp32）")
        if self.dtype == torch.bfloat16 and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            log("本卡不支持 bf16，precision 退回 fp32")
            self.dtype = torch.float32

        t0 = time.time()
        self._free_comfy_models()
        self._load_vae()
        self._load_venc()
        self._load_dit(torch_compile)
        log("引擎就绪: bundle=%s precision=%s dtype=%s device=%s 用时 %.1fs"
            % (name, precision, self.dtype, self.device, time.time() - t0))

    # ---- 加载 -------------------------------------------------------------
    @staticmethod
    def _free_comfy_models():
        """VOSR2 张量不归 ComfyUI 显存管理器管，先让它把缓存的模型放掉，避免抢显存。"""
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as exc:
            log("卸载 ComfyUI 缓存模型失败（忽略）: %s" % exc)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_vae(self):
        be = backend()
        vae_dir = os.path.join(self.path, "Qwen-Image-vae-2d")
        if not os.path.isdir(vae_dir):
            vae_dir = os.path.join(self.path, "preset", "ckpts", "Qwen-Image-vae-2d")
        self.vae = be.models.qwenimage_vae2d.AutoencoderKLQwenImage2D.from_pretrained(vae_dir)
        self.vae.to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        log("VAE 就绪: %s (fp32)" % vae_dir)

    def _load_venc(self):
        be = backend()
        path = None
        for name in ("dinov2_vitl14.safetensors", "dinov2_vitl14_pretrain.pth"):
            p = os.path.join(self.path, name)
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            raise RuntimeError("VOSR2 目录里没有 dinov2_vitl14.safetensors / _pretrain.pth")
        self.venc = be.models.dinov2.build_dinov2_vitl14()
        sd = self._load_sd(path)
        missing, unexpected = self.venc.load_state_dict(sd, strict=False)
        if missing:
            log("DINOv2 缺键 %d 个（前几个: %s）" % (len(missing), list(missing)[:3]))
        self.venc.to(self.device, dtype=self.dtype).eval()
        for p in self.venc.parameters():
            p.requires_grad_(False)
        log("DINOv2 就绪: %s" % path)

    def _load_dit(self, want_compile):
        be = backend()
        a = self.args
        base_channel = 16 if getattr(a, "ae_type", "qwen") == "qwen" else 4
        self.dit = be.models.lightningdit.LightningDiT(
            input_size=a.resolution // 8,
            patch_size=a.patch_size,
            in_channels=2 * base_channel,
            out_channels=base_channel,
            hidden_size=a.dim,
            depth=a.depth,
            num_heads=a.num_heads,
            mlp_ratio=a.mlp_ratio,
            z_dims=a.enc_dim,
            encdim_ratio=a.encdim_ratio,
            auxiliary_time_cond=a.auxiliary_time_cond,
            use_qknorm=a.use_qknorm,
            use_swiglu=a.use_swiglu,
            use_rope=a.use_rope,
            use_rmsnorm=a.use_rmsnorm,
            num_fused_layers=len(self.layer_list),
        )
        wpath = _bundle_weight(self.path)
        if wpath is None:
            raise RuntimeError("VOSR2 目录里没有 ema_model.safetensors / model.safetensors")
        sd = self._load_sd(wpath)
        missing, unexpected = self.dit.load_state_dict(sd, strict=False)
        if missing or unexpected:
            log("DiT 权重不匹配! missing=%d unexpected=%d（前几个 missing: %s）"
                % (len(missing), len(unexpected), list(missing)[:3]))
        self.dit.to(self.device, dtype=self.dtype).eval()
        for p in self.dit.parameters():
            p.requires_grad_(False)
        self.dit.forward = self.dit.forward_flexible
        try:
            log("DiT 就绪: %s (%.3fB 参数) 注意力后端=%s"
                % (wpath, sum(p.numel() for p in self.dit.parameters()) / 1e9,
                   be.models.lightningdit.vosr_attention_backend()))
        except Exception:
            pass
        if want_compile:
            self._maybe_compile()

    def _maybe_compile(self):
        try:
            import triton  # noqa: F401
        except Exception:
            log("torch_compile 打开但没装 triton，跳过（与作者行为一致）")
            return
        try:
            self.dit = torch.compile(self.dit, mode="reduce-overhead")
            log("DiT 已套 torch.compile，首次调用会编译")
        except Exception as exc:
            log("torch.compile 失败，回退 eager: %s" % exc)

    @staticmethod
    def _load_sd(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        return torch.load(path, map_location="cpu")

    # ---- 前向 -------------------------------------------------------------
    def _venc_features(self, lq_bchw):
        """lq_bchw: [B,3,H,W] in [-1,1] → [feat] 其中 feat=[B,N,enc_dim]（层 17 patch token）。"""
        with torch.no_grad():
            raw = (0.5 * lq_bchw + 0.5) * 255.0
            x = F.interpolate(raw, size=(DINOV2_SIZE, DINOV2_SIZE), mode="bicubic",
                              align_corners=False).clamp(0.0, 1.0)
            mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = ((x - mean) / std).to(self.dtype)
            feats = []
            for layer_idx in self.layer_list:
                feats.append(self.venc.forward_intermediate_layer(x, int(layer_idx)))
        return feats

    def _dit_call(self, lq_latent, z, t_cur, t_nxt, fea):
        """调用 DiT；非方形 latent 用 reflect padding 补成方形（forward_flexible 只支持方形）。"""
        b, c, lh, lw = lq_latent.shape
        if lh == lw:
            inp = torch.cat([lq_latent, z], dim=1)
            return self.dit(inp, t_cur.repeat(b), t_nxt.repeat(b), fea)
        side = max(lh, lw)
        pad_h, pad_w = side - lh, side - lw
        lq_p = F.pad(lq_latent, (0, pad_w, 0, pad_h), mode="reflect")
        z_p = F.pad(z, (0, pad_w, 0, pad_h), mode="reflect")
        u = self.dit(torch.cat([lq_p, z_p], dim=1), t_cur.repeat(b), t_nxt.repeat(b), fea)
        return u[:, :, :lh, :lw]

    def _tile_grid(self, length, tile, overlap):
        return backend().tiled_vae._make_tile_grid(length, tile, overlap)

    def _flow(self, lq_latent, lq_pixels, settings, seed, n_steps=1, fea=None):
        """一步（或 n 步）flow matching：lq_latent/z 同尺寸，返回 (SR latent, 用到的 DINOv2 特征)。"""
        be = backend()
        b, lc, lh, lw = lq_latent.shape
        patch = int(getattr(self.args, "patch_size", 2))
        tile_size = int(settings.get("tile_size") or 0)
        tile_ov = int(settings.get("tile_overlap") or 0)
        dino_batch = max(1, int(settings.get("dino_batch") or 1))

        gen = torch.Generator(device=lq_latent.device)
        gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
        z = torch.randn(lq_latent.shape, generator=gen, device=lq_latent.device, dtype=lq_latent.dtype)

        lt = max((tile_size // AE_FACTOR // patch) * patch, patch) if tile_size > 0 else 0
        lt_ov = max(tile_ov // AE_FACTOR, lt // 8) if lt else 0
        if lt:
            lt = min(lt, min(lh, lw))
            lt_ov = min(lt_ov, lt - 1)
        use_tiles = bool(lt) and (lh > lt or lw > lt)

        if not use_tiles:
            if fea is None:
                fea = self._venc_features(lq_pixels)
            t_seq = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)
            for i in range(n_steps):
                t_cur, t_nxt = t_seq[i], t_seq[i + 1]
                u = self._dit_call(lq_latent, z, t_cur, t_nxt, fea)
                z = z - (t_cur - t_nxt) * u
            return z, fea

        h_pos = self._tile_grid(lh, lt, lt_ov)
        w_pos = self._tile_grid(lw, lt, lt_ov)
        g_weight = be.tiled_vae._gaussian_weights(lt, lt, lc, lq_latent.device)
        log("分块推理: latent %dx%d, tile=%d, overlap=%d, 网格=%dx%d"
            % (lh, lw, lt, lt_ov, len(h_pos), len(w_pos)))

        crops, keys = [], []
        for hi in h_pos:
            for wi in w_pos:
                ph_s, pw_s = hi * AE_FACTOR, wi * AE_FACTOR
                ph_e = min((hi + lt) * AE_FACTOR, lq_pixels.shape[2])
                pw_e = min((wi + lt) * AE_FACTOR, lq_pixels.shape[3])
                crops.append(lq_pixels[:, :, ph_s:ph_e, pw_s:pw_e])
                keys.append((hi, wi))
        tile_fea = {}
        with torch.no_grad():
            for i in range(0, len(crops), dino_batch):
                chunk = crops[i:i + dino_batch]
                if len({tuple(c.shape) for c in chunk}) == 1:
                    fea = self._venc_features(torch.cat(chunk, dim=0))
                    fea = [f.chunk(len(chunk), dim=0) for f in fea]
                    for j, k in enumerate(keys[i:i + dino_batch]):
                        tile_fea[k] = [f[j].contiguous() for f in fea]
                else:
                    for j, k in enumerate(keys[i:i + dino_batch]):
                        tile_fea[k] = self._venc_features(chunk[j])

        t_seq = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)
        for step_i in range(n_steps):
            t_cur, t_nxt = t_seq[step_i], t_seq[step_i + 1]
            dt = t_cur - t_nxt
            u_acc = torch.zeros_like(lq_latent)
            w_acc = torch.zeros_like(lq_latent)
            for hi, wi in keys:
                he, we = hi + lt, wi + lt
                u_tile = self._dit_call(
                    lq_latent[:, :, hi:he, wi:we], z[:, :, hi:he, wi:we],
                    t_cur, t_nxt, tile_fea[(hi, wi)],
                )
                u_acc[:, :, hi:he, wi:we] += u_tile * g_weight
                w_acc[:, :, hi:he, wi:we] += g_weight
            z = z - dt * (u_acc / w_acc)
        return z, None

    # ---- 对外：放大一批图 ---------------------------------------------------
    def upscale_batch(self, images, scale, settings, seed, temporal_cache=False,
                      cache_threshold=0.003, cache_refresh=4, is_video=False):
        """images: [N,H,W,3] float 0..1（ComfyUI IMAGE）→ [N,H*scale,W*scale,3]。"""
        be = backend()
        n = images.shape[0]
        scale = max(1, int(scale))
        vae_tile = int(settings.get("vae_tile_size") or 0)
        vae_ov = int(settings.get("vae_tile_overlap") or 0)
        color_mode = str(settings.get("color_alignment") or "wavelet")
        out = []
        prev_pix = None
        prev_fea = None
        amp = self.vae_encode_amp
        t_start = time.time()
        for i in range(n):
            img = images[i].permute(2, 0, 1).unsqueeze(0)      # [1,3,H,W] 0..1
            lq_pixels = (img * 2.0 - 1.0)
            if scale != 1:
                lq_pixels = F.interpolate(lq_pixels, scale_factor=scale, mode="bicubic",
                                          align_corners=False)
            lq_pixels = lq_pixels.clamp(-1.0, 1.0).to(self.device)
            with torch.no_grad():
                if amp and self.device.type == "cuda":
                    with torch.autocast("cuda", dtype=self.dtype):
                        lq_latent, mean, std = be.tiled_vae.encode_dispatch(
                            self.vae, lq_pixels, vae_tile, vae_ov)
                else:
                    lq_latent, mean, std = be.tiled_vae.encode_dispatch(
                        self.vae, lq_pixels, vae_tile, vae_ov)
                lq_latent = lq_latent.to(self.dtype)

                s = dict(settings)
                fea = None
                if temporal_cache and is_video and prev_pix is not None and prev_fea is not None:
                    diff = (lq_pixels - prev_pix).abs().mean().item()
                    stale = (i % max(1, int(cache_refresh))) == 0
                    if diff < float(cache_threshold) and not stale:
                        fea = prev_fea
                        log("第 %d 帧与上一帧差异 %.5f < %.5f，复用上一帧 DINOv2 特征" % (i + 1, diff, cache_threshold))
                z, fea = self._flow(lq_latent, lq_pixels, s, int(seed) + i, fea=fea)
                prev_fea = fea
                prev_pix = lq_pixels
                sr = be.tiled_vae.decode_dispatch(self.vae, z, mean, std, vae_tile, vae_ov)
            # 分块 VAE 会把边长补到 8 的倍数，解码结果可能比目标大几个像素（例如 3435→3440），
            # 直接裁掉反射填充的部分；万一反而更小再插值回去。
            th, tw = lq_pixels.shape[-2:]
            if tuple(sr.shape[-2:]) != (th, tw):
                if sr.shape[-2] >= th and sr.shape[-1] >= tw:
                    sr = sr[..., :th, :tw]
                else:
                    sr = F.interpolate(sr, size=(th, tw), mode="bicubic", align_corners=False)
            sr = (sr * 0.5 + 0.5).clamp(0.0, 1.0)
            ref = (lq_pixels * 0.5 + 0.5).clamp(0.0, 1.0)
            sr = be.color.apply_color_alignment(sr, ref, color_mode)
            out.append(sr[0].permute(1, 2, 0).to(torch.float32).cpu())
            del lq_latent, z, sr, ref, lq_pixels
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            log("第 %d/%d 帧完成（%.1fs）" % (i + 1, n, time.time() - t_start))
        return torch.stack(out, dim=0)


def get_engine(bundle="VOSR2", precision="auto", memory_policy="auto",
               torch_compile=False, vae_encode_amp=False):
    name, path = resolve_bundle(bundle)
    key = (name, precision, memory_policy)
    eng = _ENGINES.get(key)
    if eng is None:
        eng = VOSREngine(name, path, precision, memory_policy, torch_compile, vae_encode_amp)
        _ENGINES[key] = eng
    else:
        log("复用已加载的引擎: %s/%s/%s" % key)
    return eng


def clear_engines():
    _ENGINES.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
