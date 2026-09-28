"""TE-Speed-VOSR2 的 Linux 兼容节点层（Hermes 实现，非作者代码）。

为什么需要它
------------
作者包 `custom_nodes/TE-Speed-VOSR2/` 把节点层与推理层编译成了 Windows PE：
`nodes.pyd` / `settings.pyd` / `inference.pyd` / `model_store.pyd` —— Linux 上 CPython 只认
`.so`，这四个文件加载不了，所以工作流里 `TESpeedVOSR2Loader/Settings/Image/Video` 全是红块
（`missing_node_type`）。而包里的 `backend/` 是**纯 Python 的模型实现**，Linux 上完全可用。

本目录做的事：注册**同名节点**（原工作流零改动即可加载），节点名/输入名/输入顺序/返回名
全部按原工作流里存下来的 `widgets_values_named` 复刻，实际推理走
`vosr_engine.py` → 作者的 `backend/` + 上游 cswry/VOSR 的 one-step 推理流程。

作者一旦发布 Linux 版（`.so`），请**先删掉本目录**再用官方包，避免同名类冲突。
上游：https://github.com/tl2012tl/TE-Speed-VOSR2 ，模型/算法：https://github.com/cswry/VOSR
"""

from .vosr_engine import (bundle_roots, clear_engines, find_bundles, get_engine, log,
                          resolve_bundle)

PRECISIONS = ["auto", "fp16", "bf16", "fp32"]
MEMORY_POLICIES = ["auto", "balanced", "staged", "legacy"]
COLOR_MODES = ["wavelet", "adain", "nofix", "none"]


def _bundle_choices():
    try:
        names = sorted(find_bundles().keys())
    except Exception:
        names = []
    return names or ["VOSR2"]


def _settings_dict(settings):
    """settings 节点输出原样是 dict；容错处理旧文件里可能存的其它形态。"""
    if isinstance(settings, dict):
        return dict(settings)
    if isinstance(settings, (list, tuple)):
        keys = ("quality_profile", "tile_strategy", "tile_size", "tile_overlap", "vae_tile_size",
                "vae_tile_overlap", "image_batch", "frame_batch", "dino_batch", "temporal_cache",
                "cache_threshold", "cache_refresh", "memory_policy", "color_alignment")
        return dict(zip(keys, settings))
    raise ValueError("TESpeedVOSR2Settings 的输出无法识别: %r" % (settings,))


def _engine_for(model):
    if not isinstance(model, dict):
        raise ValueError("请把 TESpeedVOSR2Loader 的 model 输出接到本节点")
    return get_engine(
        bundle=model.get("bundle", "VOSR2"),
        precision=model.get("precision", "auto"),
        memory_policy=model.get("memory_policy", "auto"),
        torch_compile=bool(model.get("torch_compile", False)),
        vae_encode_amp=bool(model.get("vae_encode_amp", False)),
    )


class TESpeedVOSR2Loader:
    """加载 VOSR2 模型包（Linux 兼容实现）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_bundle": (_bundle_choices(), {"default": _bundle_choices()[0]}),
                "precision": (PRECISIONS, {"default": "auto"}),
                "memory_policy": (MEMORY_POLICIES, {"default": "auto"}),
                "torch_compile": ("BOOLEAN", {"default": False}),
                "vae_encode_amp": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TE_SPEED_VOSR2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "TE-Speed"
    DESCRIPTION = "VOSR2 模型加载器（Linux 兼容版）。模型目录：ComfyUI/models/vosr2 或 $VOSR2_MODEL_ROOT"

    def load(self, model_bundle="VOSR2", precision="auto", memory_policy="auto",
             torch_compile=False, vae_encode_amp=False):
        get_engine(model_bundle, precision, memory_policy, bool(torch_compile), bool(vae_encode_amp))
        return ({"bundle": model_bundle, "precision": precision, "memory_policy": memory_policy,
                 "torch_compile": bool(torch_compile), "vae_encode_amp": bool(vae_encode_amp)},)


class TESpeedVOSR2Settings:
    """VOSR2 推理配置（分块 / 批大小 / 颜色对齐）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quality_profile": (["speed", "manual"], {"default": "speed"}),
                "tile_strategy": (["auto", "fixed"], {"default": "auto"}),
                "tile_size": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 64}),
                "tile_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8}),
                "vae_tile_size": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 64}),
                "vae_tile_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8}),
                "image_batch": ("INT", {"default": 1, "min": 1, "max": 64}),
                "frame_batch": ("INT", {"default": 2, "min": 0, "max": 64}),
                "dino_batch": ("INT", {"default": 2, "min": 1, "max": 64}),
                "temporal_cache": ("BOOLEAN", {"default": False}),
                "cache_threshold": ("FLOAT", {"default": 0.003, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cache_refresh": ("INT", {"default": 4, "min": 0, "max": 1000}),
                "memory_policy": (MEMORY_POLICIES, {"default": "auto"}),
                "color_alignment": (COLOR_MODES, {"default": "wavelet"}),
            }
        }

    RETURN_TYPES = ("TE_SPEED_VOSR2_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = "TE-Speed"

    def build(self, quality_profile="speed", tile_strategy="auto", tile_size=512, tile_overlap=32,
              vae_tile_size=1024, vae_tile_overlap=32, image_batch=1, frame_batch=2, dino_batch=2,
              temporal_cache=False, cache_threshold=0.003, cache_refresh=4, memory_policy="auto",
              color_alignment="wavelet"):
        mode = "none" if color_alignment == "nofix" else color_alignment
        return ({
            "quality_profile": quality_profile, "tile_strategy": tile_strategy,
            "tile_size": int(tile_size), "tile_overlap": int(tile_overlap),
            "vae_tile_size": int(vae_tile_size), "vae_tile_overlap": int(vae_tile_overlap),
            "image_batch": int(image_batch), "frame_batch": int(frame_batch),
            "dino_batch": int(dino_batch), "temporal_cache": bool(temporal_cache),
            "cache_threshold": float(cache_threshold), "cache_refresh": int(cache_refresh),
            "memory_policy": memory_policy, "color_alignment": mode,
        },)


class TESpeedVOSR2Image:
    """单张/多张图片超分放大（VOSR2 one-step）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TE_SPEED_VOSR2_MODEL",),
                "images": ("IMAGE",),
                "settings": ("TE_SPEED_VOSR2_SETTINGS",),
                "scale": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 666, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "TE-Speed"

    def upscale(self, model, images, settings, scale=2, seed=666):
        eng = _engine_for(model)
        s = _settings_dict(settings)
        log("图片放大: %d 张, scale=%d, seed=%d, tile=%s, vae_tile=%s"
            % (images.shape[0], scale, seed, s.get("tile_size"), s.get("vae_tile_size")))
        out = eng.upscale_batch(images, scale, s, int(seed), is_video=False)
        log("图片放大完成: %s" % (tuple(out.shape),))
        return (out,)


class TESpeedVOSR2Video:
    """视频帧批次放大（IMAGE 帧序列进、帧序列出）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TE_SPEED_VOSR2_MODEL",),
                "images": ("IMAGE",),
                "settings": ("TE_SPEED_VOSR2_SETTINGS",),
                "scale": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 666, "min": 0, "max": 0xffffffffffffffff}),
                "frame_batch": ("INT", {"default": 0, "min": 0, "max": 64}),
                "temporal_cache": ("BOOLEAN", {"default": False}),
                "cache_threshold": ("FLOAT", {"default": 0.003, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cache_refresh": ("INT", {"default": 4, "min": 0, "max": 1000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "TE-Speed"

    def upscale(self, model, images, settings, scale=2, seed=666, frame_batch=0,
                temporal_cache=False, cache_threshold=0.003, cache_refresh=4):
        import torch
        eng = _engine_for(model)
        s = _settings_dict(settings)
        batch = int(frame_batch) or int(s.get("frame_batch") or 0) or 1
        batch = max(1, min(batch, images.shape[0]))
        log("视频帧放大: %d 帧, scale=%d, frame_batch=%d, temporal_cache=%s"
            % (images.shape[0], scale, batch, bool(temporal_cache)))
        chunks = []
        for i in range(0, images.shape[0], batch):
            part = images[i:i + batch]
            chunks.append(eng.upscale_batch(
                part, scale, s, int(seed) + i,
                temporal_cache=bool(temporal_cache), cache_threshold=float(cache_threshold),
                cache_refresh=int(cache_refresh), is_video=True,
            ))
        out = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        log("视频帧放大完成: %s" % (tuple(out.shape),))
        return (out,)


NODE_CLASS_MAPPINGS = {
    "TESpeedVOSR2Loader": TESpeedVOSR2Loader,
    "TESpeedVOSR2Settings": TESpeedVOSR2Settings,
    "TESpeedVOSR2Image": TESpeedVOSR2Image,
    "TESpeedVOSR2Video": TESpeedVOSR2Video,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TESpeedVOSR2Loader": "TE-Speed VOSR2 Loader (Linux兼容)",
    "TESpeedVOSR2Settings": "TE-Speed VOSR2 Settings (Linux兼容)",
    "TESpeedVOSR2Image": "TE-Speed VOSR2 Image (Linux兼容)",
    "TESpeedVOSR2Video": "TE-Speed VOSR2 Video Frames (Linux兼容)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "clear_engines"]


# --------------------------------------------------------------------------- #
# 可选的显存释放接口：GET / POST http://HOST:8188/te_speed_vosr2/free
# VOSR2 引擎的张量不归 ComfyUI 显存管理器调度，所以 ComfyUI 自带的
# POST /free {"unload_models": true} 释放不动它（跑完视频会一直占 3~6 GiB，
# 接着跑大模型容易 OOM）。这里挂一个接口，跑完 VOSR2 想马上跑别的模型时调一下即可，
# 不用重启 ComfyUI：
#     curl -X POST http://127.0.0.1:8188/te_speed_vosr2/free
# 注册失败只影响这个接口，不影响节点注册。
# --------------------------------------------------------------------------- #
def _register_free_route():
    try:
        from aiohttp import web
        from server import PromptServer

        async def _free(request=None):
            errs = []
            try:
                clear_engines()
            except Exception as exc:  # noqa: BLE001
                errs.append("clear_engines: %s" % exc)
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache()
            except Exception as exc:  # noqa: BLE001
                errs.append("unload_all_models: %s" % exc)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as exc:  # noqa: BLE001
                errs.append("empty_cache: %s" % exc)
            log("显存释放接口被调用；VOSR2 引擎已清空%s" % ("；部分步骤出错: " + "; ".join(errs) if errs else ""))
            return web.json_response({"status": "ok" if not errs else "partial", "errors": errs})

        PromptServer.instance.routes.get("/te_speed_vosr2/free")(_free)
        PromptServer.instance.routes.post("/te_speed_vosr2/free")(_free)
        print("[TE-Speed-VOSR2-compat] 显存释放接口: /te_speed_vosr2/free", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[TE-Speed-VOSR2-compat] 显存释放接口注册失败（不影响节点）: %s" % exc, flush=True)


_register_free_route()
