import atexit
import base64
import io
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image

import folder_paths


# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-VL captioning — a local vision-language model exposed through the
# regular ComfyUI HTTP API.
#
# MagoLocal (the desktop build) has no cloud LLM, so image/video captioning,
# AutoPrompt and prompt-enhance all run through these two nodes instead: the
# app POSTs a small graph to /prompt and reads the generated text back out of
# /history.  That is why MagoQwen3VLCaption is an OUTPUT_NODE returning
# {"ui": {"text": [...]}} — it is the only way a STRING reaches an API client.
#
# Weights go in ComfyUI/models/LLM/ (any subfolder), e.g.
#   models/LLM/GGUF/Qwen3-VL-2B-Instruct-UD-Q8_K_XL.gguf
#   models/LLM/GGUF/mmproj-BF16.gguf
# from https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF
#
# Inference runs through one of two interchangeable runners (see
# _build_runner): llama-cpp-python when the installed wheel can do vision, and
# a managed `llama-server` subprocess otherwise.  Vision support in
# llama-cpp-python lags llama.cpp and needs a special wheel on Windows, so the
# subprocess path is what makes this installable without one.
# ─────────────────────────────────────────────────────────────────────────────

LLM_FOLDER = "LLM"
MAX_IMAGES = 4
PLACEHOLDER = "<no .gguf found in models/LLM>"
LOG = "[Mago VLM]"

_SERVER_STARTUP_TIMEOUT_S = 240
_SERVER_POLL_INTERVAL_S = 0.5
_REQUEST_TIMEOUT_S = 600


def _register_llm_folder():
    """Expose ComfyUI/models/LLM (recursively) as a folder_paths entry so both
    the model and its mmproj show up as ordinary combo values."""
    root = os.path.join(folder_paths.models_dir, LLM_FOLDER)
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        print(f"{LOG} could not create {root}: {exc}")
    entry = folder_paths.folder_names_and_paths.get(LLM_FOLDER)
    if entry is None:
        folder_paths.folder_names_and_paths[LLM_FOLDER] = ([root], {".gguf"})
        return
    paths, extensions = entry
    if root not in paths:
        paths.append(root)
    extensions.add(".gguf")


_register_llm_folder()


def _gguf_choices():
    try:
        files = folder_paths.get_filename_list(LLM_FOLDER)
    except Exception as exc:  # a broken models dir must not break node loading
        print(f"{LOG} could not list {LLM_FOLDER}: {exc}")
        files = []
    return list(files) or [PLACEHOLDER]


def _resolve_gguf(name):
    if not name or name == PLACEHOLDER:
        raise FileNotFoundError(
            "No GGUF selected. Put Qwen3-VL-2B-Instruct-UD-Q8_K_XL.gguf and "
            "mmproj-BF16.gguf under ComfyUI/models/LLM/GGUF/ and refresh."
        )
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"GGUF not found in models/{LLM_FOLDER}: {name}")
    return path


def _tensor_to_png(image):
    """ComfyUI IMAGE (B,H,W,C float 0-1) → PNG bytes for the first frame only.

    Callers pass one image per input slot; a batch's extra frames are ignored
    rather than silently concatenated, because the caption prompts describe
    each slot positionally.
    """
    array = image
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected an HxWxC image, got shape {array.shape}")
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _build_messages(system_prompt, user_prompt, images):
    """Images first, then the instruction — Qwen-VL grounds noticeably better
    that way, and it matches the ordering the cloud Gemini calls used."""
    content = [{"type": "image_url", "image_url": {"url": _data_uri(png)}} for png in images]
    content.append({"type": "text", "text": user_prompt})
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

# Newest first. Qwen3-VL and Qwen2.5-VL share the ChatML + <|vision_start|>
# prompt shape, so the 2.5 handler drives a Qwen3-VL mmproj correctly as long
# as the underlying llama.cpp build knows the projector. Anything older (e.g.
# Llava15ChatHandler) does not, and would emit plausible-looking nonsense — so
# it is deliberately absent from this list.
_VISION_HANDLERS = ("Qwen3VLChatHandler", "Qwen25VLChatHandler")


class LlamaCppPythonRunner:
    """In-process inference via llama-cpp-python. Preferred when available: no
    second process, no port, and the model unloads with ComfyUI."""

    def __init__(self, config):
        from llama_cpp import Llama
        from llama_cpp import llama_chat_format

        handler_cls = None
        for name in _VISION_HANDLERS:
            handler_cls = getattr(llama_chat_format, name, None)
            if handler_cls is not None:
                print(f"{LOG} llama-cpp-python vision handler: {name}")
                break
        if handler_cls is None:
            raise RuntimeError(
                "Installed llama-cpp-python has no Qwen-VL chat handler "
                f"(looked for {', '.join(_VISION_HANDLERS)}). Install a "
                "vision-capable wheel or set MAGO_LLAMA_SERVER."
            )

        self._handler = handler_cls(clip_model_path=config["mmproj_path"], verbose=False)
        self._llama = Llama(
            model_path=config["gguf_path"],
            chat_handler=self._handler,
            n_ctx=config["n_ctx"],
            n_gpu_layers=config["n_gpu_layers"],
            logits_all=False,
            verbose=False,
        )

    def generate(self, messages, max_tokens, temperature, top_p, seed):
        # `seed` moved from Llama.set_seed to a create_chat_completion kwarg
        # across llama-cpp-python versions; support both.
        try:
            self._llama.set_seed(seed)
            seed_kwargs = {}
        except AttributeError:
            seed_kwargs = {"seed": seed}
        response = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **seed_kwargs,
        )
        return (response["choices"][0]["message"]["content"] or "").strip()

    def close(self):
        try:
            self._llama.close()
        except Exception:
            pass


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LlamaServerRunner:
    """Inference via a managed `llama-server` subprocess speaking the
    OpenAI-compatible /v1/chat/completions API on localhost.

    This is the fallback that keeps the node installable when llama-cpp-python
    cannot do vision. The process is owned by this runner: it is started on
    first use, reused for every subsequent caption, and killed when the runner
    is evicted from the cache or the interpreter exits.
    """

    def __init__(self, config):
        binary = os.environ.get("MAGO_LLAMA_SERVER") or shutil.which("llama-server")
        if not binary or not os.path.exists(binary):
            raise RuntimeError(
                "llama-server not found. Install a vision-capable "
                "llama-cpp-python, or point MAGO_LLAMA_SERVER at a llama.cpp "
                "llama-server binary."
            )
        self._port = _free_port()
        args = [
            binary,
            "-m", config["gguf_path"],
            "--mmproj", config["mmproj_path"],
            "-c", str(config["n_ctx"]),
            "-ngl", str(config["n_gpu_layers"]),
            "--host", "127.0.0.1",
            "--port", str(self._port),
        ]
        print(f"{LOG} starting llama-server on port {self._port}")
        self._process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        atexit.register(self.close)
        self._await_health()

    def _await_health(self):
        deadline = time.monotonic() + _SERVER_STARTUP_TIMEOUT_S
        url = f"http://127.0.0.1:{self._port}/health"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup (code {self._process.returncode})"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(_SERVER_POLL_INTERVAL_S)
        self.close()
        raise RuntimeError(
            f"llama-server did not become healthy within {_SERVER_STARTUP_TIMEOUT_S}s"
        )

    def generate(self, messages, max_tokens, temperature, top_p, seed):
        payload = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:
            body = json.loads(response.read().decode("utf-8"))
        return (body["choices"][0]["message"].get("content") or "").strip()

    def close(self):
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def _build_runner(config):
    try:
        return LlamaCppPythonRunner(config)
    except Exception as exc:
        print(f"{LOG} llama-cpp-python unavailable ({exc}); falling back to llama-server")
    return LlamaServerRunner(config)


# One resident model. Every caption is a separate /prompt execution, so without
# this cache each one would pay a full ~2.3 GB cold load.
_CACHE_LOCK = threading.Lock()
_CACHED = {"key": None, "runner": None}


def _runner_for(config):
    key = (
        config["gguf_path"],
        config["mmproj_path"],
        config["n_ctx"],
        config["n_gpu_layers"],
    )
    with _CACHE_LOCK:
        if _CACHED["key"] == key and _CACHED["runner"] is not None:
            return _CACHED["runner"]
        previous = _CACHED["runner"]
        _CACHED["key"] = None
        _CACHED["runner"] = None
        if previous is not None:
            previous.close()
        runner = _build_runner(config)
        _CACHED["key"] = key
        _CACHED["runner"] = runner
        return runner


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────


class MagoQwen3VLLoader:
    """
    Resolve a Qwen3-VL GGUF + its mmproj and hand back a resident handle.

    The handle is a plain config dict; the model itself lives in a module-level
    cache keyed on these values, so re-running the graph reuses it.
    """

    @classmethod
    def INPUT_TYPES(s):
        choices = _gguf_choices()
        return {
            "required": {
                "gguf_name": (choices, {
                    "tooltip": "Language model, e.g. GGUF/Qwen3-VL-2B-Instruct-UD-Q8_K_XL.gguf",
                }),
                "mmproj_name": (choices, {
                    "tooltip": "Matching vision projector, e.g. GGUF/mmproj-BF16.gguf",
                }),
                "n_ctx": ("INT", {
                    "default": 16384, "min": 2048, "max": 262144, "step": 512,
                    "tooltip": "Context window. The caption prompts are up to "
                               "~2.5k tokens before images, so leave headroom.",
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1, "min": -1, "max": 200,
                    "tooltip": "-1 offloads everything to the GPU.",
                }),
            },
        }

    RETURN_TYPES = ("MAGO_VLM",)
    RETURN_NAMES = ("vlm",)
    FUNCTION = "load"
    CATEGORY = "Mago Nodes/VLM"

    def load(self, gguf_name, mmproj_name, n_ctx, n_gpu_layers):
        config = {
            "gguf_path": _resolve_gguf(gguf_name),
            "mmproj_path": _resolve_gguf(mmproj_name),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
        }
        _runner_for(config)  # fail loudly here rather than mid-caption
        return (config,)


class MagoQwen3VLCaption:
    """
    Generate text from up to four images plus an instruction.

    OUTPUT_NODE so the result lands in /history at
    outputs["<node_id>"]["text"][0] — that is the contract MagoLocal's
    comfyVlmClient depends on.

    Separate image slots (rather than one IMAGE batch) because callers mix
    resolutions: the video captioner sends a full-res first frame alongside
    smaller frame-grid montages, and batching would force them to a common
    size.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vlm": ("MAGO_VLM",),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe this image.", "multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "API callers should vary this per request; an "
                               "identical graph would otherwise be served from "
                               "ComfyUI's execution cache.",
                }),
            },
            "optional": {
                f"image{index}": ("IMAGE",) for index in range(1, MAX_IMAGES + 1)
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "caption"
    OUTPUT_NODE = True
    CATEGORY = "Mago Nodes/VLM"

    def caption(self, vlm, system_prompt, user_prompt, max_tokens, temperature,
                top_p, seed, **images):
        pngs = []
        for index in range(1, MAX_IMAGES + 1):
            image = images.get(f"image{index}")
            if image is not None:
                pngs.append(_tensor_to_png(image))
        messages = _build_messages(system_prompt, user_prompt, pngs)
        runner = _runner_for(vlm)
        started = time.monotonic()
        text = runner.generate(messages, max_tokens, temperature, top_p, seed)
        print(
            f"{LOG} captioned {len(pngs)} image(s) in "
            f"{time.monotonic() - started:.1f}s ({len(text)} chars)"
        )
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "MagoQwen3VLLoader": MagoQwen3VLLoader,
    "MagoQwen3VLCaption": MagoQwen3VLCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagoQwen3VLLoader": "Load Qwen3-VL (Mago)",
    "MagoQwen3VLCaption": "Qwen3-VL Caption (Mago)",
}
