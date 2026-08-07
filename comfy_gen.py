"""ComfyUI-backed MiniMax-H3 Ref2VA generation library.

Runs ComfyUI as an in-process local server (bound to 127.0.0.1) inside a Modal
container and drives it through the standard /prompt API with a graph built
programmatically. Two sampling modes:

- ``turbo=True``  MiniMax-H3 Turbo LoRA + 4-step dual-schedule sampler
                  (cheap iteration; preview quality).
- ``turbo=False`` stock res_multistep sampler, 20 steps by default (final hero renders).

Prompt reference tags are ported from the diffusers vocabulary ``<Subject N>``
to Comfy's ``<Picture N>``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

import requests

# ---- model / lora file names (must match files on the comfyui-models volume) --
REPO = "Comfy-Org/MiniMax-H3"
TURBO_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"

TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_turbo_4step_ema_ckpt850.safetensors"

FPS = 24


def frame_length_for(duration: float) -> int:
    """Snap a duration (seconds) to the H3 17k+5 frame grid at 24 fps."""
    n = max(5, round(duration * FPS))
    while n % 17 != 5:
        n += 1
    return n


def port_prompt(scene_text: str) -> str:
    """Strip the authoring preamble and map <Subject N> -> <Picture N>."""
    prompt = scene_text[scene_text.index("subject_definitions:"):]
    return re.sub(r"<Subject (\d+)>", r"<Picture \1>", prompt)


class _NodeGraph:
    """Tiny id-keyed dict builder for ComfyUI /prompt API format."""

    def __init__(self):
        self.p: dict = {}
        self._n = 0

    def add(self, class_type: str, inputs: dict) -> tuple[str, int]:
        self._n += 1
        nid = str(self._n)
        self.p[nid] = {"class_type": class_type, "inputs": inputs}
        return nid, 0


def build_prompt(*, ref_names: list[str], ported_prompt: str, width: int,
                 height: int, length: int, ref_image_size: str, mode: str,
                 steps: int, lora_strength: float, seed: int) -> dict:
    """Return a ComfyUI /prompt payload (the ``prompt`` key)."""
    assert 1 <= len(ref_names) <= 9, "H3 Ref2VA supports 1-9 reference images"
    g = _NodeGraph()

    load_refs: list[str] = []
    for name in ref_names:
        load_refs.append(g.add("LoadImage", {"image": name})[0])

    UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    unet, _ = g.add("UNETLoader", {"unet_name": UNET, "weight_dtype": "default"})

    model = unet
    if mode == "turbo":
        model, _ = g.add("MiniMaxH3TurboLoRA", {
            "model": [unet, 0], "lora_name": TURBO_LORA, "strength": lora_strength,
        })
        sampler, _ = g.add("MiniMaxH3TurboSampler", {})
    else:
        sampler, _ = g.add("KSamplerSelect", {"sampler_name": "res_multistep"})

    clip, _ = g.add("CLIPLoader", {
        "clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"})
    vae_v, _ = g.add("VAELoader", {"vae_name": VIDEO_VAE})
    vae_a, _ = g.add("VAELoader", {"vae_name": AUDIO_VAE})

    sched_model = model if mode == "turbo" else unet
    scheduler, _ = g.add("BasicScheduler", {
        "model": [sched_model, 0], "scheduler": "simple", "steps": steps,
        "denoise": 1.0})

    r2v_inputs: dict = {
        "clip": [clip, 0], "vae": [vae_v, 0], "audio_vae": [vae_a, 0],
        "prompt": ported_prompt, "width": width, "height": height,
        "length": length, "ref_image_size": ref_image_size,
    }
    for i, nid in enumerate(load_refs):
        r2v_inputs[f"ref_images.ref_image_{i}"] = [nid, 0]
    r2v, _ = g.add("MiniMaxH3ReferenceToVideo", r2v_inputs)

    noise, _ = g.add("RandomNoise", {"noise_seed": seed, "noise_mode": "gpu"})
    guider, _ = g.add("BasicGuider", {
        "model": [sched_model, 0], "conditioning": [r2v, 0]})

    sca, _ = g.add("SamplerCustomAdvanced", {
        "noise": noise, "guider": guider, "sampler": sampler,
        "sigmas": scheduler, "latent_image": [r2v, 1]})

    vad, _ = g.add("VAEDecode", {"samples": [sca, 0], "vae": [vae_v, 0]})
    vada, _ = g.add("VAEDecodeAudio", {"samples": [sca, 0], "vae": [vae_a, 0]})
    cvid, _ = g.add("CreateVideo", {
        "images": [vad, 0], "fps": float(FPS), "audio": [vada, 0],
        "bit_depth": 8})
    g.add("SaveVideo", {
        "video": [cvid, 0], "filename_prefix": "h3/out", "format": "auto",
        "codec": {"codec": "auto"}})
    return g.p


class ComfyRunner:
    """Boot ComfyUI on localhost and drive the /prompt API."""

    def __init__(self, comfy_dir="/ComfyUI", host="127.0.0.1", port=8188,
                 refs_dir="/h3-refs", ref_subdir="scene1"):
        self.comfy_dir = comfy_dir
        self.host = host
        self.port = port
        self.refs_dir = refs_dir
        self.ref_subdir = ref_subdir
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _stage(self, ref_names: list[str]) -> list[str]:
        input_dir = os.path.join(self.comfy_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        src_dir = os.path.join(self.refs_dir, self.ref_subdir)
        staged = []
        for name in ref_names:
            shutil.copy(os.path.join(src_dir, name),
                        os.path.join(input_dir, name))
            staged.append(name)
        return staged

    def start(self):
        python = shutil.which("python") or "python"
        log = open(os.path.join(self.comfy_dir, "comfy.log"), "ab")
        self.proc = subprocess.Popen(
            [python, os.path.join(self.comfy_dir, "main.py"),
             "--listen", self.host, "--port", str(self.port),
             "--disable-auto-launch"],
            stdout=log, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout: float = 3600.0):
        waited = 0.0
        while waited < timeout:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("ComfyUI exited early (check comfy.log)")
            try:
                if requests.get(f"{self.base_url}/system_stats", timeout=5).ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
            waited += 2
        raise TimeoutError("ComfyUI did not become ready in time")

    def _queue(self, prompt) -> str:
        r = requests.post(f"{self.base_url}/prompt",
                          json={"prompt": prompt, "client_id": "modal-comfy"},
                          timeout=60)
        r.raise_for_status()
        return r.json()["prompt_id"]

    def _wait(self, prompt_id: str, timeout: float = 7200.0) -> dict:
        waited = 0.0
        while waited < timeout:
            try:
                data = requests.get(f"{self.base_url}/history/{prompt_id}",
                                    timeout=10).json()
                if data.get(prompt_id):
                    return data[prompt_id]
            except requests.RequestException:
                pass
            time.sleep(3)
            waited += 3
        raise TimeoutError(f"prompt {prompt_id} did not finish")

    @staticmethod
    def _find_mp4(history: dict, comfy_dir: str) -> str:
        output_dir = os.path.join(comfy_dir, "output")
        for node in (history.get("outputs") or {}).values():
            for key in ("videos", "gifs", "images"):
                for item in node.get(key, []) or []:
                    fn = item.get("filename") or item.get("subfilename")
                    subfolder = item.get("subfolder") or ""
                    if fn and fn.lower().endswith(".mp4"):
                        full = os.path.join(output_dir, subfolder, fn)
                        if os.path.exists(full):
                            return full
        raise RuntimeError(f"no mp4 in history outputs: {history}")

    def generate(self, *, prompt_file: str, ref_names: list[str],
                 duration: float, width: int, height: int, mode: str = "turbo",
                 steps: int | None = None, lora_strength: float = 1.0,
                 seed: int = 0, ref_image_size: str = "match") -> str:
        if steps is None:
            steps = 4 if mode == "turbo" else 20
        self._stage(ref_names)
        scene = open(prompt_file, encoding="utf-8").read()
        payload = build_prompt(
            ref_names=ref_names, ported_prompt=port_prompt(scene),
            width=width, height=height, length=frame_length_for(duration),
            ref_image_size=ref_image_size, mode=mode, steps=steps,
            lora_strength=lora_strength, seed=seed)
        pid = self._queue(payload)
        hist = self._wait(pid)
        return self._find_mp4(hist, self.comfy_dir)

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None