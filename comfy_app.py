"""Modal app: ComfyUI-backed MiniMax-H3 Ref2VA generation on cheap GPUs.

This module is intentionally self-contained (single file): ComfyUI is cloned
into the image (pinned), the local /prompt driver (ComfyRunner + graph builder)
lives here too, so Modal never has to ship a sibling module into the worker.
Model weights live on the ``comfyui-models`` volume (populated at runtime via
``download_models``) and reference images come from ``h3-refs``. Each container
boots ComfyUI on 127.0.0.1 and drives it via the /prompt API.

Commands
--------
- Populate model volume (one-time ~36GB):   modal run comfy_app.py::download_models
- Generate Scene 1 (R2V, 20-step finals):   modal run comfy_app.py::main
- T2V turbo (cheap, fl2va):                 modal run api.py --task t2v --prompt "<text>"

Sampling modes (T2V only)
-------------------------
  turbo  - MiniMax-H3-Turbo LoRA + 4-step dual-schedule sampler (cheap/iteration, preview quality)
  full   - stock res_multistep sampler, 20 steps by default (final hero renders; pass --steps to override)

R2V (ref2va) is full-quality only: it has NO turbo path (the turbo LoRA is
T2V/I2V-only and unsupported for Reference-to-Video), and references default
to ``ref_image_size="max"`` (2048px short edge) for strong identity. Passing
``mode=turbo`` on an R2V run warns and coerces to full/20.

Prompt reference tags are ported from the diffusers vocabulary ``<Subject N>``
to ComfyUI's ``<Picture N>``.
"""

import os
import re
import shutil
import subprocess
import time
import uuid

import requests
import modal

# Reproducible pins (commits validated to contain H3 + the Turbo nodes).
COMFYUI_COMMIT = "2eb609766a749e3104485979615e062e401bab97"
TURBO_NODES_COMMIT = "55f85c6dbe58b41aaf5ee610d225ecce0a00ee17"
assert len(COMFYUI_COMMIT) == 40 and len(TURBO_NODES_COMMIT) == 40, "pins must be 40-char SHAs"

# Volume root maps directly onto ComfyUI's models/ layout (the ComfyUI repo's
# shipped models/ dir is removed in the image build so the Volume can mount here).
MODELS_DIR = "/ComfyUI/models"
REFS_MOUNT = "/h3-refs"
COMFY_DIR = "/ComfyUI"

# Model / LoRA file names (must match files laid down by download_models).
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DIFFUSION = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
DIFFUSION_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TURBO_LORA = "minimax_h3_turbo_4step_ema_ckpt850.safetensors"

FPS = 24

# (ComfyUI subdir, local basename, hub repo, repo-relative filename).
# The graph nodes reference only the basename (ComfyUI appends models/<subdir>),
# while hf_hub_download needs the full repo-relative path.
_MODEL_FILES = [
    ("diffusion_models", DIFFUSION, "Comfy-Org/MiniMax-H3",
     "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    ("diffusion_models", DIFFUSION_FL2VA, "Comfy-Org/MiniMax-H3",
     "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    ("text_encoders", TEXT_ENCODER, "Comfy-Org/MiniMax-H3",
     "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ("vae", VIDEO_VAE, "Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_video_vae_fp16.safetensors"),
    ("vae", AUDIO_VAE, "Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_audio_vae_fp32.safetensors"),
    ("loras", TURBO_LORA, "larryvrh/MiniMax-H3-Turbo-Lora",
     "minimax_h3_turbo_4step_ema_ckpt850.safetensors"),
]

app = modal.App("minimax-h3-comfyui")
models_volume = modal.Volume.from_name("comfyui-models", create_if_missing=True)
refs_volume = modal.Volume.from_name("h3-refs", create_if_missing=True)

# The Scene-1 prompt txt ships bundled with the app image via a mount of the
# working directory (see ``image`` below), so prompt text is read on the client
# and passed as a string to generate_clip.


# ---------------------------------------------------------------------------
# Prompt + frame helpers (ComfyUI R2V vocabulary)
# ---------------------------------------------------------------------------
def frame_length_for(duration: float) -> int:
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
                 height: int, length: int, ref_image_size: str,
                 steps: int, seed: int) -> dict:
    """Return a ComfyUI /prompt payload (the ``prompt`` key) for R2V.

    R2V is full-quality only: res_multistep sampler, no turbo LoRA (the turbo
    LoRA is T2V/I2V-only and unsupported for Reference-to-Video).
    """
    assert 1 <= len(ref_names) <= 9, "H3 Ref2VA supports 1-9 reference images"
    g = _NodeGraph()

    load_refs = [g.add("LoadImage", {"image": n})[0] for n in ref_names]

    unet, _ = g.add("UNETLoader", {"unet_name": DIFFUSION, "weight_dtype": "default"})
    sampler, _ = g.add("KSamplerSelect", {"sampler_name": "res_multistep"})

    clip, _ = g.add("CLIPLoader", {
        "clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"})
    vae_v, _ = g.add("VAELoader", {"vae_name": VIDEO_VAE})
    vae_a, _ = g.add("VAELoader", {"vae_name": AUDIO_VAE})

    sched_model = unet
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
        "noise": [noise, 0], "guider": [guider, 0],
        "sampler": [sampler, 0], "sigmas": [scheduler, 0],
        "latent_image": [r2v, 1]})

    vad, _ = g.add("VAEDecode", {"samples": [sca, 0], "vae": [vae_v, 0]})
    vada, _ = g.add("VAEDecodeAudio", {"samples": [sca, 0], "vae": [vae_a, 0]})
    cvid, _ = g.add("CreateVideo", {
        "images": [vad, 0], "fps": float(FPS), "audio": [vada, 0],
        "bit_depth": 8})
    g.add("SaveVideo", {
        "video": [cvid, 0], "filename_prefix": "h3/out", "format": "auto",
        "codec": "auto"})
    return g.p


def build_t2v_prompt(*, prompt: str, width: int, height: int, length: int,
                     mode: str, steps: int, lora_strength: float,
                     seed: int) -> dict:
    """ComfyUI /prompt payload for MiniMax H3 Text-to-Video (fl2va model).

    Mirrors the R2V graph wiring but uses ``MiniMaxH3ImageToVideo`` with no
    ``first_frame``/``last_frame`` (pure T2V). The prompt is a plain T2VA body
    string and uses the fl2va diffusion checkpoint.
    """
    g = _NodeGraph()

    unet, _ = g.add("UNETLoader", {"unet_name": DIFFUSION_FL2VA,
                                   "weight_dtype": "default"})
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

    i2v, _ = g.add("MiniMaxH3ImageToVideo", {
        "clip": [clip, 0], "vae": [vae_v, 0], "prompt": prompt,
        "width": width, "height": height, "length": length})

    noise, _ = g.add("RandomNoise", {"noise_seed": seed, "noise_mode": "gpu"})
    guider, _ = g.add("BasicGuider", {
        "model": [sched_model, 0], "conditioning": [i2v, 0]})

    sca, _ = g.add("SamplerCustomAdvanced", {
        "noise": [noise, 0], "guider": [guider, 0],
        "sampler": [sampler, 0], "sigmas": [scheduler, 0],
        "latent_image": [i2v, 1]})

    vad, _ = g.add("VAEDecode", {"samples": [sca, 0], "vae": [vae_v, 0]})
    vada, _ = g.add("VAEDecodeAudio", {"samples": [sca, 0], "vae": [vae_a, 0]})
    cvid, _ = g.add("CreateVideo", {
        "images": [vad, 0], "fps": float(FPS), "audio": [vada, 0],
        "bit_depth": 8})
    g.add("SaveVideo", {
        "video": [cvid, 0], "filename_prefix": "h3/out", "format": "auto",
        "codec": "auto"})
    return g.p


class ComfyRunner:
    """Boot ComfyUI on localhost and drive the /prompt API."""

    def __init__(self, comfy_dir=COMFY_DIR, host="127.0.0.1", port=8188,
                 refs_dir=REFS_MOUNT, ref_subdir="scene1"):
        self.comfy_dir = comfy_dir
        self.host = host
        self.port = port
        self.refs_dir = refs_dir
        self.ref_subdir = ref_subdir
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _stage(self, ref_names):
        input_dir = os.path.join(self.comfy_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        src_dir = os.path.join(self.refs_dir, self.ref_subdir)
        for name in ref_names:
            shutil.copy(os.path.join(src_dir, name), os.path.join(input_dir, name))

    def start(self):
        python = shutil.which("python") or "python"
        log = open(os.path.join(self.comfy_dir, "comfy.log"), "ab")
        t0 = time.time()
        self.proc = subprocess.Popen(
            [python, os.path.join(self.comfy_dir, "main.py"),
             "--listen", self.host, "--port", str(self.port),
             "--disable-auto-launch", "--use-sage-attention"],
            stdout=log, stderr=subprocess.STDOUT)
        self._boot_t = t0

    def _boot_seconds(self) -> float:
        return time.time() - getattr(self, "_boot_t", time.time())

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

    def generate(self, *, prompt: str, ref_names: list[str],
                 duration: float, width: int, height: int, mode: str = "turbo",
                 steps: int | None = None, lora_strength: float = 1.0,
                 seed: int = 0, ref_image_size: str = "max",
                 task: str = "r2v") -> str:
        t = time.time()
        if task == "t2v":
            if steps is None:
                steps = 4 if mode == "turbo" else 20
            payload = build_t2v_prompt(
                prompt=prompt, width=width, height=height,
                length=frame_length_for(duration), mode=mode, steps=steps,
                lora_strength=lora_strength, seed=seed)
        else:
            if mode == "turbo" or (steps is not None and steps < 20):
                print(f"[warn] R2V has no turbo LoRA ({TURBO_LORA} is T2V-only); "
                      f"forcing full res_multistep / 20 steps.")
            mode, steps = "full", 20
            self._stage(ref_names)
            payload = build_prompt(
                ref_names=ref_names, ported_prompt=port_prompt(prompt),
                width=width, height=height, length=frame_length_for(duration),
                ref_image_size=ref_image_size, steps=steps, seed=seed)
        t_queue = time.time()
        pid = self._queue(payload)
        t_queued = time.time()
        hist = self._wait(pid)
        t_done = time.time()
        print(f"[timing] task={task} build_prompt={t_queue-t:.1f}s "
              f"queue={t_queued-t_queue:.1f}s generate={t_done-t_queued:.1f}s "
              f"total={t_done-t:.1f}s (steps={steps})")
        return self._find_mp4(hist, self.comfy_dir)

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


# ---------------------------------------------------------------------------
# Model volume bootstrap
# ---------------------------------------------------------------------------
def _bootstrap_models():
    """Populate the comfyui-models volume: writes ComfyUI's models/ layout."""
    from huggingface_hub import hf_hub_download

    for subdir, filename, repo, hub_path in _MODEL_FILES:
        target_dir = os.path.join(MODELS_DIR, subdir)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Downloading {repo} :: {hub_path}")
        local = hf_hub_download(
            repo_id=repo, filename=hub_path,
            resume_download=True, token=os.environ.get("HF_TOKEN"))
        shutil.copy(local, os.path.join(target_dir, filename))
    print("Model download complete.")


def _models_present() -> bool:
    return all(
        os.path.isfile(os.path.join(MODELS_DIR, subdir, filename))
        for subdir, filename, _repo, _hub in _MODEL_FILES
    )


# ---------------------------------------------------------------------------
# Image + Modal functions
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libnss3")
    .pip_install("huggingface_hub", "requests", "numpy", "accelerate",
                 "imageio-ffmpeg", "av", "boto3")
    .pip_install("sageattention", "triton")
    .run_commands(
        "git clone https://github.com/Comfy-Org/ComfyUI.git " + COMFY_DIR,
        f"git -C {COMFY_DIR} checkout {COMFYUI_COMMIT}",
        f"rm -rf {COMFY_DIR}/models",
        f"python -m pip install -r {COMFY_DIR}/requirements.txt",
        f"git clone https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo "
        f"{COMFY_DIR}/custom_nodes/ComfyUI-MiniMax-H3-Turbo",
        f"git -C {COMFY_DIR}/custom_nodes/ComfyUI-MiniMax-H3-Turbo "
        f"checkout {TURBO_NODES_COMMIT}",
    )
)


@app.function(
    image=image,
    volumes={MODELS_DIR: models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def download_models():
    """Manual (re)populate of the comfyui-models volume."""
    _bootstrap_models()


@app.cls(
    image=image,
    gpu="L40S",
    secrets=[modal.Secret.from_name("video-gen-secret")],
    volumes={MODELS_DIR: models_volume, REFS_MOUNT: refs_volume},
    scaledown_window=60,
    timeout=7200,
)
class H3Generator:
    @modal.enter()
    def start(self):
        self.runner = ComfyRunner()
        self.runner.start()
        self.runner.wait_ready()
        print(f"[timing] comfy_boot={self.runner._boot_seconds():.1f}s")

    @modal.exit()
    def stop(self):
        self.runner.stop()

    @modal.method()
    def generate_clip(self, prompt: str, ref_names: list[str],
                      duration: float = 12.25, width: int = 864,
                      height: int = 480, mode: str = "turbo",
                      steps: int | None = None, lora_strength: float = 1.0,
                      seed: int = 0, ref_image_size: str = "max",
                      task: str = "r2v") -> dict:
        if not _models_present():
            raise RuntimeError(
                "comfyui-models volume is missing model files. Run first:\n"
                "  modal run comfy_app.py::download_models")
        if task != "t2v" and (mode == "turbo" or (steps is not None and steps < 20)):
            print(f"[warn] R2V has no turbo LoRA ({TURBO_LORA} is T2V-only); "
                  f"forcing full res_multistep / 20 steps.")
            mode, steps = "full", 20
        t_gen = time.time()
        mp4 = self.runner.generate(
            prompt=prompt, ref_names=ref_names,
            duration=duration, width=width, height=height, mode=mode,
            steps=steps, lora_strength=lora_strength, seed=seed,
            ref_image_size=ref_image_size, task=task)
        t_up = time.time()
        import boto3
        key = f"videos/h3_comfy_{uuid.uuid4()}.mp4"
        bucket = os.environ["S3_BUCKET_NAME"]
        boto3.client("s3", region_name=os.environ["AWS_REGION"]).upload_file(
            mp4, bucket, key)
        t_end = time.time()
        print(f"[timing] generate_clip={t_up-t_gen:.1f}s upload={t_end-t_up:.1f}s "
              f"grand_total={t_end-t_gen:.1f}s")
        return {"s3_key": key, "seed": seed, "mode": mode,
                "duration": duration, "width": width, "height": height}


REFS = [
    "hero_secretive.png", "manager.png", "car_sheet.png", "key_fob.png",
    "champion_photo.png", "dealership.png", "logo.png",
]


@app.local_entrypoint()
def main(steps: int | None = None, duration: float = 12.25,
         width: int = 864, height: int = 480, seed: int = 0,
         ref_image_size: str = "max"):
    """R2V Scene 1: full-quality only (res_multistep, max refs, 20 steps).

    R2V has no turbo path (the turbo LoRA is T2V-only); pass --steps to
    override. For cheap T2V turbo, use api.py --task t2v.
    """
    gen = H3Generator()
    with open("scene_01_dealership_minimax_ref.txt", encoding="utf-8") as f:
        prompt = f.read()
    result = gen.generate_clip.remote(
        prompt=prompt,
        ref_names=REFS, duration=duration, width=width, height=height,
        mode="full", steps=steps, seed=seed,
        ref_image_size=ref_image_size)
    print(result)


def _probe_key(bucket, region, key: str) -> str:
    import boto3, tempfile, subprocess
    import imageio_ffmpeg
    boto3.client("s3", region_name=region).download_file(
        bucket, key, f"/tmp/{key.split('/')[-1]}")
    p = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i",
         f"/tmp/{key.split('/')[-1]}"],
        capture_output=True, text=True)
    stderr = p.stderr
    lines = [l for l in stderr.splitlines()
             if "Stream" in l or "Duration" in l]
    return key + "\n" + "\n".join(lines) + "\n"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("video-gen-secret")],
    timeout=600,
)
def verify_clips(keys: list[str]) -> str:
    import os
    bucket = os.environ["S3_BUCKET_NAME"]
    region = os.environ["AWS_REGION"]
    return "\n".join(_probe_key(bucket, region, k) for k in keys)


@app.function(
    image=image,
    gpu="L40S",
    volumes={MODELS_DIR: models_volume},
    timeout=600,
)
def check_sage():
    """Boot ComfyUI and confirm whether Sage attention is active (or fell back)."""
    r = ComfyRunner()
    r.start()
    try:
        r.wait_ready(timeout=1800)
    except Exception:
        pass
    log = os.path.join(r.comfy_dir, "comfy.log")
    lines = []
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for kw in ("sage", "attention", "triton", "xformers",
                   "Traceback", "Error", "error", "cuda", "CUDA", "Failed"):
            hits = [l for l in text.splitlines() if kw.lower() in l.lower()]
            if hits:
                lines.append(f"-- [{kw}] {len(hits)} hit(s)")
                lines.extend(hits[-3:])
    else:
        lines.append(f"(no comfy.log at {log})")
    return "\n".join(lines) or "(no log lines)"


@app.function(
    image=image,
    gpu="L40S",
    timeout=600,
)
def probe_env():
    """Read-only toolchain probe: torch/CUDA/sage versions on the L40S image."""
    import torch
    out = [
        f"torch={torch.__version__}",
        f"cuda_build={torch.version.cuda}",
        f"cudnn={torch.backends.cudnn.version()}",
        f"device={torch.cuda.get_device_name(0)}",
        f"capability={'.'.join(map(str, torch.cuda.get_device_capability(0)))}",
    ]
    try:
        import sageattention
        out.append(f"sageattention={getattr(sageattention, '__version__', '?')}")
    except Exception as e:
        out.append(f"sageattention=NOT_IMPORTABLE ({type(e).__name__})")
    import triton
    out.append(f"triton={triton.__version__}")
    return "\n".join(out)