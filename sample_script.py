import modal
import os
import subprocess
import tempfile
import uuid

# --- App and Volume Setup ---
app = modal.App("video-generator-mocha-animate")
volume = modal.Volume.from_name("Wan2.1-mocha-models", create_if_missing=True)
MODEL_DIR = "/models"
WAN_MODEL_NAME = "Wan-AI/Wan2.1-T2V-14B"
MOCHA_MODEL_NAME = "Orange-3DV-Team/MoCha"


# --- Preprocessing Helpers ---
def _process_video_frames(video_path, height=480, width=832, num_frames=81):
    import imageio
    import torch
    from torchvision.transforms import v2
    from einops import rearrange
    from PIL import Image

    frame_process = v2.Compose([
        v2.CenterCrop(size=(height, width)),
        v2.Resize(size=(height, width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    if total_frames < num_frames:
        num_frames = 1 + (total_frames - 1) // 4 * 4
        print(f"⚠️ Source video shorter than requested. Using {num_frames} frames.")

    frames = []
    for i in range(num_frames):
        frame = reader.get_data(i)
        frame = Image.fromarray(frame)
        w, h = frame.size
        scale = max(width / w, height / h)
        frame = frame.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        frame = frame_process(frame)
        frames.append(frame)

    reader.close()
    frames = torch.stack(frames, dim=0)
    frames = rearrange(frames, "T C H W -> C T H W")
    return frames


def _process_mask(mask_path, height=480, width=832):
    import torch
    from torchvision.transforms import v2
    from PIL import Image

    mask_h = height // 8
    mask_w = width // 8

    mask_process = v2.Compose([
        v2.CenterCrop(size=(mask_h, mask_w)),
        v2.Resize(size=(mask_h, mask_w), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    mask = Image.open(mask_path).convert("RGB")
    w, h = mask.size
    scale = max(mask_w / w, mask_h / h)
    mask = mask.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    mask = mask_process(mask)
    mask = mask.unsqueeze(1)
    mask_cond = torch.sign(mask[0:1, 0:1, :, :]).repeat(16, 1, 1, 1)
    return mask_cond


def _process_reference_image(image_path, height=480, width=832):
    import torch
    from torchvision.transforms import v2
    from PIL import Image

    frame_process = v2.Compose([
        v2.CenterCrop(size=(height, width)),
        v2.Resize(size=(height, width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    scale = max(width / w, height / h)
    image = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    image = frame_process(image)
    image = image.unsqueeze(1)
    return image


# --- Model Download Function ---
def download_models():
    from huggingface_hub import snapshot_download

    print(f"Downloading {WAN_MODEL_NAME} to {MODEL_DIR}...")
    snapshot_download(
        WAN_MODEL_NAME,
        local_dir=f"{MODEL_DIR}/Wan2.1-T2V-14B",
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Downloading {MOCHA_MODEL_NAME} to {MODEL_DIR}...")
    snapshot_download(
        MOCHA_MODEL_NAME,
        local_dir=f"{MODEL_DIR}/MoCha",
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    ckpt_path = f"{MODEL_DIR}/MoCha/preview/step18500.ckpt"
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"MoCha checkpoint missing after download: {ckpt_path}")

    print("✅ All models downloaded and verified.")


# --- Image Definition ---
image = (
    modal.Image
    .from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;9.0",
    })
    .apt_install("git", "ffmpeg", "clang", "libaio-dev", "libsndfile1")
    .run_commands("python -m pip install --upgrade pip wheel setuptools")
    .pip_install_from_requirements("requirements.txt")
    .pip_install([
        "diffsynth",
        "imageio",
        "imageio[ffmpeg]",
        "einops",
    ])
    .run_function(
        download_models,
        volumes={MODEL_DIR: volume},
        timeout=3600,
    )
)

s3_secret = modal.Secret.from_name("video-gen-secret")


# --- Video Generation Class ---
@app.cls(
    image=image,
    secrets=[s3_secret],
    gpu="A100-80GB",
    volumes={MODEL_DIR: volume},
    timeout=3600,
    scaledown_window=300,
)
class VideoGenerator:
    @modal.enter()
    def load_model(self):
        import torch
        from diffsynth import ModelManager, WanVideoMoChaPipeline

        print("Loading Wan2.1-T2V-14B base model...")
        wan_dir = f"{MODEL_DIR}/Wan2.1-T2V-14B"

        shard_paths = [
            f"{wan_dir}/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
            for i in range(1, 7)
        ]
        t5_path = f"{wan_dir}/models_t5_umt5-xxl-enc-bf16.pth"
        vae_path = f"{wan_dir}/Wan2.1_VAE.pth"

        for p in shard_paths + [t5_path, vae_path]:
            if not os.path.exists(p):
                raise RuntimeError(f"Wan2.1 model file missing: {p}")

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([shard_paths, t5_path, vae_path])

        print("Creating MoCha pipeline...")
        self.pipe = WanVideoMoChaPipeline.from_model_manager(model_manager, device="cuda")

        print("Loading MoCha checkpoint...")
        ckpt_path = f"{MODEL_DIR}/MoCha/preview/step18500.ckpt"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.pipe.dit.load_state_dict(state_dict, strict=True)
        self.pipe.to("cuda")
        self.pipe.to(dtype=torch.bfloat16)

        del state_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print("✅ MoCha model loaded and ready.")

    @modal.method()
    def replace(
        self,
        source_video_url: str,
        mask_url: str,
        reference_image_url: str,
        second_reference_url: str = "",
        num_frames: int = 81,
        seed: int = 0,
    ):
        import boto3
        import requests

        NEGATIVE_PROMPT = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
            "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
            "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

        print("Starting MoCha character replacement...")

        with tempfile.TemporaryDirectory() as temp_dir:

            print("Downloading source video...")
            source_video_path = os.path.join(temp_dir, "source_video.mp4")
            if "youtube.com" in source_video_url or "youtu.be" in source_video_url:
                subprocess.run([
                    "yt-dlp", "-f", "best[ext=mp4]/best",
                    "-o", source_video_path, source_video_url,
                ], check=True)
            else:
                with requests.get(source_video_url, stream=True) as r:
                    r.raise_for_status()
                    with open(source_video_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

            print("Downloading mask image...")
            mask_path = os.path.join(temp_dir, "mask.png")
            with requests.get(mask_url, stream=True) as r:
                r.raise_for_status()
                with open(mask_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print("Downloading reference image...")
            ref1_path = os.path.join(temp_dir, "reference_1.png")
            with requests.get(reference_image_url, stream=True) as r:
                r.raise_for_status()
                with open(ref1_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            ref2_path = ""
            if second_reference_url:
                print("Downloading second reference image...")
                ref2_path = os.path.join(temp_dir, "reference_2.png")
                with requests.get(second_reference_url, stream=True) as r:
                    r.raise_for_status()
                    with open(ref2_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

            print("✅ Inputs downloaded. Preprocessing...")

            source_video = _process_video_frames(source_video_path, num_frames=num_frames)
            source_mask = _process_mask(mask_path)
            first_ref = _process_reference_image(ref1_path)
            second_ref = _process_reference_image(ref2_path) if ref2_path else None

            print(f"Source video shape: {source_video.shape}")
            print(f"Mask shape: {source_mask.shape}")
            print(f"First ref shape: {first_ref.shape}")
            if second_ref is not None:
                print(f"Second ref shape: {second_ref.shape}")

            print("Running MoCha pipeline...")
            video = self.pipe(
                prompt=" ",
                negative_prompt=NEGATIVE_PROMPT,
                source_video=source_video,
                source_mask=source_mask,
                first_ref=first_ref,
                second_ref=second_ref,
                cfg_scale=5.0,
                num_inference_steps=50,
                num_frames=num_frames,
                seed=seed,
                tiled=True,
            )

            print("✅ Generation complete. Saving and uploading...")

            from diffsynth import save_video
            local_video_path = os.path.join(temp_dir, "result.mp4")
            save_video(video, local_video_path, fps=30, quality=5)

            video_uuid = uuid.uuid4()
            s3_client = boto3.client("s3", region_name=os.environ["AWS_REGION"])
            bucket_name = os.environ["S3_BUCKET_NAME"]
            s3_key = f"videos/mocha_{video_uuid}.mp4"

            print(f"Uploading to S3: {s3_key}...")
            s3_client.upload_file(local_video_path, bucket_name, s3_key)

            print(f"✅ Upload complete: {s3_key}")
            return s3_key


# --- Entrypoint ---
@app.local_entrypoint()
def main():
    source_video_url = (
        "https://github.com/Wan-Video/Wan2.2/raw/main/examples/wan_animate/animate/video.mp4"
    )
    mask_url = (
        "https://example.com/mask.png"
    )
    reference_image_url = (
        "https://images.unsplash.com/photo-1685541003882-7328ef51bac8"
        "?w=627&auto=format&fit=crop"
    )

    generator = VideoGenerator()
    s3_key = generator.replace.remote(source_video_url, mask_url, reference_image_url)

    print(f"\n🚀 MoCha character replacement complete! S3 Key: {s3_key}")
