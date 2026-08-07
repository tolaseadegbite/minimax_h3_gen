import modal
import os
import tempfile
import uuid

# Must be set before torch initializes CUDA (it's imported lazily inside the
# container), otherwise expandable_segments has no effect and up to ~6GB stays
# stranded as reserved-but-unallocated fragmentation — enough to OOM the
# 7-reference text-encode. Default so it can't clobber an image-level override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

app = modal.App("minimax-h3-ref2va-video-generator")
volume = modal.Volume.from_name("MiniMax-H3-models", create_if_missing=True)
refs_volume = modal.Volume.from_name("h3-refs", create_if_missing=True)
MODEL_DIR = "/models"
MODEL_NAME = "MiniMaxAI/MiniMax-H3"
MODEL_ROOT = f"{MODEL_DIR}/MiniMax-H3"
REFS_DIR = "/refs"


def download_model():
    from huggingface_hub import snapshot_download

    # Ref2VA-only: the Ref2VA partition differs from FL2VA only in the transformer
    # weights, so we pull `transformer_ref/*` and skip `transformer/*` (FL2VA/T2VA).
    print("Downloading MiniMax-H3 Ref2VA components...")
    snapshot_download(
        MODEL_NAME,
        allow_patterns=[
            "model_index.json",
            "modular_model_index.json",
            "transformer_ref/*",
            "text_encoder/*",
            "tokenizer/*",
            "processor/*",
            "vae/*",
            "audio_vae/*",
            "scheduler/*",
            "audio_scheduler/*",
        ],
        local_dir=MODEL_ROOT,
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    print("Model download complete.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands("python -m pip install --upgrade pip wheel setuptools")
    .pip_install_from_requirements("requirements.txt")
    .run_function(
        download_model,
        volumes={MODEL_DIR: volume},
        secrets=[modal.Secret.from_name("huggingface-secret")],
        timeout=3600,
    )
    .pip_install(["av"])
)

s3_secret = modal.Secret.from_name("video-gen-secret")


@app.cls(
    image=image,
    secrets=[s3_secret],
    gpu="H200",
    memory=98304,
    volumes={MODEL_DIR: volume, REFS_DIR: refs_volume},
    timeout=7200,
    scaledown_window=300,
)
class VideoGenerator:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import (
            AutoencoderKLMiniMaxH3,
            AutoencoderKLMiniMaxH3Audio,
            ComponentsManager,
            MiniMaxH3Transformer3DModel,
            TorchAoConfig,
        )
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import (
            Qwen3VLForConditionalGeneration,
            TorchAoConfig as TransformersTorchAoConfig,
        )

        transformer_quant = TorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "proj_in",
                "audio_proj_in",
                "context_embedder",
                "time_embedder",
                "time_proj",
                "token_refiner",
                "norm_out",
                "proj_out",
                "audio_proj_out",
            ],
        )
        encoder_quant = TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "model.visual",
                "model.language_model.embed_tokens",
                "model.language_model.norm",
                "lm_head",
            ],
        )

        print("Initializing MiniMax-H3 Ref2VA ModularPipeline (int8)...")
        manager = ComponentsManager()
        self.pipe = MiniMaxH3Ref2VABlocks().init_pipeline(
            MODEL_ROOT, components_manager=manager
        )
        self.pipe.update_components(
            transformer_ref=MiniMaxH3Transformer3DModel.from_pretrained(
                f"{MODEL_ROOT}/transformer_ref",
                torch_dtype=torch.bfloat16,
                quantization_config=transformer_quant,
                low_cpu_mem_usage=True,
            ),
            text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
                f"{MODEL_ROOT}/text_encoder",
                torch_dtype=torch.bfloat16,
                quantization_config=encoder_quant,
            ),
            vae=AutoencoderKLMiniMaxH3.from_pretrained(
                f"{MODEL_ROOT}/vae", torch_dtype=torch.bfloat16
            ),
            audio_vae=AutoencoderKLMiniMaxH3Audio.from_pretrained(
                f"{MODEL_ROOT}/audio_vae", torch_dtype=torch.bfloat16
            ),
        )
        self.pipe.load_components(
            dtype=torch.bfloat16, pretrained_model_name_or_path=MODEL_ROOT
        )
        # Sequential-style offloading: evict every other resident component before
        # the active one loads, so only the model in use is on the GPU. The stock
        # AutoOffloadStrategy only evicts when the incoming model's own footprint
        # exceeds the free budget, so a small int8 text_encoder (~17GB) would load
        # on top of the resident transformer_ref/VAEs (~72GB) and OOM during the
        # 7-reference prompt encode. Returning all other on-device hooks is safe:
        # pre_forward pre-filters to resident, non-group-offloaded components.
        def offload_all_others(hooks, model_id, model, execution_device):
            return hooks

        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="24GB")
        # Inject the strategy through whichever API the installed diffusers commit
        # exposes (prefer set_offload_strategy; fall back to the hook attribute
        # that CustomOffloadHook.pre_forward reads directly).
        if hasattr(manager, "set_offload_strategy"):
            manager.set_offload_strategy(offload_all_others)
        else:
            for user_hook in manager.model_hooks:
                user_hook.hook.offload_strategy = offload_all_others
        print("MiniMax-H3 Ref2VA ready.")

    @modal.method()
    def generate(
        self,
        prompt: str,
        ref_image_paths: list[str],
        num_frames: int = 288,
        height: int = 544,
        width: int = 960,
        seed: int = 0,
    ):
        import boto3
        import torch
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
        from diffusers.utils.export_utils import encode_video

        # References are passable directly as in-container paths/URLs; each is decoded
        # (and its resolution normalized) when the reference is built. Order is semantic.
        references = [MiniMaxH3Reference(image=p) for p in ref_image_paths]

        print("Generating clip (ref2va)...")
        state = self.pipe(
            prompt=prompt,
            references=references,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=50,
            generator=torch.Generator().manual_seed(seed),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, "clip.mp4")
            encode_video(
                state.get("videos")[0],
                fps=24,
                output_path=local_path,
                audio=state.get("audio")[0],
                audio_sample_rate=state.get("sampling_rate"),
            )

            video_uuid = uuid.uuid4()
            s3_client = boto3.client("s3", region_name=os.environ["AWS_REGION"])
            bucket_name = os.environ["S3_BUCKET_NAME"]
            s3_key = f"videos/h3_ref2va_{video_uuid}.mp4"
            print(f"Uploading to S3: {s3_key}")
            s3_client.upload_file(local_path, bucket_name, s3_key)
            print("Upload complete.")
            return s3_key


@app.local_entrypoint()
def main():
    # Scene 1 - The Dealership (Ref2VA). References are passed in-prompt order:
    # each maps to a <Subject N> in scene_01_dealership_minimax_ref.txt. Stage the
    # images on the `h3-refs` volume first, e.g.:
    #   modal volume put h3-refs ./scene1 /scene1
    ref_image_paths = [
        f"{REFS_DIR}/scene1/hero_secretive.png",      # -> <Subject 1>
        f"{REFS_DIR}/scene1/manager.png",             # -> <Subject 2>
        f"{REFS_DIR}/scene1/car_sheet.png",           # -> <Subject 3>
        f"{REFS_DIR}/scene1/key_fob.png",             # -> <Subject 4>
        f"{REFS_DIR}/scene1/champion_photo.png",      # -> <Subject 5>
        f"{REFS_DIR}/scene1/dealership.png",          # -> <Subject 6>
        f"{REFS_DIR}/scene1/logo.png",                # -> <Subject 7>
    ]
    prompt = open("scene_01_dealership_minimax_ref.txt", encoding="utf-8").read()
    prompt = prompt[prompt.index("subject_definitions:"):]

    generator = VideoGenerator()
    s3_key = generator.generate.remote(
        prompt=prompt, ref_image_paths=ref_image_paths, num_frames=288
    )
    print(f"Clip generated: {s3_key}")