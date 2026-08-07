# AGENTS.md

Modal Labs pipeline that generates MiniMax-H3 **Ref2VA** clips (video + native audio) on an
A100-80GB and uploads the MP4 to S3. This is a remote-Modal app, not a local script: `main.py`
runs inside Modal containers.

## Run commands
- Stage scene reference images (once): `modal volume put h3-refs ./scene1 /scene1`
- **Diffusers ground truth (H200, ~$3.6/clip):** `modal run main.py`
- **Cheap ComfyUI path (L40S):**
  - Bootstrap model volume (one-time, ~36GB): `modal run comfy_app.py::download_models`
  - Generate Scene 1 (R2V, full/20, max refs):  `modal run api.py::batch --scene 1` (or `comfy_app.py::main`)
  - T2V (no refs, fl2va), 5s/864×480:     `modal run api.py::batch --task t2v --prompt "<text>"`
  - Multi-prompt T2V warm batch:          `modal run api.py::t2v_multi --prompts $'a\nb'`
  - Batch R2V variant takes on one warm container: `modal run api.py::batch --scene 1 --variants 4`
  - R2V with external prompt + refs: `modal run api.py::batch --prompt-file "Drunken Master/alley.txt" --ref-names "drunken_master.png,alleyway_henchman.png" --ref-subdir drunken_master --duration 10`
  - Multi-scene R2V on ONE warm container: `modal run api.py::r2v_multi --scenes "Drunken Master/scenes.json"` (JSON manifest; per-scene `prompt_file`/`ref_names[]`/`duration`/`width`/`height`/`ref_image_size`/`ref_subdir`)
  - **Full command/param reference: see `COMMANDS.md`.**
- Local checks only (`.venv` has no torch/diffusers/GPU; Modal runs the rest):
  - `python -m py_compile comfy_gen.py comfy_app.py api.py`
  - `python -c "import comfy_app, api"` (validates the Modal DSL, needs the `.venv`-installed `modal` + `requests`)

## ComfyUI-on-Modal path (cheap + scriptable; separate from `main.py`)
- **Purpose:** same Ref2VA task on L40/A10 via `Comfy-Org/ComfyUI` (pinned commit), driven headless
  through the `/prompt` API by a server bound to `127.0.0.1`. `comfy_app.py` is intentionally
  **self-contained** (graph builder + ComfyRunner inlined) so Modal never has to ship a sibling
  module into the worker. Not wan-GPU-free; fast enough for iteration/approval boards.
- **Model download is a runtime step, NOT baked into the image.** The image is software-only (ComfyUI +
  deps + Turbo nodes). Populate the volume once with `modal run comfy_app.py::download_models`
  (~36GB); `H3Generator` guards on missing files. A fresh deploy does NOT auto-populate.
- **Model files** live on the `comfyui-models` volume mounted at `/ComfyUI/models`
  (`diffusion_models/{minimax_h3_ref2va_pruned_int8_convrot,minimax_h3_fl2va_pruned_int8_convrot}`,
  `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq`,
  `vae/{video_vae_fp16,audio_vae_fp32}`, `loras/minimax_h3_turbo_4step_ema_ckpt850`).
- **Tag vocabulary differs from diffusers:** Comfy R2V node uses `<Picture N>` (1-based), NOT
  `<Subject N>`. `port_prompt()` (in `comfy_app.py`) strips the header at `subject_definitions:` and
  remaps tags. Keep ref order in lockstep — never reorder without renumbering `<Picture N>`.
- **R2V is full-quality only — no turbo.** The turbo LoRA is T2V/I2V-only and unsupported for
  Reference-to-Video, so `build_prompt` (R2V) always uses `res_multistep`/20 steps; passing `--mode
  turbo`/`--steps <20` on an R2V run warns and coerces to full/20. Turbo (4-step LoRA sampler,
  cheap/preview) is documented for **T2V only**. Pin `COMFYUI_COMMIT` + `TURBO_NODES_COMMIT` in
  `comfy_app.py`.
- **R2V refs:** default `ref_image_size="max"` (2048px short edge, strong identity; refs are
  pre-scaled to that in `scene1/`). `match` (scale refs to canvas; cheap, weak identity) is kept only
  as an explicit scout flag — `max` is the R2V default.
- **SageAttention is always-on** (`--use-sage-attention`; image adds `sageattention` + `triton`). Confirmed
  active on GPU via `comfy_app.py::check_sage`. Numerically different output than pytorch attention at same
  seed (expected). Measured ~2% at 5s/864×480 (135 vs 138s); real win at high-res/long-frame only.
- **T2V task:** `task="t2v"` uses the fl2va diffusion model and `MiniMaxH3ImageToVideo` (no refs, no
  `<Picture>` tags). Prompt is a T2VA-style body (`integrated_multimodal_description:` +
  `overall_soundscape:` + `non_diegetic_music:`). `t2v_multi` loops prompts on one warm container.
- **Frame snap:** `frame_length_for()` snaps (`%17==5`); 12.25s→294, 5s→124, 15s→362 (5–15s validated).
  864×480 cheap canvas (default for all tasks). R2V refs default to `ref_image_size="max"` (2048px
  short edge); `match` is scout-only.
- **Verified on Modal (L40S):** R2V full/20 and T2V turbo burns both succeed and upload valid MP4s
  (h264 @24fps + native AAC audio). T2V turbo 4-step sampling took ~0.5–1.5 min; R2V full ~8–10 min.
  First-model-load on a fresh worker ~1–2 min.

## Secrets (scoped — keep separate)
- `huggingface-secret` → on the **image build** only (download step), provides `HF_TOKEN`.
- `video-gen-secret` → on the **class** only, provides `AWS_REGION` + `S3_BUCKET_NAME` for upload.

## Non-obvious architecture / gotchas
- **R2V is the primary path; T2V uses the separate fl2va transformer.** `transformer_ref/` is the R2V
  transformer (full-quality only). T2V (`task="t2v"`) loads the fl2va `transformer/`; do not load both
  in one worker (each claims ~66GB). Shared `text_encoder` (Qwen3VL-32B) + VAEs plus one
  task-specific transformer per task.
- **Reference order is semantic.** `ref_image_paths` order maps positionally to `<Subject N>` in the
  prompt. Do not reorder an existing list without renumbering every `<Subject N>` reference.
- **Never feed the prompt-file header to the model.** `scene_01_*.txt` has a title/Source/NOTE
  preamble that is authoring metadata, not MiniMax format. main.py strips it via
  `prompt[prompt.index("subject_definitions:"):]` — keep this slice if you touch that line.
- **Prompt format** is defined by `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` (Ref2VA six sections) and
  `_base_en.md` (T2VA/I2VA/FL2VA). Rewrite output must start at `subject_definitions:`.
- **H3 knobs (verified against the pinned diffusers PR):** `num_frames` snaps up to `17*n+5`
  (`%17==5`); 288→294 = 12.25s, and must stay in 5–15s. `height`/`width` must be multiples of 32.
  960×544 is the cheap canvas; 768×1344 is MAX_PIXELS (~2× cost). `num_inference_steps=50`.
- **Cost/time:** ~15–18 min/clip at 960×544/50 steps (~$0.60–$0.75). Cold start downloads ~144GB
  once into the `MiniMax-H3-models` volume. Runtime needs `memory=98304` (96G RAM) for the
  CPU-offload footprint (~76GB); GPU VRAM peak stays <45GB.
- **Offload:** `load_components(..., pretrained_model_name_or_path=MODEL_ROOT)` forces the shared
  volume and avoids re-downloading ~144GB from the hub. Keep it.
- **`av` (PyAV) is required** for `encode_video` and reference-media decode — it's pip-installed
  after the model-download build step on purpose; don't remove that step.

## Key files
- `main.py` — diffusers pipeline (download/load/generate/upload) + Scene 1 entrypoint (H200 ground truth).
- `comfy_app.py` — cheap ComfyUI-on-Modal app: image build, model-volume bootstrap, `H3Generator` class, stdlib graph builder + `ComfyRunner`.
- `comfy_gen.py` — standalone copy of the merged generation logic (port/frame-snap/graph/runner) kept for local smoke tests; runtime uses the inlined copy. Not a dependency.
- `api.py` — scriptable batch entrypoint (`SCENES` manifest, `generate_scene`, `batch`
  `--prompt-file/--ref-names/--ref-subdir`, multi-scene `r2v_multi --scenes scenes.json`;
  `--variants/--steps/--seed`, COMMANDS.md).
- `Drunken Master/` — external Ref2VA project: `alley.txt` + `kitchen.txt` (verbatim
  prompts, start at `subject_definitions:`), `Assets/` (ref PNGs staged to volume
  subdir `drunken_master`), `scenes.json` manifest for `api.py::r2v_multi`.
- `scene_01_dealership_minimax_ref.txt` — Scene 1 Ref2VA prompt (7 `<Subject N>` refs).
- `car_commercial_sample_breakdown.txt` — source multi-scene beat breakdown (Higgsfield/Seedance).
- `VIDEO_PROMPT_WRITING_GUIDE_*_en.md` — H3 prompt-format specs (source of truth for prompt shape).
- `sample_script.py` — unrelated MoCha/Wan2.1 prototype; not MiniMax-H3. Ignore.
- `requirements.txt` — pins diffusers `PR 14355` (experimental modular pipeline). Don't change
  without re-verifying the Ref2VA API surface.