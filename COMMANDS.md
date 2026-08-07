# MiniMax-H3 — Command Reference

Modal/L40S ComfyUI pipeline for generating MiniMax-H3 clips (video + native
audio) and uploading the MP4 to S3. Two tasks: **R2V** (reference-to-video,
Scene 1) and **T2V** (text-to-video, fl2va). `api.py` is the primary scriptable
entrypoint. All commands below are run from the repo root using the local venv.

SageAttention is **always-on** via `--use-sage-attention` (image builds with
`sageattention` + `triton`); neutral on small canvases, helpful at high res.

## Prerequisites
- Stage Scene-1 reference images (one-time): `modal volume put h3-refs ./scene1 /scene1`
- Populate the model volume (one-time, ~36GB): `modal run comfy_app.py::download_models`
- Secrets (create if missing): `huggingface-secret` (image build), `video-gen-secret` (runtime, `AWS_REGION` + `S3_BUCKET_NAME`).

## Entrypoints

### Models bootstrap (one-time)
```
modal run comfy_app.py::download_models
```

### Single / batch scene generation (default: ComfyUI, L40S, turbo)
```
modal run api.py --scene 1                           # 1 variant, turbo, 4 steps
modal run api.py --scene 1 --variants 4                # 4 distinct takes on one warm container
modal run api.py --scene 1 --variants 4 --steps 8      # 8-step takes
modal run api.py --scene 1 --variants 2 --mode full    # 2x 20-step finals (new default)
modal run api.py --scene 1 --variants 2 --mode full --steps 50   # legacy 50-step
modal run api.py --scene 1 --variants 3 --seed 42      # fixed seed base -> 42,43,44
```
After the batch completes it prints all variants as `#i  seed  s3_key`, then a
bare list of the S3 keys. Pick the best take, then re-render its `--seed` with
`--mode full` (20-step) for finals.

### T2V (text-to-video; no reference images; fl2va model)
```
modal run api.py::batch --task t2v --prompt "<text>"                          # turbo 4-step, 5s/864x480
modal run api.py::batch --task t2v --mode full --duration 5 --width 864 --height 480 --prompt "<text>"  # 20-step finals
modal run api.py::batch --task t2v --mode full --duration 5 --width 864 --height 480 --seed 12345 --prompt "<text>"  # fixed seed
```
Multi-prompt batch on ONE warm container (each take = fresh random seed; not
exposed via `batch` for T2V, so use `t2v_multi`):
```
modal run api.py::t2v_multi --prompts $'prompt line 1\nprompt line 2'
```

### Single clip — comfy_app main (verbose; also turbo/full)
```
modal run comfy_app.py::main                    # --turbo true (default)
modal run comfy_app.py::main -- --turbo false   # full / 20-step
```

### Diagnostics
```
modal run comfy_app.py::download_models   # one-time model volume bootstrap (~36GB)
modal run comfy_app.py::verify_clips       # ffprobe-check S3 clips for codecs/fps/duration
modal run comfy_app.py::check_sage         # boot on GPU + confirm "Using sage attention"
```

### Diffusers ground truth (H200, costlier — do NOT change `main.py`)
```
modal run main.py
```

### Local checks (no GPU; `.venv` has no torch/diffusers)
```
.venv/bin/python -m py_compile comfy_app.py api.py comfy_gen.py
.venv/bin/python -c "import comfy_app, api"
.venv/bin/modal run api.py --help
```

## Parameter reference — `api.py::batch`
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--scene` | str | `"1"` | Scene id from the `SCENES` manifest (currently only `"1"`). R2V only. |
| `--variants` | int | `1` | Number of takes to render. |
| `--mode` | str | `"turbo"` | `turbo` (4-step distilled LoRA, cheap/preview) or `full` (20-step `res_multistep`, finals). |
| `--steps` | int | `None` | Override sampler steps. Omitted -> `4` (turbo) / `20` (full). |
| `--seed` | int | `None` | Seed base. Omitted -> random. Variants use `base .. base+N-1`. |
| `--task` | str | `"r2v"` | `r2v` (uses SCENES refs) or `t2v` (needs `--prompt`; fl2va model). |
| `--prompt` | str | `None` | T2V prompt body (T2VA format). Required when `--task t2v`. |
| `--duration` | float | `12.25` | Seconds; frame length snaps to `%17==5`. T2V: pass `5`. |
| `--width`/`--height` | int | `864`/`480` | Must be multiples of 32. Default canvas is 864×480 (0.4MP); 768×1344 = MAX_PIXELS. |
| `--ref-image-size` | str | `"match"` | R2V: `match` (scale refs to canvas, cheaper) or `max` (2048px short edge, stronger identity). |
| `--lora-strength` | float | `1.0` | Turbo LoRA strength (turbo mode only). |

## Parameter reference — `comfy_app.py::main`
| Flag | Default | Description |
|------|---------|-------------|
| `--turbo` | `true` | `false` -> full/20-step. |
| `--steps` | omit | Step override (4 turbo / 20 full when omitted). |
| `--duration` | `12.25` | Seconds; frame length snaps to `%17==5` (5–15s valid). |
| `--width`,`--height` | `864`,`480` | Must be multiples of 32 (`768x1344` = MAX_PIXELS). |
| `--seed` | `0` | Fixed seed. |
| `--ref-image-size` | `"match"` | See above. |

## Runtime knobs (comfy_app.py constants)
- `COMFYUI_COMMIT`, `TURBO_NODES_COMMIT` — pinned SHAs; must be 40 chars.
- `REFS` — Scene-1 reference order; maps 1:1 to `<Picture 1..7>`. Never reorder.
- Canvas: 864×480 (default; cheap) vs 768×1344 (MAX_PIXELS, ~2× cost).
- Frames: `17*n+5` grid at 24fps; 12.25s -> 294; 5s -> 124.
- SageAttention: always-on `--use-sage-attention`; image adds `sageattention` + `triton`.

## Cost / time (per your Modal pricing, L40S = $0.000542/s)
| Use | Est. time | Est. cost |
|-----|-----------|-----------|
| Turbo 4-step (864×480, 12.25s) | ~3.9 min gen (+ ~1 min boot, excl. queue) | ~$0.16 |
| Turbo 8-step | ~6.3–7.5 min | ~$0.22–0.26 |
| Turbo 4-step T2V (5s, 864×480) | ~0.5–1.5 min gen | ~$0.03–0.06 |
| Full 20-step T2V (5s, 864×480) | ~2.3 min gen (measured 135–138s) | ~$0.08 |
| Full 20-step (864×480, 12.25s) | ~8 min gen | ~$0.26–0.30 |
| Full 50-step (864×480, 12.25s) | ~20–35 min | ~$1.35–1.50 |
| Variant batch (N turbo, warm container) | 1 boot + N×~2.9 min | ~$0.16 × N |

Tips: batches reuse one warm container (1 boot, no reload). Queue wait is not
billed. Prefer `turbo` to scout then `full` (20-step) for finals. SageAttention
is always-on and measured ~neutral (2%) at 5s/864×480; its 1.5–2× win shows up
at high-res/long-frame canvases.