# MiniMax-H3 — Command Reference

Modal/L40S ComfyUI pipeline for generating MiniMax-H3 clips (video + native
audio) and uploading the MP4 to S3. Two tasks: **R2V** (reference-to-video,
Scene 1) and **T2V** (text-to-video, fl2va). `api.py` is the primary scriptable
entrypoint. All commands below are run from the repo root using the local venv.

**R2V is full-quality only** — it has NO turbo path (the turbo LoRA is
T2V/I2V-only and unsupported for Reference-to-Video). R2V always runs
`res_multistep` / 20 steps with refs at `ref_image_size="max"` (2048px short
edge) for strong identity; passing `--mode turbo` on R2V warns and coerces to
full. Turbo (4-step, cheap) is available **only** for T2V.

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

### Single / batch scene generation (R2V — full-quality only)
```
modal run api.py::batch --scene 1 --variants 1            # 1 R2V take, full/20, max refs
modal run api.py::batch --scene 1 --variants 4            # 4 distinct takes on one warm container
modal run api.py::batch --scene 1 --variants 2 --steps 30 # 2x 30-step finals
modal run api.py::batch --scene 1 --variants 3 --seed 42  # fixed seed base -> 42,43,44
```
After the batch completes it prints all variants as `#i  seed  s3_key`, then a
bare list of the S3 keys. Pick the best take and re-render with a higher canvas
for finals. R2V ignores turbo requests (warns + coerces to full/20).

#### External prompt + refs (`--prompt-file` / `--ref-names`)
```
modal run api.py::batch --prompt-file "Drunken Master/alley.txt" \
    --ref-names "drunken_master.png,alleyway_henchman.png" \
    --ref-subdir "drunken_master" --ref-image-size match --duration 10
```
`--ref-names` is a comma-separated list mapping positionally to `<Subject N>`
in the prompt file (keep ref order in lockstep with the tags). `--ref-subdir`
is the refs-volume subfolder to stage from (default `scene1`).

#### Multi-scene warm batch (`r2v_multi` — one warm L40S, back-to-back scenes)
```
modal run api.py::r2v_multi --scenes "Drunken Master/scenes.json"
```
`--scenes` is a JSON manifest, one entry per scene, looped on a single warm
container (re-boot happens only before the first scene):
```json
{
  "alley":   {"prompt_file": "Drunken Master/alley.txt",
              "ref_names": ["drunken_master.png","alleyway_henchman.png"],
              "duration": 10.0, "width": 864, "height": 480,
              "ref_image_size": "match", "ref_subdir": "drunken_master"},
  "kitchen": {"prompt_file": "Drunken Master/kitchen.txt",
              "ref_names": ["drunken_master.png","sushi_chef.png"],
              "duration": 10.0, "width": 864, "height": 480,
              "ref_image_size": "match", "ref_subdir": "drunken_master"}
}
```
Run `r2v_multi` via the remote-app path; each scene runs sequentially on the
same L40S (~7 min at 864×480/match per scene). Stage the scene's refs to the
volume subfolder first (e.g. `modal volume put -f h3-refs "Drunken Master/Assets" /drunken_master`).

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
Multi-file T2V, one whole prompt per file, read VERBATIM off disk (safest — no
shell line-splitting; each file = one take, so no fragment clips):
```
modal run api.py::t2v_files --paths "Vignettes/sweeper.txt,Vignettes/clock.txt" \
    --duration 10 --width 864 --height 480 --mode turbo
```
`t2v_multi --prompts` is one prompt per line and **aborts** if it sees collapsed
section headers (e.g. `subject_definitions:`, `[Shot N]`, `overall_soundscape:`)
— use `t2v_files` for multi-paragraph prompts.

### Single clip — comfy_app main (R2V, verbose)
```
modal run comfy_app.py::main                     # R2V full/20, max refs (default)
modal run comfy_app.py::main -- --steps 30       # 30-step override
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
| `--mode` | str | `"turbo"` | T2V only: `turbo` (4-step distilled LoRA, cheap/preview) or `full` (20-step `res_multistep`, finals). R2V ignores it (warns + coerces to full). |
| `--steps` | int | `None` | Override sampler steps. Omitted -> `20` (R2V) / `4` (T2V turbo) / `20` (T2V full). |
| `--seed` | int | `None` | Seed base. Omitted -> random. Variants use `base .. base+N-1`. |
| `--task` | str | `"r2v"` | `r2v` (uses SCENES refs, full-quality only) or `t2v` (needs `--prompt`; fl2va model; turbo OK). |
| `--prompt` | str | `None` | T2V prompt body (T2VA format). Required when `--task t2v`. |
| `--duration` | float | `12.25` | Seconds; frame length snaps to `%17==5`. T2V: pass `5`. |
| `--width`/`--height` | int | `864`/`480` | Must be multiples of 32. Default canvas is 864×480 (0.4MP); 768×1344 = MAX_PIXELS. |
| `--ref-image-size` | str | `"max"` | R2V: `max` (2048px short edge, strong identity — default) or `match` (scale refs to canvas, cheap scout only). |
| `--lora-strength` | float | `1.0` | Turbo LoRA strength (T2V turbo mode only). |
| `--prompt-file` | str | `None` | R2V: external prompt file (e.g. `Drunken Master/alley.txt`). Overrides SCENES manifest. |
| `--ref-names` | str | `None` | R2V: comma-separated ref filenames (positional, `<Subject N>` order). |
| `--ref-subdir` | str | `None` | R2V: refs-volume subfolder to stage from (default `scene1`). |

## Parameter reference — `comfy_app.py::main`
| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | omit | Step override (20 when omitted; R2V is full-only). |
| `--duration` | `12.25` | Seconds; frame length snaps to `%17==5` (5–15s valid). |
| `--width`,`--height` | `864`,`480` | Must be multiples of 32 (`768x1344` = MAX_PIXELS). |
| `--seed` | `0` | Fixed seed. |
| `--ref-image-size` | `"max"` | R2V refs: `max` (2048px short edge, default) or `match` (scout only). |

## Runtime knobs (comfy_app.py constants)
- `COMFYUI_COMMIT`, `TURBO_NODES_COMMIT` — pinned SHAs; must be 40 chars.
- `REFS` — Scene-1 reference order; maps 1:1 to `<Picture 1..7>`. Never reorder.
- Canvas: 864×480 (default; cheap) vs 768×1344 (MAX_PIXELS, ~2× cost).
- Frames: `17*n+5` grid at 24fps; 12.25s -> 294; 5s -> 124.
- SageAttention: always-on `--use-sage-attention`; image adds `sageattention` + `triton`.

## Cost / time (per your Modal pricing, L40S = $0.000542/s)
| Use | Est. time | Est. cost |
|-----|-----------|-----------|
| R2V full 20-step (864×480, 12.25s, max refs) | ~8–10 min gen (incl. max-ref encode) | ~$0.26–0.35 |
| R2V full 50-step | ~20–35 min | ~$1.35–1.50 |
| T2V turbo 4-step (5s, 864×480) | ~0.5–1.5 min gen | ~$0.03–0.06 |
| T2V full 20-step (5s, 864×480) | ~2.3 min gen (measured 135–138s) | ~$0.08 |

Tips: batches reuse one warm container (1 boot, no reload). Queue wait is not
billed. R2V is full-quality only (no turbo) — iterate cheaply on **T2V** or with
`--ref-image-size match` for rough staging, then commit to R2V full/max for
finals. SageAttention is always-on and measured ~neutral (2%) at 5s/864×480;
its 1.5–2× win shows up at high-res/long-frame canvases.