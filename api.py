"""Scriptable batch entrypoint for the ComfyUI Ref2VA pipeline.

Mirrors the Scene-1 reference/order contract of ``main.py`` (the diffusers
ground truth) but routes through the cheap ComfyUI/L40S path. Add more scenes
to SCENES to batch the whole commercial.

Render one or more variants of a scene (R2V: full-quality only, refs at
``max``, 20 steps) on a single warm L40S container, then pick the best take and
re-render its ``--seed`` at a higher canvas for finals. For cheap T2V iteration,
use ``--task t2v`` (turbo supported there).

Usage
-----
    modal run api.py::batch --scene 1 --variants 2      # R2V, full/20, max refs
    modal run api.py::batch --scene 1 --variants 4 --steps 20
    modal run api.py::batch --task t2v --prompt "..." --mode turbo  # cheap T2V takes
    modal run api.py::batch --prompt-file "path/x.txt" --ref-names "a.png,b.png" \
        --ref-subdir sub --duration 10
    modal run api.py::r2v_multi --scenes "Drunken Master/scenes.json"  # warm multi-scene
"""

from __future__ import annotations

import os
import random

import modal

from comfy_app import H3Generator, app, REFS  # noqa: F401 (app shared)

PROMPT_FILE = os.path.join(os.path.dirname(__file__),
                           "scene_01_dealership_minimax_ref.txt")

SCENES: dict[str, dict] = {
    "1": {
        "prompt_file": PROMPT_FILE,
        "ref_names": REFS,  # order maps 1:1 to <Picture 1..7>
        "duration": 12.25,
        "width": 864,
        "height": 480,
    },
}


def generate_scene(gen: H3Generator, scene_id: str, seed: int,
                   ref_image_size: str = "max",
                   ref_subdir: str | None = None) -> dict:
    """Generate one take of a scene on the cheap ComfyUI path; upload to S3.

    ``gen`` is passed in (rather than created here) so a batch loop reuses one
    warm container and re-reads the prompt only once. R2V is full-quality
    only; mode/steps are forced in ``H3Generator.generate_clip``.
    """
    scene = SCENES[scene_id]
    result = gen.generate_clip.remote(
        prompt=_read_prompt(scene["prompt_file"]),
        ref_names=scene["ref_names"],
        duration=scene["duration"],
        width=scene["width"],
        height=scene["height"],
        seed=seed,
        ref_image_size=ref_image_size,
        ref_subdir=ref_subdir,
        mode="full", steps=20,
    )
    return result


def _read_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _run_r2v(gen: H3Generator, scene: dict, seed: int) -> dict:
    """Render one R2V take from a scene dict (SCENES style or scenes.json)."""
    if not scene.get("prompt_file"):
        raise ValueError("scene dict requires 'prompt_file'")
    ref_names = scene["ref_names"]
    if isinstance(ref_names, str):
        ref_names = [n.strip() for n in ref_names.split(",") if n.strip()]
    return gen.generate_clip.remote(
        prompt=_read_prompt(scene["prompt_file"]),
        ref_names=ref_names,
        duration=scene.get("duration", 12.25),
        width=scene.get("width", 864),
        height=scene.get("height", 480),
        seed=seed,
        ref_image_size=scene.get("ref_image_size", "max"),
        ref_subdir=scene.get("ref_subdir"),
        mode="full", steps=20,
    )


@app.local_entrypoint()
def batch(scene: str = "1", variants: int = 1, mode: str = "turbo",
          steps: int | None = None, seed: int | None = None,
          ref_image_size: str = "max", lora_strength: float = 1.0,
          task: str = "r2v", prompt: str | None = None,
          duration: float = 12.25, width: int = 960, height: int = 544,
          prompt_file: str | None = None, ref_names: str | None = None,
          ref_subdir: str | None = None):
    """Generate ``variants`` takes of a scene on one warm L40S container.

    steps: omitted -> 20 (R2V always full) / 4 (turbo) or 20 (full) for T2V.
    seed:  None -> random base; variants use base .. base+N-1 (all distinct).
    task:  "r2v" (default, uses SCENES refs; full-quality only) or "t2v"
           (pure text, needs --prompt; turbo supported here).
    duration/width/height: override canvas+sine for t2v (e.g. 5s, 864x480 0.4MP).
    prompt_file + ref_names: override the SCENES manifest with an external
           prompt file and comma-separated ref filenames (e.g. the Drunken
           Master txt + png refs); ref names map positionally to <Subject N>.
    ref_subdir: refs volume subfolder to stage from (default scene1).
    Prints a per-variant summary plus the bare S3 keys after everything lands.
    """
    if variants < 1:
        raise ValueError(f"variants must be >= 1, got {variants}")
    if task == "t2v" and not prompt:
        raise ValueError("task=t2v requires --prompt <text>")
    if (prompt_file or ref_names) and task != "r2v":
        raise ValueError("prompt_file/ref_names are r2v-only")
    gen = H3Generator()
    base = seed if seed is not None else random_seed()
    results = []
    for i in range(variants):
        if task == "t2v":
            results.append(gen.generate_clip.remote(
                prompt=prompt, ref_names=None, mode=mode, steps=steps,
                seed=base + i, ref_image_size=ref_image_size,
                lora_strength=lora_strength, task="t2v",
                duration=duration, width=width, height=height))
        elif prompt_file:
            scene = {
                "prompt_file": prompt_file,
                "ref_names": ref_names,
                "duration": duration,
                "width": width,
                "height": height,
                "ref_image_size": ref_image_size,
                "ref_subdir": ref_subdir,
            }
            results.append(_run_r2v(gen, scene, seed=base + i))
        else:
            results.append(generate_scene(
                gen, scene, seed=base + i,
                ref_image_size=ref_image_size, ref_subdir=ref_subdir))

    eff_steps = steps if steps is not None else (4 if mode == "turbo" else 20)
    print(f"\nBatch complete ({variants} variant(s), task={task}, mode={mode}, "
          f"steps={eff_steps}, base_seed={base})")
    for i, r in enumerate(results):
        print(f"  #{i}  seed={r['seed']:<11} {r['s3_key']}")
    print("\nS3 keys:")
    for r in results:
        print(f"  {r['s3_key']}")


@app.local_entrypoint()
def t2v_multi(prompts: str, mode: str = "turbo",
              steps: int | None = None, duration: float = 5.0,
              width: int = 864, height: int = 480,
              ref_image_size: str = "max", lora_strength: float = 1.0):
    """Generate one T2V take per prompt line in a SINGLE client invocation.

    Unlike ``batch``, this loops all prompts on one warm L40S container so a
    back-to-back list of prompts shares the model and skips re-boot after the
    first. Pass prompts as newline-separated lines (e.g. ``--prompts $'...\n...'``).
    Each take uses a fresh random seed.

    Guard: a well-formed prompt is a single logical line. If ``--prompts``
    contains colon-headed section markers (``subject_definitions:``,
    ``integrated_multimodal_description:``, ``[Shot N]``) on their own lines,
    the caller almost certainly passed a multi-paragraph file through ``$()``;
    abort instead of firing jagged fragment clips.
    """
    lines = [p.strip() for p in prompts.splitlines()]
    non_empty = [p for p in lines if p]
    if not non_empty:
        raise ValueError("t2v_multi requires at least one non-empty --prompts line")
    frag_markers = ("subject_definitions:", "integrated_multimodal_description:",
                    "retention_analysis:", "detailed_description:",
                    "overall_soundscape:", "non_diegetic_music:", "[Shot")
    fragments = [p for p in non_empty if p.startswith(frag_markers)]
    if fragments:
        raise ValueError(
            "t2v_multi --prompts must be one prompt per line; got section "
            f"headers split across {len(fragments)} line(s) "
            "(looks like a multi-paragraph file was collapsed). Use "
            "api.py::t2v_files with --paths instead:\n"
            "  modal run api.py::t2v_files --paths a.txt,b.txt")
    prompts_list = non_empty
    gen = H3Generator()
    results = []
    for p in prompts_list:
        seed = random_seed()
        results.append(gen.generate_clip.remote(
            prompt=p, ref_names=None, mode=mode, steps=steps, seed=seed,
            ref_image_size=ref_image_size, lora_strength=lora_strength,
            task="t2v", duration=duration, width=width, height=height))

    eff_steps = steps if steps is not None else (4 if mode == "turbo" else 20)
    print(f"\nT2V multi complete ({len(results)} take(s), mode={mode}, "
          f"steps={eff_steps})")
    for i, r in enumerate(results):
        print(f"  #{i}  seed={r['seed']:<11} {r['s3_key']}")
    print("\nS3 keys:")
    for r in results:
        print(f"  {r['s3_key']}")


@app.local_entrypoint()
def t2v_files(paths: str, mode: str = "turbo", steps: int | None = None,
              duration: float = 10.0, width: int = 864, height: int = 480,
              ref_image_size: str = "max", lora_strength: float = 1.0):
    """One T2V take per prompt FILE, read verbatim, on ONE warm container.

    Safest multi-prompt path: each file is read whole (paragraphs and
    ``[Shot N]`` markers preserved) and passed as a single ``generate_clip``
    prompt — no shell ``$()`` line-splitting, no fragment clips.

    ``paths`` is a comma-separated list of prompt files, e.g.::

        modal run api.py::t2v_files --paths "Vignettes/sweeper.txt,Vignettes/clock.txt" \\
            --duration 10 --width 864 --height 480 --mode turbo

    Duration defaults to 10s (frame length snaps to 17n+5 -> 243 frames).
    Each take uses a fresh random seed.
    """
    path_list = [p.strip() for p in paths.split(",") if p.strip()]
    if not path_list:
        raise ValueError("t2v_files requires at least one --paths file")
    prompts = []
    for path in path_list:
        with open(path, encoding="utf-8") as f:
            prompts.append(f.read())

    gen = H3Generator()
    results = []
    for p in prompts:
        seed = random_seed()
        results.append(gen.generate_clip.remote(
            prompt=p, ref_names=None, mode=mode, steps=steps, seed=seed,
            ref_image_size=ref_image_size, lora_strength=lora_strength,
            task="t2v", duration=duration, width=width, height=height))

    eff_steps = steps if steps is not None else (4 if mode == "turbo" else 20)
    print(f"\nT2V files complete ({len(results)} take(s), mode={mode}, "
          f"steps={eff_steps}, duration={duration}, {width}x{height})")
    for i, r in enumerate(results):
        print(f"  #{i}  seed={r['seed']:<11} {r['s3_key']}")
    print("\nS3 keys:")
    for r in results:
        print(f"  {r['s3_key']}")


@app.local_entrypoint()
def r2v_multi(scenes: str, seed: int | None = None):
    """Render one R2V take per scene in a scenes JSON file on ONE warm container.

    Unlike ``batch``, this loads the H3Generator once and loops all scenes, so
    a back-to-back multi-scene list shares the model and skips re-boot after
    the first scene. Note: with ```modal run api.py::r2v_multi```, scenes all
    run sequentially over the same L40S (each scene is ~7 min at 864x480/match).

    ``scenes`` is a path to a JSON manifest:
        {
          "alley": {"prompt_file": "Drunken Master/alley.txt",
                    "ref_names": ["drunken_master.png", "alleyway_henchman.png"],
                    "duration": 10.0, "width": 864, "height": 480,
                    "ref_image_size": "match", "ref_subdir": "drunken_master"},
          "kitchen": {...}
        }
    Each scene uses a fresh random seed (or base --seed for all).
    Prints per-scene S3 keys.
    """
    import json

    with open(scenes, encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"scenes manifest must be non-empty dict, got {manifest!r}")

    gen = H3Generator()
    base = seed if seed is not None else random_seed()
    results = []
    for scene_id, scene in manifest.items():
        results.append((scene_id, _run_r2v(gen, scene, seed=base + len(results))))

    print(f"\nR2V multi complete ({len(results)} scene(s), base_seed={base})")
    for i, (scene_id, r) in enumerate(results):
        print(f"  #{i}  {scene_id:<12} seed={r['seed']:<11} {r['s3_key']}")
    print("\nS3 keys:")
    for _s_id, r in results:
        print(f"  {r['s3_key']}")


def random_seed() -> int:
    return random.randint(0, 2_147_483_647)