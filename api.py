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
                   mode: str = "turbo", steps: int | None = None,
ref_image_size: str = "max",
                    lora_strength: float = 1.0) -> dict:
    """Generate one take of a scene on the cheap ComfyUI path; upload to S3.

    ``gen`` is passed in (rather than created here) so a batch loop reuses one
    warm container and re-reads the prompt only once.
    """
    scene = SCENES[scene_id]
    with open(scene["prompt_file"], encoding="utf-8") as f:
        prompt = f.read()
    return gen.generate_clip.remote(
        prompt=prompt,
        ref_names=scene["ref_names"],
        duration=scene["duration"],
        width=scene["width"],
        height=scene["height"],
        mode=mode, steps=steps, seed=seed,
        ref_image_size=ref_image_size, lora_strength=lora_strength,
    )


@app.local_entrypoint()
def batch(scene: str = "1", variants: int = 1, mode: str = "turbo",
          steps: int | None = None, seed: int | None = None,
          ref_image_size: str = "max", lora_strength: float = 1.0,
          task: str = "r2v", prompt: str | None = None,
          duration: float = 12.25, width: int = 960, height: int = 544):
    """Generate ``variants`` takes of a scene on one warm L40S container.

    steps: omitted -> 20 (R2V always full) / 4 (turbo) or 20 (full) for T2V.
    seed:  None -> random base; variants use base .. base+N-1 (all distinct).
    task:  "r2v" (default, uses SCENES refs; full-quality only) or "t2v"
           (pure text, needs --prompt; turbo supported here).
    duration/width/height: override canvas+sine for t2v (e.g. 5s, 864x480 0.4MP).
    Prints a per-variant summary plus the bare S3 keys after everything lands.
    """
    if variants < 1:
        raise ValueError(f"variants must be >= 1, got {variants}")
    if task == "t2v" and not prompt:
        raise ValueError("task=t2v requires --prompt <text>")
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
        else:
            results.append(generate_scene(
                gen, scene, seed=base + i, mode=mode, steps=steps,
                ref_image_size=ref_image_size, lora_strength=lora_strength))

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
    back-to-back list shares the model and skips re-boot after the first
    prompt. Pass prompts as newline-separated lines (e.g. ``--prompts $'...\n...'``).
    Each take uses a fresh random seed.
    """
    prompts_list = [p.strip() for p in prompts.splitlines() if p.strip()]
    if not prompts_list:
        raise ValueError("t2v_multi requires at least one non-empty --prompts line")
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


def random_seed() -> int:
    return random.randint(0, 2_147_483_647)