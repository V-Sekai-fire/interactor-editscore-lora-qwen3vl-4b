# editscore-lora-qwen3vl-4b

EditScore reward-model role for RFD 1173's MaskScore loop, on
Qwen3-VL-4B. Two deployment paths in the same repo:

- **v0.1 (Mac mini)**: `mlx-community/Qwen3-VL-4B-Instruct-4bit`
  (3.1 GB) + `EditScore/EditScore-Qwen3-VL-4B-Instruct` peft LoRA
  (270 MB), served via `mlx-vlm`. Verified 2026-09-01: 0.9 s cold
  load, 1.9 s first token on Mac mini M2 Pro 32 GB. `scripts/smoke_editscore_mlx.py`.
- **v0.2 (RTX 3090)**: QAT-LoRA training on Qwen3-VL-4B against
  `EditScore/EditScore-Reward-Data` (97,300 rows, apache-2.0) to
  produce weftspun-specific adapters for the task types in
  MASKSCORE.md. `scripts/smoke.py` scaffolds the training loop.

Trained adapters get pushed to Hugging Face as
`chibifire/editscore-lora-qwen3vl-4b-<task-type>` per the
weights-live-on-huggingface rule. Model weights never live in a
GitHub repo.

## Base

`Qwen/Qwen3-VL-4B-Instruct` — apache-2.0, 8.9 GB fp16,
`pipeline_tag: image-text-to-text`. Alibaba's own release; MLX 4-bit
port at `mlx-community/Qwen3-VL-4B-Instruct-4bit` for Apple Silicon.

## Existing EditScore adapters (upstream, apache-2.0)

| repo | size | base |
| --- | --- | --- |
| `EditScore/EditScore-Qwen3-VL-4B-Instruct` | 270 MB LoRA (r=32) | Qwen3-VL-4B-Instruct |
| `EditScore/EditScore-Qwen3-VL-8B-Instruct` | 400 MB LoRA | Qwen3-VL-8B |
| `EditScore/EditScore-Qwen3-VL-32B-Instruct` | LoRA | Qwen3-VL-32B |
| `EditScore/EditScore-7B` | LoRA (license=None, rejected) | Qwen2.5-VL-7B |
| `EditScore/EditScore-32B` | LoRA | Qwen2.5-VL-32B |
| `EditScore/EditScore-72B` | LoRA | Qwen2.5-VL-72B |

v0.1 uses the 4B adapter without merging (reads the model's
instruction-follow answer as the reward). v0.2 merges the peft
adapter into an fp16 copy of the base and re-quantizes to MLX or
serves via vLLM on the 3090 in CT-w4a16 format.

## Track B (Hailo vision encoder) — parked

`scripts/gate_vision_encoder.py` exports Gemma-4-E4B's vision tower
to ONNX and diffs against a Hailo-10H op whitelist (39 ops in graph,
25 in, 14 out). The DFC-side gate (`scripts/gate_dfc_parse.py`)
needs a linux_x86_64 host with Hailo Dataflow Compiler 5.3.0
installed — see `scripts/install_dfc.md`. Whole track waits on
Linux host + Hailo hardware.

The `gate_vision_encoder.py` script targets Gemma-4-E4B because that
was the base at Track B's design time; retargeting to Qwen3-VL-4B is
straightforward once Hailo hardware is real (same ONNX export shape,
different vision-tower attribute name).

## Environment

`pixi.toml` declares four features: `lora` (training on 3090),
`gate` (ONNX-export any-desk), `dfc` (linux-64 only, DFC compile),
`mlx` (osx-arm64, EditScore inference on Mac mini). No `uv`.

## Run

    pixi install -e mlx                          # osx-arm64
    pixi run -e mlx smoke-editscore              # 0.9s load, 1.9s first token
    pixi install -e gate
    pixi run -e gate gate-vision-encoder         # exports ONNX + Hailo op-diff
    pixi install -e lora                         # osx-arm64 or linux-64
    pixi run -e lora smoke                       # small LoRA training smoke

## Licence

MIT for this project's code. Base + adapter weights inherit their
upstream licences (apache-2.0 for Qwen3-VL and the EditScore adapter).
