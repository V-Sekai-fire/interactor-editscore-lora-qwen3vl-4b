# Installing the Hailo Dataflow Compiler

`pixi install` does NOT fetch DFC — the wheel is proprietary and gated behind
Hailo developer registration. Install it once, manually, on a linux_x86_64
host (or in a Rosetta/Docker linux-64 container).

## Prereqs

- Linux x86_64 (Ubuntu 22.04 / 24.04 tested by Hailo)
- Python 3.10, 3.11, or 3.12
- The wheel: `hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl`
  present at `~/Applications/`, `~/Documents/asus-hailo/`, and
  `~/Documents/FireFiles/Applications/` on the operator's Mac (mirrored for
  reference — install actually runs on the Linux workstation)

## Install into the pixi `dfc` env

    # From this project root, on a linux-64 host:
    pixi shell -e dfc
    pip install /path/to/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl
    hailo --version    # should print 5.3.0

Do NOT install via `uv`. The `no-uv` gate exists because Hailo's
twenty-ModuleNotFoundError install saga is what got uv blocklisted; pixi
holds the manifest of every transitive dep that install pulls in.

## What the DFC gives us

- `hailo` CLI — top-level workflow (parse, optimize, compile, HAR/HEF)
- `hailo_sdk_client` — Python bindings for the same
- `hailo_model_optimization` — QFT (Quantization Fine-Tuning), Hailo's QAT-equivalent
- `hailo_tools`, `hailo_sdk_common`, `hailo_sdk_server` — supporting libs

## Tasks that need it

    pixi run -e dfc dfc-parse    # parse the ONNX the gate exported
    pixi run -e dfc qft          # QFT training pass on the vision encoder

## Osx-arm64 workaround

There isn't a clean one. Options:

1. Run on the linux workstation (`~/Documents/asus-hailo/`)
2. Rent a Vast linux_x86_64 CPU instance (~$0.02-0.05/hr, ~1 GB image pull)
3. Podman/Docker with `--platform linux/amd64` via Rosetta on Apple Silicon
   — slow, works for DFC parsing but not for hardware access
