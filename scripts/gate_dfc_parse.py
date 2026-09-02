#!/usr/bin/env python3
"""DFC-side gate: run the REAL Hailo Dataflow Compiler parser on the ONNX
`gate_vision_encoder.py` exported. Linux x86_64 only — DFC ships as a
linux_x86_64 wheel. See `scripts/install_dfc.md`.

The pair (this + `gate_vision_encoder.py`) either agree or disagree:
  - both PASS: encoder is compilable
  - both FAIL: encoder needs op-set surgery
  - gate PASS + this FAIL: our op-set whitelist is out of date, DFC caught
    something the whitelist missed. Update the whitelist.
  - gate FAIL + this PASS: our op-set whitelist is too strict, DFC accepts
    ops we thought it did not. Relax the whitelist.

DISAGREEMENT is the actionable finding, per rf-detr-cpp's pattern.
"""
import sys
from pathlib import Path

def main():
    onnx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "build/vision_encoder.onnx")
    if not onnx_path.exists():
        print(f"missing: {onnx_path} — run `pixi run -e gate gate-vision-encoder` first")
        sys.exit(2)

    try:
        from hailo_sdk_client import ClientRunner
    except ImportError:
        print("hailo_sdk_client not installed; see scripts/install_dfc.md")
        print("this task runs on linux_x86_64 only")
        sys.exit(2)

    runner = ClientRunner(hw_arch="hailo10h")
    print(f"parsing {onnx_path} against hailo10h...", flush=True)
    hn, npz = runner.translate_onnx_model(str(onnx_path), "vision_encoder")
    print(f"DFC parse PASS: {len(hn['layers'])} layers, {sum(1 for l in hn['layers'] if l.get('type')=='input_layer')} inputs")
    # Save HAR for the QFT step to consume
    har = onnx_path.with_suffix(".har")
    runner.save_har(str(har))
    print(f"saved {har}")

if __name__ == "__main__":
    main()
