#!/usr/bin/env python3
"""Vision-encoder gate: does Gemma-4-E4B's image tower stay inside the
Hailo operator set? Runs on any desk, no DFC required.

Same shape as rf-detr-cpp/scripts/gate_onnx_device.py:
  1. Load Gemma-4-E4B from HF cache
  2. Extract the vision encoder module only
  3. Export it to ONNX with a synthetic image batch
  4. Parse the ONNX operator set
  5. Diff against Hailo-10H's accepted op set (from hailo_ops.usda / DEVICE_OPS)
  6. Print the diff; nonzero exit iff any op falls outside

The DFC-side gate (`gate_dfc_parse.py`, linux-64 only) runs the real
compiler on the same ONNX. This gate and that one's disagreement is the
actionable finding.
"""
import argparse, sys
from pathlib import Path

# Hailo-10H accepted op set (subset, checked against hailo_model_zoo's
# HAILO_MODELS.rst and the DFC 5.3.0 op catalog). Grows as Hailo adds
# op support; kept as a checked-in list rather than fetched at runtime so
# the gate is offline-reproducible.
HAILO_10H_OPS = {
    "Add", "Sub", "Mul", "Div", "MatMul", "Gemm",
    "Conv", "ConvTranspose", "AveragePool", "MaxPool", "GlobalAveragePool",
    "Relu", "Sigmoid", "Tanh", "Gelu", "Softmax", "LayerNormalization",
    "Reshape", "Transpose", "Concat", "Split", "Slice", "Gather",
    "Cast", "Constant", "Identity", "Shape", "Squeeze", "Unsqueeze",
    "Erf", "Sqrt", "Pow", "Neg", "Where", "Equal",
    "ReduceMean", "ReduceSum", "ReduceMax",
    # ViT-shape essentials
    "Einsum",  # attention; DFC 5.3 supports some Einsum patterns
    "Expand", "Range", "Pad",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it-qat-q4_0-unquantized")
    ap.add_argument("--onnx", default="build/vision_encoder.onnx")
    args = ap.parse_args()

    print(f"loading {args.model}, extracting vision encoder...", flush=True)
    try:
        from transformers import AutoModel
        import torch
    except ImportError as e:
        print(f"gate needs pixi env `gate`: pixi shell -e gate. missing: {e}")
        sys.exit(2)

    # CPU-only on purpose (see pixi.toml gate feature). No device_map so we
    # don't drag in accelerate; no torch_dtype (deprecated) — modern arg is `dtype`.
    m = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    # Gemma-4 multimodal architecture: locate the vision tower
    vision = None
    for name in ("vision_tower", "vision_model", "image_encoder", "visual"):
        vision = getattr(m, name, None)
        if vision is not None:
            print(f"  found vision module: {name}"); break
    if vision is None:
        # Look under a .model attribute (transformers wrapping convention)
        inner = getattr(m, "model", m)
        for name in ("vision_tower", "vision_model", "image_encoder", "visual"):
            vision = getattr(inner, name, None)
            if vision is not None:
                print(f"  found vision module: model.{name}"); break
    if vision is None:
        print("could not locate vision tower on the model; add its attribute name")
        sys.exit(2)

    Path(args.onnx).parent.mkdir(parents=True, exist_ok=True)
    # Discovered from Gemma4Processor on a 224x224 image:
    #   pixel_values (1, 2520, 768) f32       — batch, num_patches, C*P*P (3*16*16)
    #   image_position_ids (1, 2520, 2) i64   — 2D (row, col) per patch
    # The forward signature names the second arg pixel_position_ids; the
    # processor emits image_position_ids. Same tensor, different name.
    n_patches = 2520
    hidden_per_patch = 3 * 16 * 16
    pixel_values = torch.randn(1, n_patches, hidden_per_patch)
    pixel_position_ids = torch.zeros(1, n_patches, 2, dtype=torch.long)
    vision.eval()
    torch.onnx.export(
        vision, (pixel_values, pixel_position_ids), args.onnx, opset_version=17,
        input_names=["pixel_values", "pixel_position_ids"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch", 1: "num_patches"},
            "pixel_position_ids": {0: "batch", 1: "num_patches"},
            "image_features": {0: "batch"},
        },
    )
    print(f"exported {args.onnx}")

    import onnx
    model = onnx.load(args.onnx)
    ops_in_graph = {node.op_type for node in model.graph.node}
    outside = ops_in_graph - HAILO_10H_OPS
    inside  = ops_in_graph & HAILO_10H_OPS
    print(f"\nops in graph: {len(ops_in_graph)}")
    print(f"  inside Hailo-10H set:  {len(inside)}")
    print(f"  OUTSIDE Hailo-10H set: {len(outside)}")
    if outside:
        print("\noffending ops (fail the gate):")
        for op in sorted(outside):
            print(f"  {op}")
        sys.exit(1)
    print("\ngate PASS: all ops in Hailo-10H set")

if __name__ == "__main__":
    main()
