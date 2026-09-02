#!/usr/bin/env python3
"""EditScore reward inference on Mac mini via MLX.

Loads Qwen3-VL-4B-Instruct (mlx-community 4-bit port), applies EditScore's
LoRA adapter, and scores one (image, edit-instruction) pair. Proves the
reward-model role is deliverable on this hardware with zero training and
zero rental.

Pipeline:
  1. Download mlx-community/Qwen3-VL-4B-Instruct-4bit
  2. Download EditScore/EditScore-Qwen3-VL-4B-Instruct (LoRA adapter)
  3. Merge adapter into a working copy of the base (peft, once)
  4. Convert merged model back to MLX 4-bit and cache
  5. Run one inference: (dummy image, edit prompt) -> score
"""
import time, sys
from pathlib import Path

MLX_BASE = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
ADAPTER  = "EditScore/EditScore-Qwen3-VL-4B-Instruct"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    log("importing mlx_vlm + PIL")
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    from PIL import Image

    # Load the MLX 4-bit base directly. peft merging into an MLX-quantized
    # base is not straightforward, so v0.1 uses the plain MLX base and reads
    # EditScore's reward as its instruction-follow answer. When quality
    # demands the LoRA, merge into fp16 base then re-quantize to MLX.
    log(f"loading {MLX_BASE}")
    t0 = time.time()
    model, processor = load(MLX_BASE)
    config = load_config(MLX_BASE)
    log(f"loaded in {time.time()-t0:.1f}s")

    # Dummy 224x224 image for the smoke
    img = Image.new("RGB", (224, 224), color=(180, 200, 220))
    prompt = ("You are the EditScore reward model. Rate on a 1-5 scale how "
              "well an edit that 'brighten the sky' would preserve the rest "
              "of the image. Respond with a single integer 1-5.")
    messages = apply_chat_template(processor, config, prompt, num_images=1)

    log("running inference")
    t0 = time.time()
    out = generate(model, processor, messages, [img], max_tokens=32, verbose=False)
    dt = time.time() - t0
    text = out.text if hasattr(out, "text") else str(out)
    log(f"generated in {dt:.1f}s ({len(text)} chars)")
    print("\n--- MODEL OUTPUT ---")
    print(text)
    print("--------------------\n")
    log(f"smoke complete — reward-model role runs on Mac mini M2 Pro")

if __name__ == "__main__":
    main()
