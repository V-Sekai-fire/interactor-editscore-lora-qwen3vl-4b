# interactor-editscore-lora-qwen3vl-4b

The image-edit reward model for the avatar pipeline: a published edit-scoring adapter on a small vision-language model.

## What it is for

It scores an image edit against its instruction, so the mask-scoring loop of RFD 1173 can reward the generators. The repository holds an inference smoke test and the scaffold for training task-specific adapters. Trained weights go to a model hub and are never committed here.

## Build and run

```sh
pixi run -e mlx smoke-editscore
pixi run -e lora smoke
```

The `mlx` environment, and so the inference smoke test, runs on Apple silicon only.

## Licence

MIT. See [LICENSE](LICENSE). Base model and adapter weights keep their upstream licences.
