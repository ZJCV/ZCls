
# trick-train

In addition to normal training, zcls provides three additional training tricks

1. shuffle data per round
2. hybrid percision training
3. gradient accumulate

## Shuffle data per round

If you choose to shuffle data in training, then config like this:

```
DATALOADER:
  ...
  ...
  SHUFFLE: True
```

The data will be shuffled per round

## Hybrid percision training

Pytorch has been supporting mixed precision training since version 1.6. zcls integrates this function, just config like this:

```
TRAIN:
  ...
  ...
  HYBRID_PRECISION: True
```

## Gradient accumulate

Accumulating the gradient of multiple rounds of calculation can simulate the parallel effect of multiple cards on a single card, so the loss can converge more quickly

open config item like this:

```
TRAIN:
  ...
  ...
  GRADIENT_ACCUMULATE_STEP: 1
```

Set `GRADIENT_ACCUMULATE_STEP=1` means normal training, each round is updated with a gradient

## Related references

* [Introducing native PyTorch automatic mixed precision for faster training on NVIDIA GPUs](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
* [How to Break GPU Memory Boundaries Even with Large Batch Sizes](https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce)