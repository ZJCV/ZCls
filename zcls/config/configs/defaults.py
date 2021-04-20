from yacs.config import CfgNode as CN

_C = CN()

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# ---------------------------------------------------------------------------- #
# Distributed options
# ---------------------------------------------------------------------------- #
# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_NODES = 1

# The index of the current machine.
_C.RANK_ID = 0

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# Initialization method, includes TCP or shared file-system
_C.INIT_METHOD = "tcp://localhost:39129"

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.LOG_STEP = 10
# refert to
# [How to Break GPU Memory Boundaries Even with Large Batch Sizes](https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce)
# [How to implement accumulated gradient？](https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient/3822)
_C.TRAIN.GRADIENT_ACCUMULATE_STEP = 1
_C.TRAIN.SAVE_EPOCH = 5
_C.TRAIN.EVAL_EPOCH = 5
_C.TRAIN.MAX_EPOCH = 200
_C.TRAIN.RESUME = False
_C.TRAIN.USE_TENSORBOARD = True
# Hybrid precision training
# refer to [pytorch-distributed](https://github.com/zjykzj/pytorch-distributed)
_C.TRAIN.HYBRID_PRECISION = False