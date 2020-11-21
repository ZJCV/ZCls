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

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.LOG_STEP = 10
_C.TRAIN.SAVE_EPOCH = 10
_C.TRAIN.EVAL_EPOCH = 10
_C.TRAIN.MAX_EPOCH = 100
_C.TRAIN.RESUME = False
_C.TRAIN.USE_TENSORBOARD = True

# ---------------------------------------------------------------------------- #
# DataSets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.NAME = 'CIFAR100'
_C.DATASETS.DATA_DIR = './data/cifar'

# ---------------------------------------------------------------------------- #
# Transform
# ---------------------------------------------------------------------------- #
_C.TRANSFORM = CN()
_C.TRANSFORM.MEAN = (0.5071, 0.4865, 0.4409)
_C.TRANSFORM.STD = (0.1942, 0.1918, 0.1958)

_C.TRANSFORM.TRAIN = CN()
_C.TRANSFORM.TRAIN.SHORTER_SIDE = 224
_C.TRANSFORM.TRAIN.CENTER_CROP = True
_C.TRANSFORM.TRAIN.TRAIN_CROP_SIZE = 224

_C.TRANSFORM.TEST = CN()
_C.TRANSFORM.TEST.SHORTER_SIDE = 224
_C.TRANSFORM.TEST.CENTER_CROP = True
_C.TRANSFORM.TEST.TEST_CROP_SIZE = 224

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 16
_C.DATALOADER.TEST_BATCH_SIZE = 16
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = 'ResNet'
_C.MODEL.PRETRAINED = ""
_C.MODEL.SYNC_BN = False

_C.MODEL.BACKBONE = CN()
# for ResNet
_C.MODEL.BACKBONE.ARCH = 50

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.FEATURE_DIMS = 2048
_C.MODEL.HEAD.NUM_CLASSES = 1000

_C.MODEL.RECOGNIZER = CN()
_C.MODEL.RECOGNIZER.NAME = 'R50_Pytorch'

_C.MODEL.CRITERION = CN()
_C.MODEL.CRITERION.NAME = 'CrossEntropyLoss'

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'SGD'
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 3e-5
# for sgd
_C.OPTIMIZER.SGD = CN()
_C.OPTIMIZER.SGD.MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# LR_Scheduler
# ---------------------------------------------------------------------------- #
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.NAME = 'MultiStepLR'
_C.LR_SCHEDULER.IS_WARMUP = False
_C.LR_SCHEDULER.GAMMA = 0.1

# for MultiStepLR
_C.LR_SCHEDULER.MULTISTEP_LR = CN()
_C.LR_SCHEDULER.MULTISTEP_LR.MILESTONES = [50, 80]
# for Warmup
_C.LR_SCHEDULER.WARMUP = CN()
_C.LR_SCHEDULER.WARMUP.ITERATION = 5
_C.LR_SCHEDULER.WARMUP.MULTIPLIER = 1.0
