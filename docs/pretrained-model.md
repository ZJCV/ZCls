
# Pretrained Model

ZCls provides many pretrained model file in remote, and you can also use pths in local. There are three config items to load pretraining model:

1. `cfg.MODEL.RECOGNIZER.PRETRAINED_REMOTE`
2. `cfg.MODEL.RECOGNIZER.PRETRAINED_LOCAL`
3. `cfg.MODEL.RECOGNIZER.PRELOADED`

## PRETRAINED_REMOTE

Open the `PRETRAINED_REMOTE` item in config file, if zcls provides a pretrained model in remote, then will download and load it automatically

```
MODEL:
  ...
  ...
  RECOGNIZER:
    PRETRAINED_REMOTE: True
    PRETRAINED_NUM_CLASSES: 1000
```

Set the number of output categories of the pre-training model correctly. If the current number of training categories is inconsistent with the pre-training categories, the pre-training parameters of the last classification layer will not be loaded. 

## PRETRAINED_LOCAL

Configure the local pretraining model path in the configuration file like this:

```
MODEL:
  ...
  ...
  RECOGNIZER:
    PRETRAINED_LOCAL: "/path/to/pretrained"
    PRETRAINED_NUM_CLASSES: 1000
```

If both `PRETRAINED_LOCAL` and `PRETRAINED_REMOTE` are set, then `PRETRAINED_LOCAL` takes precedence. 

## PRELOADED

If the pre-training model contains the training-time module (use `cfg.MODEL.CONV.ADD_BLOCKS`) provided by zcls, the path of the pre-training model can be set in `PRELOADED`. 

```
MODEL:
  ...
  ...
  RECOGNIZER:
    PRELOADED: "/path/to/pretrained"
```