
# GeneralDataset

## File structure

Suppose your dataset is in the following format

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

## Usage

modify config_file like this

```
DATASET:
  NAME: 'GeneralDataset'
  TRAIN_ROOT: /path/to/train_root
  TEST_ROOT: /path/to/test_root
  TOP_K: (1, 5)
```

## Related references

* [ImageFolder](https://pytorch.org/vision/stable/datasets.html#imagefolder)