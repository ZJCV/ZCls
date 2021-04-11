
# Use Custom Datasets

zcls provides two ways to use custom datasets

## Commonly used

Suppose your dataset is in the following format

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

modify config_file like this

```
DATASET:
  NAME: 'GeneralDataset'
  TRAIN_ROOT: /path/to/train_root
  TEST_ROOT: /path/to/test_root
  TOP_K: (1, 5)
```

## Optimized used

zcls supports [LMDB](https://lmdb.readthedocs.io/en/release/) to speed up data set reading

### How to create LMDBDataset

See [pnno](https://github.com/zjykzj/pnno)

*tip: before use `pnno` to create LMDB file, you can use `tools/zoom.py` to resize image*

### How to Use

modify config_file like this

```
DATASET:
  NAME: 'LMDBDataset'
  TRAIN_ROOT: '/path/to/train_lmdb_path'
  TEST_ROOT: '/path/to/test_lmdb_path'
  TOP_K: (1, 5)
```

**Note: `TRAIN_ROOT` and `TEST_ROOT` should fill in the corresponding path**

### For ImageNet

zcls specifically implements the use of Imagenet. Modify config_file like this

```
DATASET:
  NAME: 'LMDBImageNet'
  TRAIN_ROOT: '/path/to/train_lmdb_path'
  TEST_ROOT: '/path/to/test_lmdb_path'
  TOP_K: (1, 5)
```

## Related references

* [ImageFolder](https://pytorch.org/vision/stable/datasets.html#imagefolder)
* [What’s the best way to load large data?](https://discuss.pytorch.org/t/whats-the-best-way-to-load-large-data/2977)