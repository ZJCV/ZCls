
## LMDBDataset

zcls supports [LMDB](https://lmdb.readthedocs.io/en/release/) to speed up data set reading

## How to create LMDBDataset

See [pnno](https://github.com/zjykzj/pnno)

*tip: before use `pnno` to create LMDB file, you can use `tools/zoom.py` to resize image*

## How to Use

modify config_file like this

```
DATASET:
  NAME: 'LMDBDataset'
  TRAIN_ROOT: '/path/to/train_lmdb_path'
  TEST_ROOT: '/path/to/test_lmdb_path'
  TOP_K: (1, 5)
```

**Note: `TRAIN_ROOT` and `TEST_ROOT` should fill in the corresponding path**

## For ImageNet

zcls specifically implements the use of Imagenet. Modify config_file like this

```
DATASET:
  NAME: 'LMDBImageNet'
  TRAIN_ROOT: '/path/to/train_lmdb_path'
  TEST_ROOT: '/path/to/test_lmdb_path'
  TOP_K: (1, 5)
```

## Related references

* [What’s the best way to load large data?](https://discuss.pytorch.org/t/whats-the-best-way-to-load-large-data/2977)