
# trick-data

zcls provides three ways to speed up data processing time

1. lmdb
2. prefetcher
3. zoom

## lmdb

Use [lmdb](https://github.com/jnwatson/py-lmdb/) can speed up the read from disk to memory

* First, use [pnno](https://github.com/zjykzj/pnno) to create `.lmdb` file from dataset 
* Second, see [How to use LMDBDataset](./custom-datasets.md)

## prefetcher

Using prefetcher can speed up the reading from CPU memory to GPU memory

just open config item like this:

```
DATALOADER:
  ...
  ...
  ...
  PREFETCHER: True
```

## zoom

Before training, the data can be prescaled. zcls provides a script `tools/zoom.py` to realize it

## Related references

* [What’s the best way to load large data?](https://discuss.pytorch.org/t/whats-the-best-way-to-load-large-data/2977)
* [TypeError: can't pickle Environment objects when num_workers > 0 for LSUN #689](https://github.com/pytorch/vision/issues/689)
* [Dose data_prefetcher() really speed up training? #304](https://github.com/NVIDIA/apex/issues/304)