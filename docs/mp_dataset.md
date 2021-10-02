
# MPDataset

ZCls provides a new iterable-style dataset class for large-scale data loading. 

## File structure

Suppose your dataset save in json file with following format

```
{
    'imgs': [img_path1, img_path2, ...],
    'targets': [label1, label2, ...],
    'classes': [class1, class2, ...]
}
```

About how to create json file, you can see [get_cifar_data.py](https://github.com/zjykzj/MPDataset/blob/master/tools/data/get_cifar_data.py)

## Usage

modify config_file like this

```
DATASET:
  NAME: 'MPDataset'
  TRAIN_ROOT: './data/cifar/cifar100_train.json'
  TEST_ROOT: './data/cifar/cifar100_test.json'
  TOP_K: (1, 5)
```

## Related references

* [ zjykzj/MPDataset](https://github.com/zjykzj/MPDataset)