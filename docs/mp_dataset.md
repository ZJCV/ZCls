
# MPDataset

ZCls provides a new iterable-style dataset class for large-scale data loading. 

## File structure

It is assumed that the folder structure is as follows

```angular2html
data/
└── dataset_name
    ├── test
    │   ├── cls.csv
    │   ├── data.csv
    └── train
        ├── cls.csv
        └── data.csv
```

* `data.csv` means stores the image path

```
/path/to/img1.jpg,,target1
/path/to/img2.jpg,,target2
/path/to/img3.jpg,,target3
...
...
```

* `cls.csv` stores the classes

```angular2html
cls_name1
cls_name2
cls_name3
...
...
```

*Note: `,,` means separator*

## Usage

modify config_file like this

```
DATASET:
  NAME: 'MPDataset'
  TRAIN_ROOT: './data/dataset_name/train/'
  TEST_ROOT: './data/dataset_name/test/'
  TOP_K: (1, 5)
```

## Related references

* [ zjykzj/MPDataset](https://github.com/zjykzj/MPDataset)