<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/ZCls"><img align="center" src="./imgs/ZCls.png"></a></div>

<p align="center">
  «ZCls»是一个分类模型训练/推理框架
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
  <a href="https://pypi.org/project/zcls/"><img src="https://img.shields.io/badge/PYPI-zcls-brightgreen"></a>
  <a href='https://zcls.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/zcls/badge/?version=latest' alt='Documentation Status' />
  </a>
</p>

当前已实现：

<p align="center">
<img align="center" src="./imgs/roadmap.svg">
</p>

*更多细节请参考[路线图](https://zcls.readthedocs.io/en/latest/roadmap.html)*

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

在目标检测/目标分割/动作识别领域，已经出现了许多集成度高、流程完善的训练框架，比如[facebookresearch/detectron2](https://github.com/facebookresearch/detectron2), [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2)等等。

目标分类是深度学习中最早发展的、理论最基础的领域。参考现有的训练框架，实现一个基于目标分类模型的训练/推理框架。希望`ZCls`能给你带来更好的实现。

## 安装

查看[INSTALL](https://zcls.readthedocs.io/en/latest/install.html)

## 用法

关于如何训练，查看[Get Started with ZCls](https://zcls.readthedocs.io/en/latest/get-started.html)

关于如何使用内置数据集，查看[Use Builtin Datasets](https://zcls.readthedocs.io/en/latest/builtin-datasets.html)

关于如何使用自定义数据集，查看[Use Custom Datasets](https://zcls.readthedocs.io/en/latest/custom-datasets.html)

使用预训练模型，查看 [Use Pretrained Model](https://zcls.readthedocs.io/en/latest/pretrained-model.html)

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

```
@misc{ding2021diverse,
      title={Diverse Branch Block: Building a Convolution as an Inception-like Unit}, 
      author={Xiaohan Ding and Xiangyu Zhang and Jungong Han and Guiguang Ding},
      year={2021},
      eprint={2103.13425},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{ding2021repvgg,
      title={RepVGG: Making VGG-style ConvNets Great Again}, 
      author={Xiaohan Ding and Xiangyu Zhang and Ningning Ma and Jungong Han and Guiguang Ding and Jian Sun},
      year={2021},
      eprint={2101.03697},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}

@misc{zhang2020resnest,
      title={ResNeSt: Split-Attention Networks}, 
      author={Hang Zhang and Chongruo Wu and Zhongyue Zhang and Yi Zhu and Haibin Lin and Zhi Zhang and Yue Sun and Tong He and Jonas Mueller and R. Manmatha and Mu Li and Alexander Smola},
      year={2020},
      eprint={2004.08955},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{han2020ghostnet,
      title={GhostNet: More Features from Cheap Operations}, 
      author={Kai Han and Yunhe Wang and Qi Tian and Jianyuan Guo and Chunjing Xu and Chang Xu},
      year={2020},
      eprint={1911.11907},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

*更多致谢内容，查看[THANKS](./THANKS)*

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/ZJCV/ZCls/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2020 zjykzj