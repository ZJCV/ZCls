<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.en.md">🇺🇸</a>
  <!-- <a title="俄语" href="../ru/README.md">🇷🇺</a> -->
</div>

 <div align="center"><a title="" href="git@github.com:ZJCV/PyCls.git"><img align="center" src="./imgs/PyCls.png"></a></div>

<p align="center">
  «PyCls»是一个分类模型基准代码库
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

使用相同的参数和配置进行基准测试：

* 硬件：`Nvidia RTX 2080Ti`
* 数据：
  * 数据集：`CIFAR100`
  * 预处理：缩放 + 中心采样 + 归一化
  * 批量大小：`32`
* 训练：
  * 损失函数：`CrossEntropyLoss`
  * 优化器：`SGD`
  * 迭代次数：`100`轮
  * 学习率调度器：`MultiStepLR`，第`50/80`轮进行一次调度，学习率因子为`0.1`
  * 初始学习率：`1e-3`
  * 权重衰减因子：`1e-5`
* 测试：
  * 评判标准：`Top-1/Top-5`准确率


## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [主要维护人员](#主要维护人员)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

为了进一步提高分类性能，通常需要对已有模型进行改进，不可避免的就涉及到代码重构。创建本仓库，一方面作为分类模型的`CodeBase`，另一方面也记录下自定义模型与成熟的开源实现之间的比较

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjykzj/PyCls/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2020 zjykzj