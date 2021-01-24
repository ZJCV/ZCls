
# 基准测试

## 轻量级模型

当前已实现的轻量级模型有：`MobileNetV1/V2/V3、MNasNet、ShuffleNetV1/V2`

相同训练配置如下：

1. 训练次数：`100`轮
2. 数据集：`CIFAR100`
3. 图像预处理：
   1. 较短边扩展到`224`，然后中心裁剪`224x224`
   2. 图像均值：`(0.45, 0.45, 0.45)`
   3. 标准差：`(0.225, 0.225, 0.225)`
4. 批量大小：`32`
5. 损失函数：`CrossEntropyLoss`
6. 学习率/权重衰减：
   1. 针对`RMSProp`：`3e-4/1e-5`
   2. 针对`SGD`：`0.0125(0.9 momentum)/1e-5`
7. 学习率调度：`WarmUP`（前`5`轮） + `MultiStepLR`（第`40/70`轮衰减一次，衰减因子为`0.1`）
8. 训练加速：混合精度训练
9.  `GPU/CPU：GeForce RTX 2080 Ti + Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz`

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">model</th>
    <th class="tg-7btt">custom</th>
    <th class="tg-7btt">optimizer</th>
    <th class="tg-7btt">top1 acc</th>
    <th class="tg-7btt">top5 acc</th>
    <th class="tg-7btt">model size(MB)</th>
    <th class="tg-7btt">gflops</th>
    <th class="tg-7btt">gpu infer(s)</th>
    <th class="tg-7btt">cpu infer(s)</th>
    <th class="tg-7btt">config/log</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">mbv1_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-0pky">66.474</td>
    <td class="tg-0pky">88.309</td>
    <td class="tg-0pky">12.625</td>
    <td class="tg-0pky">1.156</td>
    <td class="tg-fymr">*0.006</td>
    <td class="tg-fymr">*0.030</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/eJ5soxokHqDY9ni" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mbv2_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-0pky">69.069</td>
    <td class="tg-0pky">90.915</td>
    <td class="tg-0pky">8.972</td>
    <td class="tg-0pky">0.626</td>
    <td class="tg-0pky">0.009</td>
    <td class="tg-0pky">0.049</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/rPSjTiK2XdPwkPB" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mbv2_torchvision_cifar100_224_e100</td>
    <td class="tg-0pky">torchvision</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-0pky">68.940</td>
    <td class="tg-0pky">91.114</td>
    <td class="tg-0pky">8.972</td>
    <td class="tg-0pky">0.626</td>
    <td class="tg-0pky">0.008</td>
    <td class="tg-0pky">0.060</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/obJLoRRsoH8CJzT" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mnasnet_a1_1_3_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-0pky">67.762</td>
    <td class="tg-0pky">89.587</td>
    <td class="tg-0pky">14.811</td>
    <td class="tg-0pky">1.083</td>
    <td class="tg-0pky">0.009</td>
    <td class="tg-0pky">0.058</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/JkkRqw2j4NQjsGd" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mnasnet_a1_1_3_se_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-0pky">70.098</td>
    <td class="tg-fymr">*91.504</td>
    <td class="tg-0pky">65.567</td>
    <td class="tg-0pky">2.857</td>
    <td class="tg-0pky">0.015</td>
    <td class="tg-0pky">0.124</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/nKP2ExeiDrnR4N6" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mnasnet_b1_1_3_custom_cifar100_224_e100_sgd</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">sgd</td>
    <td class="tg-0pky">69.119</td>
    <td class="tg-0pky">89.687</td>
    <td class="tg-0pky">19.567</td>
    <td class="tg-0pky">1.080</td>
    <td class="tg-0pky">0.011</td>
    <td class="tg-0pky">0.062</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/2xR6RTmcj8Son9E" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mnasnet_b1_1_3_torchvision_cifar100_224_e100_sgd</td>
    <td class="tg-0pky">torchvision</td>
    <td class="tg-0pky">sgd</td>
    <td class="tg-0pky">69.119</td>
    <td class="tg-0pky">89.726</td>
    <td class="tg-0pky">19.567</td>
    <td class="tg-0pky">1.080</td>
    <td class="tg-0pky">0.009</td>
    <td class="tg-0pky">0.058</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/jdKcfs4tznERdHA" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mbv3_large_custom_cifar100_224_e100_sgd</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">sgd</td>
    <td class="tg-0pky">66.304</td>
    <td class="tg-0pky">88.349</td>
    <td class="tg-0pky">10.750</td>
    <td class="tg-0pky">0.453</td>
    <td class="tg-0pky">0.009</td>
    <td class="tg-0pky">0.040</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/SfmNT8E8an99xWa" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mbv3_large_se_custom_cifar100_224_e100_sgd</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">sgd</td>
    <td class="tg-0pky">68.031</td>
    <td class="tg-0pky">89.607</td>
    <td class="tg-0pky">16.493</td>
    <td class="tg-0pky">0.457</td>
    <td class="tg-0pky">0.010</td>
    <td class="tg-0pky">0.046</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/JttoQzK6X59fAdA" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mbv3_large_se_hsigmoid_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-0pky">68.151</td>
    <td class="tg-0pky">89.287</td>
    <td class="tg-0pky">16.493</td>
    <td class="tg-0pky">0.457</td>
    <td class="tg-0pky">0.010</td>
    <td class="tg-0pky">0.050</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/tm5ZL9qHrsC7s4W" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mbv3_small_custom_cifar100_224_e100_sgd</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">sgd</td>
    <td class="tg-0pky">66.364</td>
    <td class="tg-0pky">88.688</td>
    <td class="tg-fymr">*7.977</td>
    <td class="tg-0pky">0.445</td>
    <td class="tg-0pky">0.008</td>
    <td class="tg-0pky">0.042</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/j9t5bkn2XSX89eZ" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mbv3_small_se_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-0pky">67.891</td>
    <td class="tg-0pky">89.467</td>
    <td class="tg-0pky">13.719</td>
    <td class="tg-fymr">*0.450</td>
    <td class="tg-0pky">0.010</td>
    <td class="tg-0pky">0.054</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/eJ5soxokHqDY9ni" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">mbv3_small_se_hsigmoid_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-0pky">68.411</td>
    <td class="tg-0pky">89.517</td>
    <td class="tg-0pky">13.719</td>
    <td class="tg-fymr">*0.450</td>
    <td class="tg-0pky">0.010</td>
    <td class="tg-0pky">0.044</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/FAHGmWCQkf587ea" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">sfv1_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-fymr">*71.715</td>
    <td class="tg-0pky">91.334</td>
    <td class="tg-0pky">36.461</td>
    <td class="tg-0pky">2.608</td>
    <td class="tg-0pky">0.010</td>
    <td class="tg-0pky">0.062</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/osWkJZyst4wD8B8" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">sfv2_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-0pky">70.457</td>
    <td class="tg-0pky">90.645</td>
    <td class="tg-0pky">27.356</td>
    <td class="tg-0pky">1.571</td>
    <td class="tg-0pky">0.013</td>
    <td class="tg-0pky">0.056</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/6Gbj9GfNc7tjM9c" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
  <td class="tg-0pky">sfv2_torchvision_cifar100_224_e100</td>
  <td class="tg-0pky">torchvision</td>
  <td class="tg-0pky">rmsprop</td>
  <td class="tg-0pky">70.148</td>
  <td class="tg-0pky">91.314</td>
  <td class="tg-0pky">21.171</td>
  <td class="tg-0pky">1.178</td>
  <td class="tg-0pky">0.011</td>
  <td class="tg-0pky">0.062</td>
  <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/cpeJREKmyRsxBko" target="_blank" rel="noopener noreferrer">link</a></td>
</tr>
</tbody>
</table>

## ResNet系列

当前已实现的`ResNet`系列模型有：`ResNet/ResNeXt/ResNet_d/ResNeXt_d/SKNet/ResNeSt`

相同训练配置如下：

1. 训练次数：`100`轮
2. 数据集：`CIFAR100`
3. 图像预处理：
   1. 较短边扩展到`224`，然后中心裁剪`224x224`
   2. 图像均值：`(0.45, 0.45, 0.45)`
   3. 标准差：`(0.225, 0.225, 0.225)`
4. 批量大小：`32`
5. 损失函数：`CrossEntropyLoss`
6. 学习率/权重衰减：
   1. 针对`RMSProp`：`3e-4/1e-4`
   2. 针对`SGD`：`0.0125(0.9 momentum)/1e-5`
7. 学习率调度：`WarmUP`（前`5`轮） + `MultiStepLR`（第`40/70`轮衰减一次，衰减因子为`0.1`）
8. 训练加速：混合精度训练
9. `GPU/CPU：GeForce RTX 2080 Ti + Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz`

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">model</th>
    <th class="tg-7btt">custom</th>
    <th class="tg-7btt">optimizer</th>
    <th class="tg-7btt">bn2/relu<br></th>
    <th class="tg-7btt">avg<br></th>
    <th class="tg-7btt">top1 acc</th>
    <th class="tg-7btt">top5 acc</th>
    <th class="tg-7btt">model size(MB)</th>
    <th class="tg-7btt">gflops</th>
    <th class="tg-7btt">gpu infer(s)</th>
    <th class="tg-7btt">cpu infer(s)</th>
    <th class="tg-amwm">config/log</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">resnet50<br></td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">64.137</td>
    <td class="tg-0lax">88.858</td>
    <td class="tg-0lax">90.458</td>
    <td class="tg-0lax">8.219</td>
    <td class="tg-0lax">0.014</td>
    <td class="tg-0lax">0.106</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/xgSNccjgbYFoc6w" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnet50</td>
    <td class="tg-0lax">torchvision</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">64.716</td>
    <td class="tg-0lax">88.648</td>
    <td class="tg-0lax">90.458</td>
    <td class="tg-0lax">8.219</td>
    <td class="tg-0lax">0.009</td>
    <td class="tg-0lax">0.110</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/gESKNm5E9XZnsbF" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnet_d50</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">70.607</td>
    <td class="tg-0lax">92.492</td>
    <td class="tg-0lax">90.531</td>
    <td class="tg-0lax">8.703</td>
    <td class="tg-0lax">0.010</td>
    <td class="tg-0lax">0.090</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/x6jaK2FLFX333i9" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnet_d50</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">sgd</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">71.615</td>
    <td class="tg-0lax">91.673</td>
    <td class="tg-0lax">90.531</td>
    <td class="tg-0lax">8.703</td>
    <td class="tg-0lax">0.010</td>
    <td class="tg-0lax">0.120</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/XpDP7e5CBPJrAb6" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnextd50_32x4d</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">avg<br></td>
    <td class="tg-0lax">67.812</td>
    <td class="tg-0lax">90.905</td>
    <td class="tg-0lax">88.443</td>
    <td class="tg-0lax">8.776</td>
    <td class="tg-0lax">0.016</td>
    <td class="tg-0lax">0.088</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/aRzFfg5iP6A9ENj" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnextd50_32x4d</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">fast_avg<br></td>
    <td class="tg-0lax">70.138</td>
    <td class="tg-0lax">92.103</td>
    <td class="tg-0lax">88.443</td>
    <td class="tg-0lax">8.516</td>
    <td class="tg-0lax">0.013</td>
    <td class="tg-0lax">0.088</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/7dfjbpToMek7Q2L" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnext50_32x4d</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">69.888</td>
    <td class="tg-0lax">91.673</td>
    <td class="tg-0lax">88.443</td>
    <td class="tg-0lax">8.516</td>
    <td class="tg-0lax">0.013</td>
    <td class="tg-0lax">0.089</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/7b3etLXzHeKo4ZR" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnext50_32x4d</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">sgd</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">62.380</td>
    <td class="tg-0lax">86.302</td>
    <td class="tg-0lax">88.443</td>
    <td class="tg-0lax">8.516</td>
    <td class="tg-0lax">0.013</td>
    <td class="tg-0lax">0.087</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/3ajREQkkK45bG7W" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnext50_32x4d</td>
    <td class="tg-0lax">torchvision</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">67.572</td>
    <td class="tg-0lax">91.024</td>
    <td class="tg-0lax">88.443</td>
    <td class="tg-0lax">8.515</td>
    <td class="tg-0lax">0.013</td>
    <td class="tg-0lax">0.090</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/GFQ4ZAQeiXdtrCd" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnext50_32x4d</td>
    <td class="tg-0lax">torchvision</td>
    <td class="tg-0lax">sgd</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">62.151</td>
    <td class="tg-0lax">86.102</td>
    <td class="tg-0lax">88.443</td>
    <td class="tg-0lax">8.515</td>
    <td class="tg-0lax">0.013</td>
    <td class="tg-0lax">0.096</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/b7Efx6s5PYg2Hjg" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnetxtd50_32x4d</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">*71.655</td>
    <td class="tg-0lax">92.412</td>
    <td class="tg-0lax">88.443</td>
    <td class="tg-0lax">8.516</td>
    <td class="tg-0lax">0.014</td>
    <td class="tg-0lax">0.088</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/zpCKpMHzckzAo4D" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnetxtd50_32x4d</td>
    <td class="tg-0lax">zcls<br></td>
    <td class="tg-0lax">sgd</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">71.366</td>
    <td class="tg-0lax">91.404</td>
    <td class="tg-0lax">88.443</td>
    <td class="tg-0lax">8.515</td>
    <td class="tg-0lax">0.027</td>
    <td class="tg-0lax">0.089</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/BwtCx5JmJtePjSt" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">sknet50</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">no</td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">69.978</td>
    <td class="tg-0lax">92.153</td>
    <td class="tg-0lax">/</td>
    <td class="tg-0lax">/</td>
    <td class="tg-0lax">/</td>
    <td class="tg-0lax">/</td>
    <td class="tg-0lax">/</td>
  </tr>
  <tr>
    <td class="tg-0lax">sknet50</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">yes<br></td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">70.397</td>
    <td class="tg-0lax">92.382</td>
    <td class="tg-0lax">88.443</td>
    <td class="tg-0lax">8.515</td>
    <td class="tg-0lax">0.014</td>
    <td class="tg-0lax">0.088</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/X5igYtFJH8A8xMx" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnest50_2s2x40d</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">avg<br></td>
    <td class="tg-0lax">68.760</td>
    <td class="tg-0lax">91.444</td>
    <td class="tg-0lax">92.936</td>
    <td class="tg-0lax">8.789</td>
    <td class="tg-0lax">0.025</td>
    <td class="tg-0lax">0.117</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/c6kcAx5mEzwsiTt" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnest50_2s2x40d</td>
    <td class="tg-0lax">zcls</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">fast_avg<br></td>
    <td class="tg-0lax">71.256</td>
    <td class="tg-0lax">92.742</td>
    <td class="tg-0lax">92.936</td>
    <td class="tg-0lax">8.789</td>
    <td class="tg-0lax">0.022</td>
    <td class="tg-0lax">0.124</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/C4f3nejFfKpQjpi" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnest50_2s2x40d</td>
    <td class="tg-0lax">torchvision</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">avg<br></td>
    <td class="tg-0lax">69.069</td>
    <td class="tg-0lax">91.554</td>
    <td class="tg-0lax">95.748</td>
    <td class="tg-0lax">10.418</td>
    <td class="tg-0lax">0.021</td>
    <td class="tg-0lax">0.131</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/A3C5DxxrQMapLaS" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">resnest50_2s2x40d</td>
    <td class="tg-0lax">torchvision</td>
    <td class="tg-0lax">rmsprop</td>
    <td class="tg-0lax">no<br></td>
    <td class="tg-0lax">fast_avg<br></td>
    <td class="tg-0lax">71.526</td>
    <td class="tg-0lax">*92.921</td>
    <td class="tg-0lax">95.748</td>
    <td class="tg-0lax">8.787</td>
    <td class="tg-0lax">0.021</td>
    <td class="tg-0lax">0.132</td>
    <td class="tg-0lax"><a href="https://cloud.zhujian.tech:9300/s/yb5wHXpZZ23RCJo" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
</tbody>
</table>