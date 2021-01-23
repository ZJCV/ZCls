
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
6. 学习率调度：`WarmUP`（前`5`轮） + `MultiStepLR`（第`40/70`轮衰减一次，衰减因子为`0.1`）
7. 训练加速：混合精度训练
8. `GPU/CPU：GeForce RTX 2080 Ti + Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz`

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
</style>
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
    <td class="tg-fymr">0.006</td>
    <td class="tg-fymr">0.030</td>
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
    <td class="tg-fymr">91.504</td>
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
    <td class="tg-fymr">7.977</td>
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
    <td class="tg-fymr">0.450</td>
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
    <td class="tg-fymr">0.450</td>
    <td class="tg-0pky">0.010</td>
    <td class="tg-0pky">0.044</td>
    <td class="tg-0pky"><a href="https://cloud.zhujian.tech:9300/s/FAHGmWCQkf587ea" target="_blank" rel="noopener noreferrer">link</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">sfv1_custom_cifar100_224_e100</td>
    <td class="tg-0pky">zcls</td>
    <td class="tg-0pky">rmsprop</td>
    <td class="tg-fymr">71.715</td>
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

<!-- ## ResNet系列 -->

