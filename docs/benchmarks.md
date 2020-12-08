
# 基准测试

## ResNet

<table>
<thead>
  <tr>
    <th>config</th>
    <th>backbone</th>
    <th>pretrain</th>
    <th>custom</th>
    <th>gpus</th>
    <th>batchs</th>
    <th>top1 acc</th>
    <th>top5 acc</th>
    <th>resolution(TxHxW)</th>
    <th>model_size(MB)</th>
    <th>GFlops</th>
    <th>inference_time(s)</th>
    <th>log</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/RFnPtJ4WsgryrbB" target="_blank" rel="noopener noreferrer">r50_custom_cifar100_224</a></td>
    <td>resnet50</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>96</td>
    <td>40.665</td>
    <td>68.403</td>
    <td>3x224x224</td>
    <td>90.458</td>
    <td>8.219</td>
    <td>0.010</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/nLBNPH9pR5ZDrEr" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/4ADbnYzsgk2SQfi" target="_blank" rel="noopener noreferrer">r50_custom_pretrained_cifar100_224</a></td>
    <td>resnet50</td>
    <td>torchvision pretrained</td>
    <td>custom</td>
    <td>1</td>
    <td>96</td>
    <td>82.500</td>
    <td>97.381</td>
    <td>3x224x224</td>
    <td>90.458</td>
    <td>8.219</td>
    <td>0.011</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/wjx27CMPFMFpyNm" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/bMoXAjTcxtnmjCw" target="_blank" rel="noopener noreferrer">r50_pytorch_cifar100_224</a></td>
    <td>resnet50</td>
    <td>from scratch</td>
    <td>torchvision</td>
    <td>1</td>
    <td>96</td>
    <td>40.813</td>
    <td>69.147</td>
    <td>3x224x224</td>
    <td>90.458</td>
    <td>8.219</td>
    <td>0.010</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/Htifk7XR3TrWEnk" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/KTjK4LsoPMayBFY" target="_blank" rel="noopener noreferrer">r50_pytorch_pretrained_cifar100_224</a></td>
    <td>resnet50</td>
    <td>torchvision pretrained</td>
    <td>torchvision</td>
    <td>1</td>
    <td>96</td>
    <td>82.351</td>
    <td>97.450</td>
    <td>3x224x224</td>
    <td>90.458</td>
    <td>8.219</td>
    <td>0.010</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/CxoaLWEsQJ6QSMo" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
</tbody>
</table>

## Norm

<table>
<thead>
  <tr>
    <th>config</th>
    <th>backbone</th>
    <th>norm_layer</th>
    <th>pretrain</th>
    <th>custom</th>
    <th>gpus</th>
    <th>batchs</th>
    <th>top1 acc</th>
    <th>top5 acc</th>
    <th>resolution(TxHxW)</th>
    <th>inference_time(image/s)</th>
    <th>gpu_mem(G)</th>
    <th>ckpt</th>
    <th>log</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/CL58YL7S7wHPTyr" target="_blank" rel="noopener noreferrer">r50_custom_cifar100_224</a></td>
    <td>resnet50</td>
    <td>bn</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>96</td>
    <td>39.504</td>
    <td>67.659</td>
    <td>3x224x224</td>
    <td>/</td>
    <td>8.05</td>
    <td>/</td>
    <td>/</td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/aqfxmEGEBJjCAdE" target="_blank" rel="noopener noreferrer">r50_gn_custom_cifar100_224</a></td>
    <td>resnet50</td>
    <td>gn</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>96</td>
    <td>45.992</td>
    <td>74.266</td>
    <td>3x224x224</td>
    <td>/</td>
    <td>8.05</td>
    <td>/</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/TLzdegHpHjzDWA7" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/bFNc4ttozdf4Y32" target="_blank" rel="noopener noreferrer">r50_custom_pretrained_cifar100_224</a></td>
    <td>resnet50</td>
    <td>bn</td>
    <td>torchvision pretrained</td>
    <td>custom</td>
    <td>1</td>
    <td>96</td>
    <td>82.183</td>
    <td>97.321</td>
    <td>3x224x224</td>
    <td>/</td>
    <td>8.05</td>
    <td>/</td>
    <td>/</td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/fgnmCpyH8w3YpPQ" target="_blank" rel="noopener noreferrer">r50_partial_bn_custom_pretrained_cifar100_224</a></td>
    <td>resnet50</td>
    <td>partial bn</td>
    <td>torchvision pretrained</td>
    <td>custom</td>
    <td>1</td>
    <td>96</td>
    <td>84.286</td>
    <td>97.708</td>
    <td>3x224x224</td>
    <td>/</td>
    <td>8.05</td>
    <td>/</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/5PTbRiT8G2QN2ok" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/N4aXzZRRb9MtMom" target="_blank" rel="noopener noreferrer">r50_fix_bn_custom_pretrained_cifar100_224</a></td>
    <td>resnet50</td>
    <td>fix bn</td>
    <td>torchvision pretrained</td>
    <td>custom</td>
    <td>1</td>
    <td>96</td>
    <td>84.325</td>
    <td>97.639</td>
    <td>3x224x224</td>
    <td>/</td>
    <td>8.05</td>
    <td><br>/</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/FmscE4jeHkLtp9H" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/kyz2HKrEf5H3ifE" target="_blank" rel="noopener noreferrer">r50_custom_cifar100_224_g2</a></td>
    <td>resnet50</td>
    <td>bn</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>2</td>
    <td>96</td>
    <td>40.035</td>
    <td>67.934</td>
    <td>3x224x224</td>
    <td>/</td>
    <td>8.14</td>
    <td>/</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/nLBNPH9pR5ZDrEr" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/oz7rg4Lex5kgYwP" target="_blank" rel="noopener noreferrer">r50_sync_bn_custom_cifar100_224</a></td>
    <td>resnet50</td>
    <td>sync bn</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>2</td>
    <td>96</td>
    <td>38.443</td>
    <td>66.637</td>
    <td>3x224x224</td>
    <td>/</td>
    <td>8.14</td>
    <td>/</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/GwdxDsR7dq7kkDb" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
</tbody>
</table>

## MobileNet

<table>
<thead>
  <tr>
    <th>config</th>
    <th>backbone</th>
    <th>act</th>
    <th>pretrain</th>
    <th>custom</th>
    <th>gpus</th>
    <th>batchs</th>
    <th>top1 acc</th>
    <th>top5 acc</th>
    <th>resolution(TxHxW)</th>
    <th>model_size(MB)</th>
    <th>GFlops</th>
    <th>inference_time(s)</th>
    <th>log</th>
  </tr>
</thead>
<tbody>
<tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/WHKdiXn89TqmwdD" target="_blank" rel="noopener noreferrer">mbv1_0.25x_cifar100_224</a></td>
    <td>mobilenetv1</td>
    <td>ReLU</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>128</td>
    <td>34.464</td>
    <td>65.467</td>
    <td>3x224x224</td>
    <td>1.793</td>
    <td>0.087</td>
    <td>0.006</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/qi8ETx2n8kCz24k" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
<tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/ZWztqQEgioEa6oz" target="_blank" rel="noopener noreferrer">mbv1_0.5x_cifar100_224</a></td>
    <td>mobilenetv1</td>
    <td>ReLU</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>128</td>
    <td>37.718</td>
    <td>69.195</td>
    <td>3x224x224</td>
    <td>5.080</td>
    <td>0.309</td>
    <td>0.005</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/A89zy857erXd8FZ" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/XHDY2aCTyQdbwWZ" target="_blank" rel="noopener noreferrer">mbv1_0.75x_cifar100_224</a></td>
    <td>mobilenetv1</td>
    <td>ReLU</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>128</td>
    <td>40.012</td>
    <td>69.660</td>
    <td>3x224x224</td>
    <td>9.863</td>
    <td>0.666</td>
    <td>0.005</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/oQbY2FgxpKHTF29" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/pd5EfXtpFpRgHfT" target="_blank" rel="noopener noreferrer">mbv1_1x_cifar100_224</a></td>
    <td>mobilenetv1</td>
    <td>ReLU</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>128</td>
    <td>41.970</td>
    <td>73.012</td>
    <td>3x224x224</td>
    <td>16.144</td>
    <td>1.158</td>
    <td>0.005</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/mgLQr3Ad4eYPeT4" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/SFT8QqafeMyHWPd" target="_blank" rel="noopener noreferrer">mbv2_custom_0.25x_relu6_cifar100_224</a></td>
    <td>mobilenetv2</td>
    <td>ReLU6</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>128</td>
    <td>40.358</td>
    <td>71.865</td>
    <td>3x224x224</td>
    <td>1.833</td>
    <td>0.074</td>
    <td>0.009</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/iBpbDWLYyBbdmzb" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/PJr3fSoTPTeZ72T" target="_blank" rel="noopener noreferrer">mbv2_custom_0.5x_relu6_cifar100_224</a></td>
    <td>mobilenetv2</td>
    <td>ReLU6</td>
    <td>from scratch </td>
    <td>custom</td>
    <td>1</td>
    <td>128</td>
    <td>42.573</td>
    <td>72.656</td>
    <td>3x224x224</td>
    <td>4.673</td>
    <td>0.197</td>
    <td>0.010</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/tMP4i2KTaXm9wst" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/t3P96ayHngKF8HG" target="_blank" rel="noopener noreferrer">mbv2_custom_0.75x_relu6_cifar100_224</a></td>
    <td>mobilenetv2</td>
    <td>ReLU6</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>128</td>
    <td>45.342</td>
    <td>74.219</td>
    <td>3x224x224</td>
    <td>8.541</td>
    <td>0.433</td>
    <td>0.010</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/jePrMzLiq6Cmdsb" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/4YPcMJtwaHetSPr" target="_blank" rel="noopener noreferrer">mbv2_custom_1x_relu6_cifar100_224</a></td>
    <td>mobilenetv2</td>
    <td>ReLU6</td>
    <td>from scratch</td>
    <td>custom</td>
    <td>1</td>
    <td>128</td>
    <td>46.746</td>
    <td>75.781</td>
    <td>3x224x224</td>
    <td>13.370</td>
    <td>0.628</td>
    <td>0.009</td>
    <td><a href="https://cloud.zhujian.tech:9300/s/qoNQwAW7DJWJj44" target="_blank" rel="noopener noreferrer">log</a></td>
  </tr>
  <tr>
      <td><a href="https://cloud.zhujian.tech:9300/s/t3jiDqjCngAEtYN" target="_blank" rel="noopener noreferrer">mbv2_pytorch_1x_relu6_cifar100_224</a></td>
      <td>mobilenetv2</td>
      <td>ReLU6</td>
      <td>from scratch</td>
      <td>torchvision</td>
      <td>1</td>
      <td>128</td>
      <td>46.143</td>
      <td>74.733</td>
      <td>3x224x224</td>
      <td>8.972</td>
      <td>0.62</td>
      <td>0.010</td>
      <td><a href="https://cloud.zhujian.tech:9300/s/37bJT6eXz7D6aiN" target="_blank" rel="noopener noreferrer">log</a></td>
    </tr>
</tbody>
</table>