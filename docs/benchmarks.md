
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
    <th>inference_time(image/s)</th>
    <th>gpu_mem(G)</th>
    <th>ckpt</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/CL58YL7S7wHPTyr" target="_blank" rel="noopener noreferrer">r50_custom_cifar100_224</a></td>
    <td>resnet50</td>
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
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/4ADbnYzsgk2SQfi" target="_blank" rel="noopener noreferrer">r50_custom_pretrained_cifar100_224</a></td>
    <td>resnet50</td>
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
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/ZfHkTxSqce4zCwB" target="_blank" rel="noopener noreferrer">r50_pytorch_cifar100_224</a></td>
    <td>resnet50</td>
    <td>from scratch</td>
    <td>torchvision</td>
    <td>1</td>
    <td>96</td>
    <td>41.032</td>
    <td>69.018</td>
    <td>3x224x224</td>
    <td>/</td>
    <td>8.05</td>
    <td>/</td>
  </tr>
  <tr>
    <td><a href="https://cloud.zhujian.tech:9300/s/k4aqzLAnqtXCM8X" target="_blank" rel="noopener noreferrer">r50_pytorch_pretrained_cifar100_224</a></td>
    <td>resnet50</td>
    <td>torchvision pretrained</td>
    <td>torchvision</td>
    <td>1</td>
    <td>96</td>
    <td>82.183</td>
    <td>97.321</td>
    <td>3x224x224</td>
    <td>/</td>
    <td>8.05</td>
    <td>/</td>
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