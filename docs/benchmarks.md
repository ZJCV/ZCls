
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