
# Benchmark - ResNet/ResNeXt

## Convert

The torchvision pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/torchvision_resnet_to_zcls_resnet.py
```

## Benchmark

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj">arch</th>
    <th class="tg-uzvj">framework</th>
    <th class="tg-uzvj">top1</th>
    <th class="tg-uzvj">top5</th>
    <th class="tg-7btt">input_size</th>
    <th class="tg-7btt">dataset</th>
    <th class="tg-amwm">params_size/MB<br></th>
    <th class="tg-amwm">gflops<br></th>
    <th class="tg-amwm">cpu_infer/s</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8">resnet18</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">69.224</td>
    <td class="tg-9wq8">88.808</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">44.592</td>
    <td class="tg-baqh">3.638</td>
    <td class="tg-baqh">0.030</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnet18</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">69.222</td>
    <td class="tg-9wq8">88.808</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">44.592</td>
    <td class="tg-baqh">3.638</td>
    <td class="tg-baqh">0.032</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnet34</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">72.821</td>
    <td class="tg-9wq8">91.071</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">83.152</td>
    <td class="tg-baqh">7.343</td>
    <td class="tg-baqh">0.060</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnet34</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">72.817</td>
    <td class="tg-9wq8">91.073</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">83.152</td>
    <td class="tg-baqh">7.343</td>
    <td class="tg-baqh">0.055</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnet50</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">75.692</td>
    <td class="tg-9wq8">92.768</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">97.492M</td>
    <td class="tg-baqh">8.223</td>
    <td class="tg-baqh">0.082</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnet50</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">75.690</td>
    <td class="tg-9wq8">92.770</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">97.492</td>
    <td class="tg-baqh">8.223</td>
    <td class="tg-baqh">0.078</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnet101</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">76.965</td>
    <td class="tg-9wq8">93.526</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">169.942</td>
    <td class="tg-baqh">15.668</td>
    <td class="tg-baqh">0.145</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnet101</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">76.967</td>
    <td class="tg-9wq8">93.528</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">169.942</td>
    <td class="tg-baqh">15.668</td>
    <td class="tg-baqh">0.144</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnet152</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">78.141</td>
    <td class="tg-9wq8">93.946</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">229.617</td>
    <td class="tg-baqh">23.118</td>
    <td class="tg-baqh">0.208</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnet152</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">78.139</td>
    <td class="tg-9wq8">93.946</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">229.617</td>
    <td class="tg-baqh">23.118</td>
    <td class="tg-baqh">0.210</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnext50_32x4d</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">77.353</td>
    <td class="tg-9wq8">93.610</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">95.478</td>
    <td class="tg-baqh">8.519</td>
    <td class="tg-baqh">0.106</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnext50_32x4d</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">77.353</td>
    <td class="tg-9wq8">93.612</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">95.478</td>
    <td class="tg-baqh">8.519</td>
    <td class="tg-baqh">0.103</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnext101_32x8d</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">79.075</td>
    <td class="tg-9wq8">94.540</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">338.712</td>
    <td class="tg-baqh">32.953</td>
    <td class="tg-baqh">0.311</td>
  </tr>
  <tr>
    <td class="tg-9wq8">resnext101_32x8d</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">79.063</td>
    <td class="tg-9wq8">94.538</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">338.712</td>
    <td class="tg-baqh">32.953</td>
    <td class="tg-baqh">0.298</td>
  </tr>
</tbody>
</table>

*CPU_INFO: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz*