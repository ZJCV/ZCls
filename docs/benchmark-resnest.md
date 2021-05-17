
# Benchmark - SENet/SKNet/ResNeSt

## Convert

The senet/sknet/resnest pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/moskomule_se_resnet_to_zcls_se_resnet.py
$ python tools/converters/syt2_sknet_to_zcls_sknet.py
$ python tools/converters/official_resnest_to_zcls_resnest.py
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
    <td class="tg-c3ow">sknet50</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">78.343</td>
    <td class="tg-c3ow">93.944</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">104.885</td>
    <td class="tg-baqh">9.000</td>
    <td class="tg-baqh">0.143</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest50_fast_2s1x64d</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">70.395</td>
    <td class="tg-c3ow">89.393</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">104.840</td>
    <td class="tg-baqh">8.719</td>
    <td class="tg-baqh">0.092</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest50_fast_2s1x64d</td>
    <td class="tg-c3ow">official</td>
    <td class="tg-c3ow">80.380</td>
    <td class="tg-c3ow">95.252</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">104.840</td>
    <td class="tg-baqh">8.716</td>
    <td class="tg-baqh">0.083</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest50</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">104.840</td>
    <td class="tg-baqh">10.805</td>
    <td class="tg-baqh">0.435</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest50</td>
    <td class="tg-c3ow">official</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">104.840</td>
    <td class="tg-baqh">10.801</td>
    <td class="tg-baqh">0.424</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest101</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">184.155</td>
    <td class="tg-baqh">20.496</td>
    <td class="tg-baqh">0.667</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest101</td>
    <td class="tg-c3ow">official</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">184.155</td>
    <td class="tg-baqh">20.491</td>
    <td class="tg-baqh">0.623</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest200</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">267.798</td>
    <td class="tg-baqh">34.991</td>
    <td class="tg-baqh">0.344</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest200</td>
    <td class="tg-c3ow">official</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">267.798</td>
    <td class="tg-baqh">34.981</td>
    <td class="tg-baqh">0.350</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest269</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">423.162</td>
    <td class="tg-baqh">45.083</td>
    <td class="tg-baqh">0.454</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnest269</td>
    <td class="tg-c3ow">official</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
    <td class="tg-baqh">423.162</td>
    <td class="tg-baqh">45.070</td>
    <td class="tg-baqh">0.476</td>
  </tr>
  </tbody>
</table>

*CPU_INFO: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz*