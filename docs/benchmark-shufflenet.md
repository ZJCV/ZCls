
# Benchmark - ShuffleNet

## Convert

The torchvision pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/torchvision_shufflenet_to_zcls_shufflenet.py
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
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj">arch</th>
    <th class="tg-uzvj">framework</th>
    <th class="tg-uzvj">top1</th>
    <th class="tg-uzvj">top5</th>
    <th class="tg-uzvj">input_size</th>
    <th class="tg-uzvj">dataset</th>
    <th class="tg-amwm">params_size/MB<br></th>
    <th class="tg-amwm">gflops<br></th>
    <th class="tg-amwm">cpu_infer/s</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8">shufflenet_v2_x0_5</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">59.735</td>
    <td class="tg-9wq8">81.222</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">5.214</td>
    <td class="tg-baqh">0.085</td>
    <td class="tg-baqh">0.015</td>
  </tr>
  <tr>
    <td class="tg-9wq8">shufflenet_v2_x0_5</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">59.735</td>
    <td class="tg-9wq8">81.226</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">5.214</td>
    <td class="tg-baqh">0.085</td>
    <td class="tg-baqh">0.014</td>
  </tr>
  <tr>
    <td class="tg-9wq8">shufflenet_v2_x1_0</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">68.944</td>
    <td class="tg-9wq8">88.214</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">8.692</td>
    <td class="tg-baqh">0.298</td>
    <td class="tg-baqh">0.021</td>
  </tr>
  <tr>
    <td class="tg-9wq8">shufflenet_v2_x1_0</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">68.942</td>
    <td class="tg-9wq8">88.214</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">8.692</td>
    <td class="tg-baqh">0.298</td>
    <td class="tg-baqh">0.021</td>
  </tr>
</tbody>
</table>

*CPU_INFO: Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz*