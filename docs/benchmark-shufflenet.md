
# Benchmark - ShuffleNet

## Convert

The official shufflenet_v1/torchvision shufflenet_v2 pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/official_shufflenet_v1_to_zcls_shufflenet_v1.py
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
    <td class="tg-9wq8">shufflenet_v1_3g1x</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">/</td>
    <td class="tg-9wq8">/</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">7.113</td>
    <td class="tg-baqh">0.287</td>
    <td class="tg-baqh">0.019</td>
  </tr>
  <tr>
    <td class="tg-9wq8">shufflenet_v1_8g1x</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">/</td>
    <td class="tg-9wq8">/</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">9.284</td>
    <td class="tg-baqh">0.296</td>
    <td class="tg-baqh">0.022</td>
  </tr>
  <tr>
    <td class="tg-9wq8">shufflenet_v2_x0_5</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">59.729</td>
    <td class="tg-9wq8">81.226</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">5.210</td>
    <td class="tg-baqh">0.085</td>
    <td class="tg-baqh">0.013</td>
  </tr>
  <tr>
    <td class="tg-9wq8">shufflenet_v2_x0_5</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">59.729</td>
    <td class="tg-9wq8">81.230</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">5.214</td>
    <td class="tg-baqh">0.085</td>
    <td class="tg-baqh">0.012</td>
  </tr>
  <tr>
    <td class="tg-9wq8">shufflenet_v2_x1_0</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">68.942</td>
    <td class="tg-9wq8">88.218</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">8.693</td>
    <td class="tg-baqh">0.300</td>
    <td class="tg-baqh">0.018</td>
  </tr>
  <tr>
    <td class="tg-9wq8">shufflenet_v2_x1_0</td>
    <td class="tg-9wq8">torchvision</td>
    <td class="tg-9wq8">68.940</td>
    <td class="tg-9wq8">88.218</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">8.692</td>
    <td class="tg-baqh">0.298</td>
    <td class="tg-baqh">0.018</td>
  </tr>
</tbody>
</table>

*CPU_INFO: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz*