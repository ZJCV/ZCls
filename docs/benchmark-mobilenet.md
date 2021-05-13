
# Benchmark - MobileNet

## Convert

The torchvision pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/torchvision_mobilenet_to_zcls_mobilenet.py
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
.tg .tg-r4m4{color:rgba(0, 0, 0, 0.87);text-align:center;vertical-align:top}
.tg .tg-gg09{background-color:rgba(0, 0, 0, 0.035);color:rgba(0, 0, 0, 0.87);text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-b0mr{border-color:inherit;color:rgba(0, 0, 0, 0.87);text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">arch</th>
    <th class="tg-7btt">framework</th>
    <th class="tg-7btt">top1</th>
    <th class="tg-7btt">top5</th>
    <th class="tg-uzvj">input_size</th>
    <th class="tg-uzvj">dataset</th>
    <th class="tg-amwm">params_size/MB<br></th>
    <th class="tg-amwm">gflops<br></th>
    <th class="tg-amwm">cpu_infer/s</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-b0mr"><span style="font-weight:400">mobilenet_v2</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">zcls</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">71.429</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">90.151</span></td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">13.370</td>
    <td class="tg-baqh">0.628</td>
    <td class="tg-baqh">0.027</td>
  </tr>
  <tr>
    <td class="tg-b0mr"><span style="font-weight:400">mobilenet_v2</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">torchvision</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">71.429</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">90.151</span></td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">13.370</td>
    <td class="tg-baqh">0.628</td>
    <td class="tg-baqh">0.025</td>
  </tr>
  <tr>
    <td class="tg-b0mr"><span style="font-weight:400">mnasnet0_5</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">zcls</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">66.961</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">86.946</span></td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">8.463</td>
    <td class="tg-baqh">0.221</td>
    <td class="tg-baqh">0.017</td>
  </tr>
  <tr>
    <td class="tg-b0mr"><span style="font-weight:400">mnasnet0_5</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">torchvision</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">66.961</span></td>
    <td class="tg-b0mr"><span style="font-weight:400">86.946</span></td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">8.463</td>
    <td class="tg-baqh">0.221</td>
    <td class="tg-baqh">0.015</td>
  </tr>
  <tr>
    <td class="tg-r4m4"><span style="font-weight:400">mnasnet1_0</span></td>
    <td class="tg-r4m4"><span style="font-weight:400">zcls</span></td>
    <td class="tg-r4m4"><span style="font-weight:400">73.249</span></td>
    <td class="tg-r4m4"><span style="font-weight:400">91.235</span></td>
    <td class="tg-baqh">224x224</td>
    <td class="tg-baqh">imagenet</td>
    <td class="tg-baqh">16.721</td>
    <td class="tg-baqh">0.651</td>
    <td class="tg-baqh">0.021</td>
  </tr>
  <tr>
    <td class="tg-gg09"><span style="font-weight:400">mnasnet1_0</span></td>
    <td class="tg-gg09"><span style="font-weight:400">torchvision</span></td>
    <td class="tg-gg09"><span style="font-weight:400">73.249</span></td>
    <td class="tg-gg09"><span style="font-weight:400">91.235</span></td>
    <td class="tg-baqh">224x224</td>
    <td class="tg-baqh">imagenet</td>
    <td class="tg-baqh">16.721</td>
    <td class="tg-baqh">0.651G</td>
    <td class="tg-baqh">0.023</td>
  </tr>
</tbody>
</table>

*CPU_INFO: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz*