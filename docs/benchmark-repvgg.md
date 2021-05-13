
# Benchmark - RepVGG

## Convert

The RepVGG pretraining model can be transformed into zcls by script

```
$ cd /path/to/ZCls
$ python tools/converters/official_repvgg_to_zcls_repvgg.py
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
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj">arch</th>
    <th class="tg-uzvj">framework<br></th>
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
    <td class="tg-9wq8">repvgg_a0_train</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">72.009</td>
    <td class="tg-9wq8">90.389</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">34.748</td>
    <td class="tg-baqh">3.043</td>
    <td class="tg-baqh">0.031</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_a0_infer</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">72.007</td>
    <td class="tg-9wq8">90.389</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">31.698</td>
    <td class="tg-baqh">2.727</td>
    <td class="tg-baqh">0.017</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_a1_train</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">74.092</td>
    <td class="tg-9wq8">91.695</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">53.758</td>
    <td class="tg-baqh">5.277</td>
    <td class="tg-baqh">0.044</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_a1_infer</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">74.092</td>
    <td class="tg-9wq8">91.699</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">48.789</td>
    <td class="tg-baqh">4.733</td>
    <td class="tg-baqh">0.030</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_a2_train</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">76.286</td>
    <td class="tg-9wq8">93.034</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">107.615</td>
    <td class="tg-baqh">11.403</td>
    <td class="tg-baqh">0.086</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_a2_infer</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">76.284</td>
    <td class="tg-9wq8">93.036</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">97.275</td>
    <td class="tg-baqh">10.240</td>
    <td class="tg-baqh">0.054</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_b0_train</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">74.934</td>
    <td class="tg-9wq8">92.312</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">60.341</td>
    <td class="tg-baqh">6.827</td>
    <td class="tg-baqh">0.059</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_b0_infer</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">74.944</td>
    <td class="tg-9wq8">92.312</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">54.699</td>
    <td class="tg-baqh">6.121</td>
    <td class="tg-baqh">0.041</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_b1_train</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">78.127</td>
    <td class="tg-9wq8">94.182</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">219.021</td>
    <td class="tg-baqh">26.314</td>
    <td class="tg-baqh">0.187</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_b1_infer</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">78.121</td>
    <td class="tg-9wq8">94.182</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">197.714</td>
    <td class="tg-baqh">23.642</td>
    <td class="tg-baqh">0.126</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_b1g2_train</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">77.803</td>
    <td class="tg-9wq8">93.866</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">174.646</td>
    <td class="tg-baqh">19.634</td>
    <td class="tg-baqh">0.155</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_b1g2_infer</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">77.805</td>
    <td class="tg-9wq8">93.864</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">157.776</td>
    <td class="tg-baqh">17.630</td>
    <td class="tg-baqh">0.107</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_b1g4_train</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">77.579</td>
    <td class="tg-9wq8">93.726</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">152.458</td>
    <td class="tg-baqh">16.295</td>
    <td class="tg-baqh">0.152</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_b1g4_infer</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">77.579</td>
    <td class="tg-9wq8">93.726</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">137.808</td>
    <td class="tg-baqh">14.625</td>
    <td class="tg-baqh">0.114</td>
  </tr>
  <tr>
    <td class="tg-9wq8">repvgg_b2_train</td>
    <td class="tg-9wq8">zcls</td>
    <td class="tg-9wq8">78.789</td>
    <td class="tg-9wq8">94.326</td>
    <td class="tg-9wq8">224x224</td>
    <td class="tg-9wq8">imagenet</td>
    <td class="tg-baqh">339.593</td>
    <td class="tg-baqh">40.907</td>
    <td class="tg-baqh">0.282</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_b2_infer</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">78.787</td>
    <td class="tg-nrix">94.324</td>
    <td class="tg-nrix">224x224</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">306.378</td>
    <td class="tg-baqh">36.766</td>
    <td class="tg-baqh">0.230</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_b2g4_train</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">79.491</td>
    <td class="tg-nrix">94.768</td>
    <td class="tg-nrix">224x224</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">235.590</td>
    <td class="tg-baqh">25.252</td>
    <td class="tg-baqh">0.192</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_b2g4_infer</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">79.495</td>
    <td class="tg-nrix">94.768</td>
    <td class="tg-nrix">224x224</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">212.774</td>
    <td class="tg-baqh">22.677</td>
    <td class="tg-baqh">0.141</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_b3_train</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">80.436</td>
    <td class="tg-nrix">95.278</td>
    <td class="tg-nrix">224x224</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">469.533</td>
    <td class="tg-baqh">58.324</td>
    <td class="tg-baqh">0.373</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_b3_infer</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">80.442</td>
    <td class="tg-nrix">95.280</td>
    <td class="tg-nrix">224x224</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">423.282</td>
    <td class="tg-baqh">52.433</td>
    <td class="tg-baqh">0.309</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_b3g4_train</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">80.070</td>
    <td class="tg-nrix">95.154</td>
    <td class="tg-nrix">224x224</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">319.767</td>
    <td class="tg-baqh">35.781</td>
    <td class="tg-baqh">0.270</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_b3g4_infer</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">80.072</td>
    <td class="tg-nrix">95.156</td>
    <td class="tg-nrix">224x224</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">288.493</td>
    <td class="tg-baqh">32.144</td>
    <td class="tg-baqh">0.203</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_b3g4_train</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">81.174</td>
    <td class="tg-nrix">95.811</td>
    <td class="tg-nrix">320x320</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">319.767</td>
    <td class="tg-baqh">73.018</td>
    <td class="tg-baqh">0.486</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_b3g4_infer</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">81.180</td>
    <td class="tg-nrix">95.811</td>
    <td class="tg-nrix">320x320</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">288.493</td>
    <td class="tg-baqh">65.595</td>
    <td class="tg-baqh">0.368</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_d2se_train</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">81.748</td>
    <td class="tg-nrix">95.811</td>
    <td class="tg-nrix">224x224</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">508.632</td>
    <td class="tg-baqh">73.107</td>
    <td class="tg-baqh">0.521</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_d2se_infer</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">81.752</td>
    <td class="tg-nrix">95.813</td>
    <td class="tg-nrix">224x224</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">459.242</td>
    <td class="tg-baqh">65.705</td>
    <td class="tg-baqh">0.413</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_d2se_train</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">83.555</td>
    <td class="tg-nrix">96.657</td>
    <td class="tg-nrix">320x320<br></td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">508.632</td>
    <td class="tg-baqh">149.188</td>
    <td class="tg-baqh">0.924</td>
  </tr>
  <tr>
    <td class="tg-nrix">repvgg_d2se_infer</td>
    <td class="tg-nrix">zcls</td>
    <td class="tg-nrix">83.559</td>
    <td class="tg-nrix">96.657</td>
    <td class="tg-nrix">320x320</td>
    <td class="tg-nrix">imagenet</td>
    <td class="tg-baqh">459.242</td>
    <td class="tg-baqh">134.082</td>
    <td class="tg-baqh">0.754</td>
  </tr>
</tbody>
</table>

*CPU_INFO: Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz*