
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
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">arch</th>
    <th class="tg-c3ow">framework<br></th>
    <th class="tg-c3ow">top1</th>
    <th class="tg-c3ow">top5</th>
    <th class="tg-c3ow">input_size</th>
    <th class="tg-c3ow">dataset</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">repvgg_a0_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">72.009</td>
    <td class="tg-c3ow">90.389</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_a0_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">72.007</td>
    <td class="tg-c3ow">90.389</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_a1_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">74.092</td>
    <td class="tg-c3ow">91.695</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_a1_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">74.092</td>
    <td class="tg-c3ow">91.699</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_a2_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">76.286</td>
    <td class="tg-c3ow">93.034</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_a2_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">76.284</td>
    <td class="tg-c3ow">93.036</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b0_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">74.934</td>
    <td class="tg-c3ow">92.312</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b0_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">74.944</td>
    <td class="tg-c3ow">92.312</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b1_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">78.127</td>
    <td class="tg-c3ow">94.182</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b1_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">78.121</td>
    <td class="tg-c3ow">94.182</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b1g2_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">77.803</td>
    <td class="tg-c3ow">93.866</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b1g2_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">77.805</td>
    <td class="tg-c3ow">93.864</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b1g4_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">77.579</td>
    <td class="tg-c3ow">93.726</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b1g4_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">77.579</td>
    <td class="tg-c3ow">93.726</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b2_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">78.789</td>
    <td class="tg-c3ow">94.326</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b2_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">78.787</td>
    <td class="tg-c3ow">94.324</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b2g4_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">79.491</td>
    <td class="tg-c3ow">94.768</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b2g4_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">79.495</td>
    <td class="tg-c3ow">94.768</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b3_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">80.436</td>
    <td class="tg-c3ow">95.278</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b3_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">80.442</td>
    <td class="tg-c3ow">95.280</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b3g4_train</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">80.070</td>
    <td class="tg-c3ow">95.154</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-c3ow">repvgg_b3g4_infer</td>
    <td class="tg-c3ow">zcls</td>
    <td class="tg-c3ow">80.072</td>
    <td class="tg-c3ow">95.156</td>
    <td class="tg-c3ow">224x224</td>
    <td class="tg-c3ow">imagenet</td>
  </tr>
  <tr>
    <td class="tg-baqh">repvgg_b3g4_train</td>
    <td class="tg-baqh">zcls</td>
    <td class="tg-baqh">81.174</td>
    <td class="tg-baqh">95.811</td>
    <td class="tg-baqh">320x320</td>
    <td class="tg-baqh">imagenet</td>
  </tr>
  <tr>
    <td class="tg-baqh">repvgg_b3g4_infer</td>
    <td class="tg-baqh">zcls</td>
    <td class="tg-baqh">81.180</td>
    <td class="tg-baqh">95.811</td>
    <td class="tg-baqh">320x320</td>
    <td class="tg-baqh">imagenet</td>
  </tr>
  <tr>
    <td class="tg-baqh">repvgg_d2se_train</td>
    <td class="tg-baqh">zcls</td>
    <td class="tg-baqh">81.748</td>
    <td class="tg-baqh">95.811</td>
    <td class="tg-baqh">224x224</td>
    <td class="tg-baqh">imagenet</td>
  </tr>
  <tr>
    <td class="tg-baqh">repvgg_d2se_infer</td>
    <td class="tg-baqh">zcls</td>
    <td class="tg-baqh">81.752</td>
    <td class="tg-baqh">95.813</td>
    <td class="tg-baqh">224x224</td>
    <td class="tg-baqh">imagenet</td>
  </tr>
  <tr>
    <td class="tg-baqh">repvgg_d2se_train</td>
    <td class="tg-baqh">zcls</td>
    <td class="tg-baqh">83.555</td>
    <td class="tg-baqh">96.657</td>
    <td class="tg-baqh">320x320<br></td>
    <td class="tg-baqh">imagenet</td>
  </tr>
  <tr>
    <td class="tg-baqh">repvgg_d2se_infer</td>
    <td class="tg-baqh">zcls</td>
    <td class="tg-baqh">83.559</td>
    <td class="tg-baqh">96.657</td>
    <td class="tg-baqh">320x320</td>
    <td class="tg-baqh">imagenet</td>
  </tr>
</tbody>
</table>