# Consistency Metrics Summary

Visual similarity between alpha and bravo generated frames (turnToLookEval expects similar scenery, turnToLookOppositeEval expects different scenery).

## turnToLookEval


| Model                | N   | LPIPS (mean +/- std) | CLIP cos sim (mean +/- std) | DINOv2 cos sim (mean +/- std) | DINOv3 cos sim (mean +/- std) |
| -------------------- | --- | -------------------- | --------------------------- | ----------------------------- | ----------------------------- |
| causvid_dmd          | 32  | 0.4261 +/- 0.0668    | 0.8979 +/- 0.0272           | 0.8267 +/- 0.0459             | 0.9310 +/- 0.0179             |
| causvid_regression   | 32  | 0.3700 +/- 0.0863    | 0.9418 +/- 0.0253           | 0.8407 +/- 0.0747             | 0.9340 +/- 0.0318             |
| concat_c             | 32  | 0.3919 +/- 0.0809    | 0.9409 +/- 0.0141           | 0.8305 +/- 0.0485             | 0.9191 +/- 0.0201             |
| flagship             | 32  | 0.3291 +/- 0.0816    | 0.9572 +/- 0.0165           | 0.8883 +/- 0.0411             | 0.9618 +/- 0.0126             |
| from_scratch         | 32  | 0.3591 +/- 0.0915    | 0.9406 +/- 0.0201           | 0.8794 +/- 0.0533             | 0.9546 +/- 0.0258             |
| no_kv_cache_backprop | 32  | 0.3238 +/- 0.0794    | 0.9574 +/- 0.0152           | 0.8822 +/- 0.0433             | 0.9628 +/- 0.0138             |


## turnToLookOppositeEval


| Model                | N   | LPIPS (mean +/- std) | CLIP cos sim (mean +/- std) | DINOv2 cos sim (mean +/- std) | DINOv3 cos sim (mean +/- std) |
| -------------------- | --- | -------------------- | --------------------------- | ----------------------------- | ----------------------------- |
| causvid_dmd          | 32  | 0.4361 +/- 0.0804    | 0.8944 +/- 0.0296           | 0.8425 +/- 0.0602             | 0.9461 +/- 0.0214             |
| causvid_regression   | 32  | 0.3984 +/- 0.0661    | 0.9311 +/- 0.0287           | 0.8115 +/- 0.0719             | 0.9320 +/- 0.0348             |
| concat_c             | 32  | 0.3733 +/- 0.0588    | 0.9453 +/- 0.0145           | 0.8209 +/- 0.0647             | 0.9163 +/- 0.0194             |
| flagship             | 32  | 0.3635 +/- 0.0626    | 0.9343 +/- 0.0210           | 0.8390 +/- 0.0568             | 0.9473 +/- 0.0178             |
| from_scratch         | 32  | 0.3631 +/- 0.0766    | 0.9395 +/- 0.0239           | 0.8731 +/- 0.0535             | 0.9553 +/- 0.0197             |
| no_kv_cache_backprop | 32  | 0.3556 +/- 0.0554    | 0.9508 +/- 0.0164           | 0.8306 +/- 0.0637             | 0.9503 +/- 0.0181             |


## Combined (averaged across both eval types)


| Model                | LPIPS (mean) | CLIP cos sim (mean) | DINOv2 cos sim (mean) | DINOv3 cos sim (mean) |
| -------------------- | ------------ | ------------------- | --------------------- | --------------------- |
| causvid_dmd          | 0.4311       | 0.8962              | 0.8346                | 0.9386                |
| causvid_regression   | 0.3842       | 0.9364              | 0.8261                | 0.9330                |
| concat_c             | 0.3826       | 0.9431              | 0.8257                | 0.9177                |
| flagship             | 0.3463       | 0.9458              | 0.8636                | 0.9545                |
| from_scratch         | 0.3611       | 0.9401              | 0.8763                | 0.9550                |
| no_kv_cache_backprop | 0.3397       | 0.9541              | 0.8564                | 0.9566                |


