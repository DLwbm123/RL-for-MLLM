# MedEvidence pilot-3 实际训练

每组固定256步；checkpoint0/64/128/256，最终固定256。COV仅L反传，同时同图A仅no_grad诊断；COV-A为L+1*A，患者平均，不除2。训练曲线和实际forward/vision/backward/剪裁计数见JSON。

```json
{
  "COV": {
    "status": "completed_diagnostic",
    "steps": 256,
    "planned_steps": 256,
    "alpha": 0.0,
    "total_L_exposures": 1024,
    "runtime_seconds": 759.1311531066895,
    "optimizer_reset": true,
    "test_pixels_read": 0,
    "patient_exposure_histogram": {
      "8": 15,
      "2": 256,
      "7": 56
    }
  },
  "COV-A": {
    "status": "completed_diagnostic",
    "steps": 256,
    "planned_steps": 256,
    "alpha": 1.0,
    "total_L_exposures": 1024,
    "runtime_seconds": 959.5040936470032,
    "optimizer_reset": true,
    "test_pixels_read": 0,
    "patient_exposure_histogram": {
      "8": 15,
      "2": 256,
      "7": 56
    }
  }
}
```
