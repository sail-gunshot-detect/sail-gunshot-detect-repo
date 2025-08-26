## Lightweight Rainforest Gunshot Detection & Sensor Integration
_Towards Effective Real-Time Poaching & Gunshot Detection_

---

Datasets:
1. Training & Validation: Belizean dataset collected by Katsis et al. (2022)
https://data.mendeley.com/datasets/x48cwz364j/3
2. Testing: Vietnamese dataset collected by Thinh Ten Vu et al. (2024)
https://github.com/DenaJGibbon/Vietnam-Gunshots

Usage:
1. Serialize the existing model or train your own using `serialize.py`
2. (Optionally) Train your own model using `train.py`
3. Evaluate the existing model on your own datasets using `evaluate.py`
3. Use and test the SAIL sensor integration function using `sail.py`