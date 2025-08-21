## Lightweight Rainforest Gunshot Detection & Sensor Integration
### Towards Effective Real-Time Poaching & Gunshot Detection

---

Datasets:
1. Training & Validation: Belizean dataset collected by Katsis et al. (2022)
https://data.mendeley.com/datasets/x48cwz364j/3
2. Testing: Vietnamese dataset collected by Thinh Ten Vu et al. (2024)
https://github.com/DenaJGibbon/Vietnam-Gunshots

Usage:
1. Serialize the existing model or train your own using `serialize.py`
2. Train or evalute your own model using `train.py` and `evaluate.py`
3. Test the SAIL sensor integration function using `sail.py`