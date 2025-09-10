## Lightweight Rainforest Gunshot Detection Model & Sensor Integration Function
_Fighting Poaching Through Targeted Deep Learning and Sensor Integration_

---

Datasets:
1. Training & Validation: Belizean dataset collected by Katsis et al. (2022)
https://data.mendeley.com/datasets/x48cwz364j/3
2. Testing: Vietnamese dataset collected by Thinh Ten Vu et al. (2024)
https://github.com/DenaJGibbon/Vietnam-Gunshots

Usage:

The `config.yaml` file contains configuration parameters for audio processing, model training, and evaluation. It includes settings for sample rate, mel-spectrogram parameters, training hyperparameters (batch size, epochs, learning rate), and evaluation thresholds. These parameters can be overridden by command-line arguments in the respective scripts.

1. Serialize the existing model or train your own using `serialize.py`
   Run: `python serialize.py --model_path path/to/model.keras --output_path path/to/model.tflite`
   This converts a trained Keras model (.keras file) to TensorFlow Lite format (.tflite file) for efficient deployment on edge devices. The script loads the model, wraps it in a tf.function for proper input signature, optimizes it, and saves the converted model.

2. (Optionally) Train your own model using `train.py`
   Run: `python train.py --train_neg_path /path/to/training/negative/samples --train_pos_path /path/to/training/positive/samples --val_neg_path /path/to/validation/negative/samples --val_pos_path /path/to/validation/positive/samples`
   This trains a convolutional neural network for gunshot detection using audio data. The model architecture includes separable convolutional layers, 1D convolutions, GRU layers, and dense layers. It processes audio files by converting them to mel-spectrograms and applies data augmentation during training. Optional arguments include --config (default: config.yaml), --epochs, --batch_size, --model_name, --model_save_dir, and --seed.

3. Evaluate the existing model on your own datasets using `evaluate.py`
   Run: `python evaluate.py --model_path path/to/model.keras --neg_path /path/to/negative/test/samples --pos_path /path/to/positive/test/samples --output_dir ./results`
   This evaluates a trained model's performance on test datasets. It loads the model, processes audio files, generates predictions, calculates metrics (accuracy, precision, recall, F1), and creates precision-recall curves. Optional arguments include --config (default: config.yaml), --threshold, --batch_size, --save_csv (to save misclassified files), --save_pr_curve (to save PR curve plot), and --model_name.

4. Use and test the SAIL sensor integration function using `sail.py`
   Run: `python sail.py --preds "[0.1, 0.2, 0.3, 0.4]" --start 1`
   This applies the SAIL (Sensor Array Integration Layer) algorithm to combine multiple predictions from different sensors. The algorithm mathematically combines predictions using:

   ```
   inner(p₁,...,pₙ) = 1 - Σ_{1≤i<j≤n} (1-pᵢ)(1-pⱼ) / Σ_{k=1}^n p_k
   ```

   then applies a sigmoid transformation. Predictions can be provided as Python list format "[0.1, 0.2, 0.3]" or comma-separated values "0.1, 0.2, 0.3". The --start parameter (default: 1) specifies which column index to begin using predictions from.