"""
CDC Simulation Script for Gunshot Detection with SAIL Integration

This script performs acoustic propagation simulations to test SAIL (Sensor Array Integration Layer), 
which combines multiple sensor predictions.

Usage:
    python simulation.py --model_path /path/to/model.keras --run_name my_experiment --dataset small_test --th 0.5 --runs 10 --seed 42 --run_note "Experiment description"
"""

import argparse
import json
import logging
import math
import os
import random
import time
import warnings
from scipy.stats import wilcoxon, ttest_rel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import tensorflow as tf
import tensorflow_io as tfio

# Configure logging and suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Global constants
DESIRED_SR = 8000
DESIRED_SR_TENSOR = tf.constant(DESIRED_SR, dtype=tf.int64)
MEL_SPEC_LENGTH = 31872
FRAME_LENGTH = 256
FRAME_STEP = 128
MEL_BINS = 64
FMAX = 4000
FMIN = 100
AUTOTUNE = tf.data.AUTOTUNE


def load_dataset(key="small_test"):
    """
    Load dataset from predefined paths.

    Args:
        key (str): Dataset identifier ("small_test", "big_test", or "b_val")

    Returns:
        list: List of tuples (filepath, label) for each audio file
    """
    if key == "big_test":
        big_test_POS = "C:/Users/foxir/OneDrive/Desktop/Smart Rock/gunshot-detection/model/working directory/vietnam-data/testdatacombined_clips/test/gunshot"
        big_test_NEG = "C:/Users/foxir/OneDrive/Desktop/Smart Rock/gunshot-detection/model/working directory/vietnam-data/testdatacombined_clips/test/noise"

        big_test_pos = [os.path.join(big_test_POS, f) for f in os.listdir(big_test_POS) if f.endswith(".wav")]
        big_test_neg = [os.path.join(big_test_NEG, f) for f in os.listdir(big_test_NEG) if f.endswith(".wav")]

        big_test_pos = [(f, 1) for f in big_test_pos]
        big_test_neg = [(f, 0) for f in big_test_neg]
        big_test = big_test_pos + big_test_neg
        print(f"Dataset length: {len(big_test)}")
        return big_test

    if key == "small_test":
        small_test_POS = "C:/Users/foxir/OneDrive/Desktop/Smart Rock/gunshot-detection/model/working directory/vietnam-data/imagesvietnamunbalanced_clips/test/gunshot"
        small_test_NEG = "C:/Users/foxir/OneDrive/Desktop/Smart Rock/gunshot-detection/model/working directory/vietnam-data/imagesvietnamunbalanced_clips/test/noise"

        small_test_pos = [os.path.join(small_test_POS, f) for f in os.listdir(small_test_POS) if f.endswith(".wav")]
        small_test_neg = [os.path.join(small_test_NEG, f) for f in os.listdir(small_test_NEG) if f.endswith(".wav")]

        small_test_pos = [(f, 1) for f in small_test_pos]
        small_test_neg = [(f, 0) for f in small_test_neg]
        small_test = small_test_pos + small_test_neg
        print(f"Dataset length: {len(small_test)}")
        return small_test

    elif key == "b_val":
        b_val_POS = "C:/Users/foxir/OneDrive/Desktop/Smart Rock/gunshot-detection/model/working directory/belize-data/Validation data/Gunshot"
        b_val_NEG = "C:/Users/foxir/OneDrive/Desktop/Smart Rock/gunshot-detection/model/working directory/belize-data/Validation data/Background"

        b_val_pos = [os.path.join(b_val_POS, f) for f in os.listdir(b_val_POS) if f.endswith(".wav")]
        b_val_neg = [os.path.join(b_val_NEG, f) for f in os.listdir(b_val_NEG) if f.endswith(".wav")]

        b_val_pos = [(f, 1) for f in b_val_pos]
        b_val_neg = [(f, 0) for f in b_val_neg]
        b_val = b_val_pos + b_val_neg
        print(f"Dataset length: {len(b_val)}")
        return b_val


@tf.function(experimental_relax_shapes=True)
def load_wav_8k_mono_tf_nolabel(filename):
    """
    Load and resample WAV file to 8kHz mono.

    Args:
        filename: Path to WAV file

    Returns:
        tf.Tensor: Resampled audio waveform
    """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)

    # Resample wav if sample rate is not as desired
    wav = tf.cond(tf.not_equal(sample_rate, DESIRED_SR_TENSOR),
                  lambda: tfio.audio.resample(wav, rate_in=sample_rate, rate_out=DESIRED_SR_TENSOR),
                  lambda: wav)
    wav = tf.squeeze(wav, axis=-1)
    return wav


@tf.function(experimental_relax_shapes=True)
def preprocess_mel_db_tf_nolabel(wav):
    """
    Convert waveform to mel-spectrogram in dB scale.

    Args:
        wav: Input audio waveform

    Returns:
        tf.Tensor: Mel-spectrogram in dB scale
    """
    # Pad wav if too short and cut it if too long
    wav_len = tf.shape(wav)[0]
    wav = tf.cond(wav_len > MEL_SPEC_LENGTH,
                  lambda: wav[:MEL_SPEC_LENGTH],
                  lambda: tf.cond(wav_len < MEL_SPEC_LENGTH,
                                  lambda: tf.concat([wav, tf.zeros([MEL_SPEC_LENGTH - wav_len], dtype=tf.float32)], axis=0),
                                  lambda: wav))

    # Create the spectrogram
    spectrogram = tfio.audio.spectrogram(wav, FRAME_LENGTH, FRAME_LENGTH, FRAME_STEP)
    mel_spectrogram = tfio.audio.melscale(spectrogram, DESIRED_SR, MEL_BINS, FMIN, FMAX)
    dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)

    return dbscale_mel_spectrogram


@tf.function(experimental_relax_shapes=True)
def expand_dim_tf_nolabel(spec):
    """Add channel dimension for Conv2D input."""
    spec = tf.expand_dims(spec, axis=-1)
    return spec


# Augmentation functions

@tf.function(experimental_relax_shapes=True)
def rand_uniform(low, high):
    """Return a random float in the bounds."""
    return tf.random.uniform([], low, high)


@tf.function(experimental_relax_shapes=True)
def apply_with_p(p, fn, x):
    """Apply function fn(x) with probability p, else return x."""
    return tf.cond(tf.less(tf.random.uniform([], 0, 1), p),
                   lambda: fn(x),
                   lambda: x)


@tf.function(experimental_relax_shapes=True)
def add_gaussian_snr(wav, min_snr=16.0, max_snr=32.0):
    """Add Gaussian noise with random SNR."""
    rms_signal = tf.sqrt(tf.reduce_mean(wav**2))
    if isinstance(min_snr, tf.Tensor) and isinstance(max_snr, tf.Tensor):
        min_snr_tensor = min_snr
        max_snr_tensor = max_snr
    else:
        min_snr_tensor = tf.constant(min_snr, dtype=tf.float32)
        max_snr_tensor = tf.constant(max_snr, dtype=tf.float32)

    snr_db = tf.cond(
        tf.equal(min_snr_tensor, max_snr_tensor),
        lambda: min_snr_tensor,
        lambda: rand_uniform(min_snr_tensor, max_snr_tensor)
    )
    snr = 10.0**(snr_db / 20.0)
    noise_rms = rms_signal / snr
    noise = tf.random.normal(tf.shape(wav), dtype=wav.dtype) * noise_rms
    return wav + noise


@tf.function(experimental_relax_shapes=True)
def gain(wav, min_db=-8.0, max_db=24.0):
    """Apply random gain amplification."""
    if isinstance(min_db, tf.Tensor) and isinstance(max_db, tf.Tensor):
        min_db_tensor = min_db
        max_db_tensor = max_db
    else:
        min_db_tensor = tf.constant(min_db, dtype=tf.float32)
        max_db_tensor = tf.constant(max_db, dtype=tf.float32)

    db = tf.cond(
        tf.equal(min_db_tensor, max_db_tensor),
        lambda: min_db_tensor,
        lambda: rand_uniform(min_db_tensor, max_db_tensor)
    )
    factor = 10.0**(db / 20.0)
    return wav * factor


def time_stretch(wav, sr=DESIRED_SR, min_rate=0.9, max_rate=1.112):
    """Apply time stretching via resampling."""
    length = tf.shape(wav)[0]

    # Pick random speed factor
    rate = rand_uniform(min_rate, max_rate)

    # Compute new output sample rate
    sr_out = tf.cast(tf.cast(sr, tf.float32) * rate, tf.int64)

    # Resample
    stretched = tfio.audio.resample(wav, rate_in=sr, rate_out=sr_out)

    # Pad or trim back to original length
    stretched_len = tf.shape(stretched)[0]
    return tf.cond(
        stretched_len < length,
        lambda: tf.pad(stretched, [[0, length - stretched_len]]),
        lambda: stretched[:length]
    )


@tf.function(experimental_relax_shapes=True)
def shift(wav, min_shift=-4.0, max_shift=4.0, sample_rate=DESIRED_SR):
    """Apply time shift by rolling the signal."""
    if isinstance(min_shift, tf.Tensor) and isinstance(max_shift, tf.Tensor):
        min_shift_tensor = min_shift
        max_shift_tensor = max_shift
    else:
        min_shift_tensor = tf.constant(min_shift, dtype=tf.float32)
        max_shift_tensor = tf.constant(max_shift, dtype=tf.float32)

    shift_seconds = tf.cond(
        tf.equal(min_shift_tensor, max_shift_tensor),
        lambda: min_shift_tensor,
        lambda: rand_uniform(min_shift_tensor, max_shift_tensor)
    )
    shift_amt = tf.cast(shift_seconds * tf.cast(sample_rate, tf.float32), tf.int32)

    # Pad with zeros instead of wrapping around
    length = tf.shape(wav)[0]
    return tf.cond(
        shift_amt >= 0,
        # Positive shift: pad zeros at beginning, trim end
        lambda: tf.concat([tf.zeros([shift_amt], dtype=wav.dtype), wav[:-shift_amt]], axis=0),
        # Negative shift: pad zeros at end, trim beginning
        lambda: tf.concat([wav[-shift_amt:], tf.zeros([-shift_amt], dtype=wav.dtype)], axis=0)
    )


def make_4_variants(wav):
    """
    Create 4 variants of input waveform for CDC simulation:
    - Original waveform
    - 3 augmented variants with randomized noise, gain, and time shifts

    Args:
        wav: Input waveform tensor

    Returns:
        Tensor of shape (4, length) containing original + 3 augmented variants
    """
    # Original waveform
    wav0 = wav

    # Base parameters for augmentation
    base_noise = tf.convert_to_tensor(40, dtype=tf.float32)
    base_gain = tf.convert_to_tensor(0, dtype=tf.float32)
    base_time = tf.convert_to_tensor(1/50, dtype=tf.float32)

    # Create 3 augmented variants with randomized parameters
    val_1 = tf.random.uniform([], 0, 30)
    wav_1 = shift(
        add_gaussian_snr(
            gain(wav0, base_gain-val_1, base_gain-val_1),
            base_noise-val_1, base_noise-val_1
        ),
        base_gain+tf.math.multiply(base_time, val_1),
        base_gain+tf.math.multiply(base_time, val_1)
    )

    val_2 = tf.random.uniform([], 0, 30)
    wav_2 = shift(
        add_gaussian_snr(
            gain(wav0, base_gain-val_2, base_gain-val_2),
            base_noise-val_2, base_noise-val_2
        ),
        base_gain+tf.math.multiply(base_time, val_2),
        base_gain+tf.math.multiply(base_time, val_2)
    )

    val_3 = tf.random.uniform([], 0, 30)
    wav_3 = shift(
        add_gaussian_snr(
            gain(wav0, base_gain-val_3, base_gain-val_3),
            base_noise-val_3, base_noise-val_3
        ),
        base_gain+tf.math.multiply(base_time, val_3),
        base_gain+tf.math.multiply(base_time, val_3)
    )

    return tf.stack([wav0, wav_1, wav_2, wav_3], axis=0)


@tf.function(experimental_relax_shapes=True)
def wavs_to_stacked_specs(wav_stack):
    """
    Convert stack of waveforms to stacked mel spectrograms.

    Args:
        wav_stack: Tensor of shape (4, length) containing waveforms

    Returns:
        Tensor of shape (4, H, W, 1) containing mel spectrograms
    """
    # Apply preprocess_mel_db_tf_nolabel to each variant
    specs = tf.map_fn(lambda w: preprocess_mel_db_tf_nolabel(w), wav_stack, dtype=tf.float32)
    # Add channel dimension
    specs = tf.expand_dims(specs, axis=-1)
    return specs


def build_dataset_stack(ds_list, batch_size=16, cache_raw=False):
    """
    Build a tf.data.Dataset pipeline for CDC simulation.

    Args:
        ds_list: List of tuples (filepath, label)
        batch_size: Batch size for processing
        cache_raw: Whether to cache raw waveforms in memory

    Returns:
        tf.data.Dataset yielding batches of (spec_batch, labels_batch, paths_batch)
    """
    filepaths = [p for p, l in ds_list]
    labels = [l for p, l in ds_list]
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # Load wavs in parallel
    def _load(path, label):
        wav = load_wav_8k_mono_tf_nolabel(path)
        return wav, label, path

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    # Optionally cache raw waveforms
    if cache_raw:
        ds = ds.cache()

    # Create randomized 4 variants and convert to stacked specs
    def _to_stack_spec(wav, label, path):
        wav_stack = make_4_variants(wav)
        spec_stack = wavs_to_stacked_specs(wav_stack)
        return spec_stack, label, path

    ds = ds.map(_to_stack_spec, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# SAIL functions

def combine_preds_vec(p1, p2, p3):
    """Vectorized SAIL combination for multiple predictions."""
    p1 = np.asarray(p1); p2 = np.asarray(p2); p3 = np.asarray(p3)
    numer = (1-p1)*(1-p2) + (1-p2)*(1-p3) + (1-p1)*(1-p3)
    denom = (p1 + p2 + p3) + 1e-12
    combined = 1 - (numer / denom)
    return combined


def sigmoid_pred_vec(pf):
    """Numerically stable sigmoid transformation."""
    x = 10.0*(pf - 0.5)
    x = np.clip(x, -10, 10)
    return np.where(x >= 0,
                   1.0 / (1.0 + np.exp(-x)),
                   np.exp(x) / (1.0 + np.exp(x)))


def sail_preds_vec(preds_4):
    """
    Apply SAIL combination to 4 predictions.

    Args:
        preds_4: Array of shape (B, 4) with predictions

    Returns:
        Array of shape (B,) with SAIL combined predictions
    """
    p1 = preds_4[:,1]; p2 = preds_4[:,2]; p3 = preds_4[:,3]
    combined = combine_preds_vec(p1, p2, p3)
    return sigmoid_pred_vec(combined)


def sensor_score_with_target_vec(preds_4, target, lam=1.0, eps=1e-9):
    """
    Apply sensor integration function with target threshold.

    Args:
        preds_4: Array of shape (B, 4) with predictions
        target: Target threshold value
        lam: Weight parameter
        eps: Small epsilon for numerical stability

    Returns:
        Array of shape (B,) with integrated predictions
    """
    t = float(target)
    xs = np.clip(preds_4, 0.0, 1.0)
    n = np.maximum(0.0, (t - xs) / t)
    p = np.maximum(0.0, (xs - t) / (1.0 - t))

    # Pairwise sums
    N = 0.5 * ((np.sum(n, axis=1)**2) - np.sum(n**2, axis=1))
    P = 0.5 * ((np.sum(p, axis=1)**2) - np.sum(p**2, axis=1))
    A = np.sum(xs, axis=1)
    score = 1.0 - (N / (A + eps)) + lam * (P / (A + eps))
    return sigmoid_pred_vec(score)


def combine_preds(p1, p2, p3):
    """Scalar SAIL combination."""
    combined_pred = (1-(((1-p1)*(1-p2)+(1-p2)*(1-p3)+(1-p1)*(1-p3))/(p1+p2+p3+1e-06)))
    return combined_pred


def sigmoid_pred(pf):
    """Scalar sigmoid transformation."""
    x = (10)*(pf-0.5)
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def sail_preds(p1, p2, p3):
    """Apply SAIL to three scalar predictions."""
    combined = combine_preds(p1, p2, p3)
    final = sigmoid_pred(combined)
    return final


def sensor_score_with_target(xs, target, lam=1.0, eps=1e-9):
    """
    Apply sensor integration function to scalar predictions.

    Args:
        xs: Iterable of probabilities in [0,1]
        target: Target threshold in (0,1)
        lam: Weight on consistent above-target boost
        eps: Small epsilon for numerical stability

    Returns:
        float: Integrated score
    """
    xs = list(xs)
    t = float(target)
    if not (0.0 < t < 1.0):
        raise ValueError("target must be in (0,1)")

    # Scaled negatives and positives
    n = [max(0.0, (t - x) / t) for x in xs]
    p = [max(0.0, (x - t) / (1.0 - t)) for x in xs]

    # Pairwise sums
    N = 0.0
    P = 0.0
    nlen = len(xs)
    for i in range(nlen):
        for j in range(i+1, nlen):
            N += n[i] * n[j]
            P += p[i] * p[j]

    A = sum(xs)
    return 1.0 - (N / (A + eps)) + lam * (P / (A + eps))


# Result generation functions

def create_results_fast(ds_list, loaded_model, th, batch_size=16, cache_raw=False):
    """
    Generate results using fast batched processing.

    Args:
        ds_list: List of tuples (filepath, label)
        loaded_model: TensorFlow model
        th: Threshold for sensor integration
        batch_size: Processing batch size
        cache_raw: Whether to cache raw waveforms

    Returns:
        List of results per file: [filepath, label, p0, p1, p2, p3, sail_pred, fancy_pred]
    """
    ds = build_dataset_stack(ds_list, batch_size=batch_size, cache_raw=cache_raw)
    results = []
    total = 0

    for spec_batch, labels_batch, paths_batch in ds:
        # Reshape to (B*4, H, W, 1) for single model call
        b = spec_batch.shape[0]
        spec_reshaped = tf.reshape(spec_batch, (b*4, spec_batch.shape[2], spec_batch.shape[3], 1))

        # Run model once on the whole reshaped batch
        preds = loaded_model.predict(spec_reshaped, batch_size=max(8, batch_size*4), verbose=0)
        preds = np.asarray(preds).reshape(b, 4)[:, 0:4]

        # Compute SAIL and fancy predictions vectorized
        sail_pred_batch = sail_preds_vec(preds)
        fancy_pred_batch = sensor_score_with_target_vec(preds, th)

        # Append per-file results
        for i in range(b):
            filepath = paths_batch[i].numpy().decode('utf-8')
            results.append([
                filepath, int(labels_batch[i].numpy()),
                float(preds[i,0]), float(preds[i,1]), float(preds[i,2]), float(preds[i,3]),
                float(sail_pred_batch[i]), float(fancy_pred_batch[i])
            ])
        total += b

    return results


def full_loop_fast(ds, loaded_model, th, n_runs):
    """
    Run multiple simulation runs with fast processing.

    Args:
        ds: Dataset list of tuples (filepath, label)
        loaded_model: TensorFlow model
        th: Threshold for sensor integration
        n_runs: Number of runs to perform

    Returns:
        List of results for each run
    """
    full_results = []
    for i in range(n_runs):
        full_results.append(create_results_fast(ds, loaded_model, th))
        print(f"Appended run {i}")
    return full_results


# Analysis utilities

def flatten_all_runs_to_df(all_runs):
    """
    Convert nested run results to a flattened DataFrame.

    Args:
        all_runs: List of runs, each containing list of file results

    Returns:
        pandas.DataFrame with columns: run, filepath, label, p0, p1, p2, p3, sail_pred, fancy_pred
    """
    rows = []
    for run_idx, run in enumerate(all_runs):
        for row in run:
            filepath = row[0]
            label = row[1]
            p0, p1, p2, p3 = row[2], row[3], row[4], row[5]
            sail = row[6]
            fancy = row[7]
            rows.append([run_idx, filepath, label, p0, p1, p2, p3, sail, fancy])

    df = pd.DataFrame(rows, columns=['run', 'filepath', 'label', 'p0','p1','p2','p3','sail_pred','fancy_pred'])
    return df


def calculate_metrics_manually(predictions, labels, threshold):
    """
    Calculate metrics using manual TP/TN/FP/FN computation.

    Args:
        predictions: Array of prediction scores
        labels: Array of true labels (0/1)
        threshold: Classification threshold

    Returns:
        dict: Dictionary with fpr, tpr, f1, precision, recall, tp, tn, fp, fn
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Convert predictions to binary using threshold
    predicted_labels = (predictions >= threshold).astype(int)

    # Calculate confusion matrix components
    tp = np.sum((labels == 1) & (predicted_labels == 1))
    tn = np.sum((labels == 0) & (predicted_labels == 0))
    fp = np.sum((labels == 0) & (predicted_labels == 1))
    fn = np.sum((labels == 1) & (predicted_labels == 0))

    # Calculate metrics with safe division
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'fpr': fpr,
        'tpr': tpr,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def calculate_metrics(results, threshold, run_note):
    """
    Calculate metrics for all runs using manual computation.

    Args:
        results: List of results per run
        threshold: Classification threshold
        run_note: Note to include in first run metrics

    Returns:
        dict: Metrics per run
    """
    metrics = {}
    for i, result_list in enumerate(results):
        labels = []
        inference = []
        sail_preds = []
        fancy_preds = []

        for result in result_list:
            labels.append(result[1])
            inference.append(result[2])
            sail_preds.append(result[-2])
            fancy_preds.append(result[-1])

        # Calculate metrics manually for each prediction type
        inf_metrics = calculate_metrics_manually(inference, labels, threshold)
        sail_metrics = calculate_metrics_manually(sail_preds, labels, threshold)
        fancy_metrics = calculate_metrics_manually(fancy_preds, labels, threshold)

        if i == 0:
            metrics[i] = {
                'run': i,
                'inf_fpr': inf_metrics['fpr'],
                'inf_tpr': inf_metrics['tpr'],
                'inf_f1': inf_metrics['f1'],
                'inf_prec': inf_metrics['precision'],
                'inf_rec': inf_metrics['recall'],
                'sail_fpr': sail_metrics['fpr'],
                'sail_tpr': sail_metrics['tpr'],
                'sail_f1': sail_metrics['f1'],
                'sail_prec': sail_metrics['precision'],
                'sail_rec': sail_metrics['recall'],
                'fancy_fpr': fancy_metrics['fpr'],
                'fancy_tpr': fancy_metrics['tpr'],
                'fancy_f1': fancy_metrics['f1'],
                'fancy_prec': fancy_metrics['precision'],
                'fancy_rec': fancy_metrics['recall'],
                'run_note': run_note
            }
        else:
            metrics[i] = {
                'run': i,
                'inf_fpr': inf_metrics['fpr'],
                'inf_tpr': inf_metrics['tpr'],
                'inf_f1': inf_metrics['f1'],
                'inf_prec': inf_metrics['precision'],
                'inf_rec': inf_metrics['recall'],
                'sail_fpr': sail_metrics['fpr'],
                'sail_tpr': sail_metrics['tpr'],
                'sail_f1': sail_metrics['f1'],
                'sail_prec': sail_metrics['precision'],
                'sail_rec': sail_metrics['recall'],
                'fancy_fpr': fancy_metrics['fpr'],
                'fancy_tpr': fancy_metrics['tpr'],
                'fancy_f1': fancy_metrics['f1'],
                'fancy_prec': fancy_metrics['precision'],
                'fancy_rec': fancy_metrics['recall']
            }
    return metrics


def get_csv(filepath, metrics):
    """Save metrics to CSV file."""
    df = pd.DataFrame.from_dict(metrics, orient='index')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def main():
    """Main function to run CDC simulation."""
    parser = argparse.ArgumentParser(description='CDC Simulation for Gunshot Detection with SAIL')
    parser.add_argument("--model_path", type=str,
                       default="C:/Users/foxir/OneDrive/Desktop/Smart Rock/gunshot-detection/model/working directory/new notebooks/one_shot_GRU_light_augment_seed_2980657396.keras",
                       help="Path to trained model")
    parser.add_argument("--run_name", type=str, required=True,
                       help="Name for this simulation run")
    parser.add_argument("--dataset", type=str, default="small_test",
                       choices=["small_test", "big_test", "b_val"],
                       help="Dataset to use for simulation")
    parser.add_argument("--th", type=float, default=0.5, required=True,
                       help="Threshold for sensor integration function")
    parser.add_argument("--runs", type=int, required=True,
                       help="Number of simulation runs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--run_note", type=str, required=True,
                       help="Note describing this run")

    args = parser.parse_args()

    def set_seeds(seed):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Setting seed: {args.seed}")
    set_seeds(args.seed)

    # Load model and dataset
    model = tf.keras.models.load_model(args.model_path)
    ds = load_dataset(args.dataset)

    print(f"Starting full loop for {args.runs} runs...")
    start_time = time.time()
    results = full_loop_fast(ds, model, args.th, args.runs)
    end_time = time.time()
    print(".1f"
    # Calculate and save metrics
    print("Calculating per-run metrics...")
    metrics = calculate_metrics(results, args.th, args.run_note)
    csv_metrics_path = os.path.join("cdc_sim_results", f"{args.run_name}_perrun_metrics.csv")
    get_csv(csv_metrics_path, metrics)
    print(f"Saved per-run metrics CSV -> {csv_metrics_path}")

    # Save flattened results for further analysis
    df_all = flatten_all_runs_to_df(results)
    flat_csv = os.path.join("cdc_sim_results", f"{args.run_name}_all_runs_flat.csv")
    os.makedirs(os.path.dirname(flat_csv), exist_ok=True)
    df_all.to_csv(flat_csv, index=False)
    print(f"Saved flattened per-run-per-file CSV -> {flat_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
