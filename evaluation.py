import os
import argparse
import time
import csv
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from scipy.interpolate import interp1d


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_load_wav_tf_nolabel(desired_sr):
    """Create a load_wav function with specified sample rate for inference."""
    @tf.function
    def load_wav_8k_mono_tf_nolabel(filename):
        """Load WAV file without label for inference."""
        file_contents = tf.io.read_file(filename)
        wav, sample_rate_file = tf.audio.decode_wav(file_contents, desired_channels=1)
        sample_rate_file = tf.cast(sample_rate_file, dtype=tf.int64)
        # resample wav if sample rate is not as desired
        wav = tf.cond(tf.not_equal(sample_rate_file, desired_sr),
                      lambda: tfio.audio.resample(wav, rate_in=sample_rate_file, rate_out=desired_sr),
                      lambda: wav)
        wav = tf.squeeze(wav, axis=-1)
        return wav
    return load_wav_8k_mono_tf_nolabel


def create_preprocess_mel_tf_nolabel(sample_rate, mel_spec_length, frame_length, frame_step, mel_bins, fmax, fmin, top_db):
    """Create a mel preprocessing function with specified parameters for inference."""
    @tf.function
    def preprocess_mel_db_tf_nolabel(wav):
        """Convert waveform to mel-spectrogram in dB scale without label."""
        # pad wav if too short and cut it if too long
        wav_len = tf.shape(wav)[0]
        wav = tf.cond(wav_len > mel_spec_length,
                      lambda: wav[:mel_spec_length],
                      lambda: tf.cond(wav_len < mel_spec_length,
                                      lambda: tf.concat([wav, tf.zeros([mel_spec_length - wav_len], dtype=tf.float32)], axis=0),
                                      lambda: wav))

        # create the spectrogram
        spectrogram = tfio.audio.spectrogram(wav, frame_length, frame_length, frame_step)
        mel_spectrogram = tfio.audio.melscale(spectrogram, sample_rate, mel_bins, fmin, fmax)
        dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=top_db)

        return dbscale_mel_spectrogram
    return preprocess_mel_db_tf_nolabel


@tf.function
def expand_spec_dim_tf_nolabel(spec):
    """Add channel dimension for Conv2D input without label."""
    spec = tf.expand_dims(spec, axis=-1)
    return spec


def get_preds(filepaths, model, config, batch_size=None):
    """Get predictions from model for given filepaths."""
    if batch_size is None:
        batch_size = config['evaluation']['inference_batch_size']
    
    data_config = config['data']
    
    # Create configurable functions
    load_wav_tf = create_load_wav_tf_nolabel(data_config['desired_sr'])
    preprocess_mel_tf = create_preprocess_mel_tf_nolabel(
        data_config['sample_rate'],
        data_config['mel_spec_length'],
        data_config['frame_length'],
        data_config['frame_step'],
        data_config['mel_bins'],
        data_config['fmax'],
        data_config['fmin'],
        data_config['top_db']
    )
    
    start = time.time()
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    ds = ds.map(load_wav_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(preprocess_mel_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(expand_spec_dim_tf_nolabel)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    start = time.time()
    for _ in ds.take(1):
        pass
    print(f'Time to build and fetch one batch: {time.time() - start:.2f}s')

    start = time.time()
    y_preds = model.predict(ds)
    print(f'Time to process files and perform inference: {time.time()-start}')
    return y_preds


def get_metrics_custom(labels, preds, filepaths, threshold, config, make_csv=False, 
                       make_true_specs=False, num_true_specs=0, 
                       make_misclassified_specs=False, num_misclassified_specs=0, 
                       model_name="model", output_dir="."):
    """Calculate custom metrics and optionally save results."""
    data_config = config['data']
    
    # Create configurable functions for spectrogram visualization
    load_wav_tf = create_load_wav_tf_nolabel(data_config['desired_sr'])
    preprocess_mel_tf = create_preprocess_mel_tf_nolabel(
        data_config['sample_rate'],
        data_config['mel_spec_length'],
        data_config['frame_length'],
        data_config['frame_step'],
        data_config['mel_bins'],
        data_config['fmax'],
        data_config['fmin'],
        data_config['top_db']
    )
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    misclassified_files = []

    for i in range(len(labels)):
        pred = ((preds[i])[0])
        label = labels[i]
        filepath = filepaths[i]
        if pred < threshold and label==0:
            true_negatives += 1
            prediction_type = 'TN'
            distance = abs(label - pred)
            misclassified_files.append([filepath, label, pred, prediction_type, distance, threshold])
            if make_true_specs and (true_positives + true_negatives) < num_true_specs:
                print(label, pred, prediction_type, filepath)
                wav = load_wav_tf(filepath)
                spec = preprocess_mel_tf(wav)
                plt.figure()
                plt.imshow(tf.transpose(spec))
                plt.show()
        elif pred < threshold and label==1:
            false_negatives += 1
            prediction_type = 'FN'
            distance = abs(label - pred)
            misclassified_files.append([filepath, label, pred, prediction_type, distance, threshold])
            if make_misclassified_specs and (false_negatives + false_positives) < num_misclassified_specs:
                print(label, pred, prediction_type, filepath)
                wav = load_wav_tf(filepath)
                spec = preprocess_mel_tf(wav)
                plt.figure()
                plt.imshow(tf.transpose(spec))
                plt.show()
        elif pred >= threshold and label==0:
            false_positives += 1
            prediction_type = 'FP'
            distance = abs(label - pred)
            misclassified_files.append([filepath, label, pred, prediction_type, distance, threshold])
            if make_misclassified_specs and (false_negatives + false_positives) < num_misclassified_specs:
                print(label, pred, prediction_type, filepath)
                wav = load_wav_tf(filepath)
                spec = preprocess_mel_tf(wav)
                plt.figure()
                plt.imshow(tf.transpose(spec))
                plt.show()
        elif pred >= threshold and label==1:
            true_positives += 1
            prediction_type = 'TP'
            distance = abs(label - pred)
            misclassified_files.append([filepath, label, pred, prediction_type, distance, threshold])
            if make_true_specs and (true_positives + true_negatives) < num_true_specs:
                print(label, pred, prediction_type, filepath)
                wav = load_wav_tf(filepath)
                spec = preprocess_mel_tf(wav)
                plt.figure()
                plt.imshow(tf.transpose(spec))
                plt.show()

    precision = true_positives / (true_positives + false_positives + 1e-5)
    recall = true_positives / (true_positives + false_negatives + 1e-5)
    f1 = 2*(precision*recall) / (precision + recall + 1e-5)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    
    print(f"Evaluation Metrics for {model_name}:")
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if make_csv:
        misclassified_files = sorted(misclassified_files, key=lambda x: x[4], reverse=True)
        metrics_row = ['', '', '', '', '', '', '', 'Prediction Type', f'True Positives: {true_positives:.2f}', f'False Positives: {false_positives:.2f}', f'True Negatives: {true_negatives:.2f}', f'False Negatives: {false_negatives:.2f}', f'Accuracy: {accuracy:.2f}', f'Recall: {recall:.2f}', f'Precision: {precision:.2f}']
        misclassified_files.insert(0, metrics_row)
        now = datetime.now()
        date = now.strftime("%m-%d-%Y,%H-%M-%S")
        csv_path = os.path.join(output_dir, f'misclassified_files_{model_name}_{date}.csv')
        with open(csv_path,'w', newline='') as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['Filepath', 'Label', 'Prediction', 'Prediction Type', 'Distance', 'Threshold'])
            for row in misclassified_files:
                newline=''
                csv_out.writerow(row)
        print(f"CSV saved to: {csv_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def get_pr_curve(y_true, y_pred_probs, save_figs=False, model_name="model", output_dir="."):
    """Generate precision-recall curve and optionally save it."""
    y_pred_probs = [pred for pred in y_pred_probs]
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    area = auc(recall,precision)
    avg_precision = average_precision_score(y_true, y_pred_probs)

    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Find the best F1 score and its associated precision, recall, threshold
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_f1_precision = precision[best_f1_idx]
    best_f1_recall = recall[best_f1_idx]
    best_f1_threshold = thresholds[best_f1_idx if best_f1_idx < len(thresholds) else best_f1_idx-1]

    # Find the point closest to (1,1) on the PR curve
    distances = np.sqrt((1 - precision)**2 + (1 - recall)**2)
    best_pr_idx = np.argmin(distances)
    best_pr_precision = precision[best_pr_idx]
    best_pr_recall = recall[best_pr_idx]
    best_pr_threshold = thresholds[best_pr_idx if best_pr_idx < len(thresholds) else best_pr_idx-1]
    best_pr_f1 = f1_scores[best_pr_idx]

    # Interpolate precision as a function of recall
    # Note: Recall values decrease in order for precision-recall curve, so we reverse them
    recall_for_interp = recall[::-1]
    precision_for_interp = precision[::-1]

    # Create interpolator
    precision_interp_func = interp1d(recall_for_interp, precision_for_interp, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Get interpolated precision at recall = 0.95
    recall_target = 0.95
    precision_at_95_exact = float(precision_interp_func(recall_target))

    # Plot metrics
    plt.figure(figsize=(10,6))
    plt.plot(recall, precision, label=f'Average Precision (AP) = {avg_precision:.3f})', color='blue')
    plt.scatter(best_f1_recall, best_f1_precision, color='red', label=f'Best F1 (F1={best_f1:.3f}, Th={best_f1_threshold:.3f})')
    plt.scatter(best_pr_recall, best_pr_precision, color='green', label=f'Precision of min distance: {best_pr_precision:.3f}, Recall of min distance={best_pr_recall:.3f}')

    # Plot the No Skill classifier
    no_skill = len([label for label in y_true if label==1]) / len(y_true)
    plt.plot([0,1], [no_skill, no_skill], ls='--', color='black', label='No-Skill Classifier')

    # Plot the exact point on PR curve
    plt.scatter(0.95, precision_at_95_exact, color='orange', label=f'Interpolated Precision at Recall=0.95: {precision_at_95_exact:.3f}')
    plt.axvline(0.95, color='orange', linestyle='--')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    
    if save_figs:
        now = datetime.now()
        date = now.strftime("%m-%d-%Y,%H-%M-%S")
        fig_path = os.path.join(output_dir, f'PR_curve_{model_name}_{date}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"PR curve saved to: {fig_path}")
    
    plt.show()
    
    return {
        'best_f1_threshold': best_f1_threshold,
        'best_f1': best_f1,
        'avg_precision': avg_precision,
        'precision_at_95_recall': precision_at_95_exact
    }


def prepare_dataset_for_evaluation(neg_path, pos_path):
    """Prepare dataset for evaluation."""
    neg_files = [[(neg_path + '\\' + item), 0] for item in os.listdir(neg_path)]
    pos_files = [[(pos_path + '\\' + item), 1] for item in os.listdir(pos_path)]
    
    # Remove clipped gunshot file if it's the validation set (index 8)
    # This is a specific case from the original notebook
    if len(neg_files) > 8 and 'Validation' in neg_path:
        del(neg_files[8])
    
    full_dataset = pos_files + neg_files
    
    filepaths = [x[0] for x in full_dataset]
    labels = [x[1] for x in full_dataset]
    
    return filepaths, labels


def main():
    parser = argparse.ArgumentParser(description='Evaluate gunshot detection model')
    
    # Configuration argument
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.keras file)')
    parser.add_argument('--neg_path', type=str, required=True,
                        help='Path to negative samples folder for evaluation')
    parser.add_argument('--pos_path', type=str, required=True,
                        help='Path to positive samples folder for evaluation')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save outputs (default: current directory)')
    
    # Optional arguments
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for inference (default: from config)')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save misclassified files to CSV')
    parser.add_argument('--save_pr_curve', action='store_true',
                        help='Save precision-recall curve plot')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name for output files (default: derived from model path)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Use config defaults if not overridden
    threshold = args.threshold if args.threshold is not None else config['evaluation']['default_threshold']
    batch_size = args.batch_size if args.batch_size is not None else config['evaluation']['inference_batch_size']
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.neg_path):
        raise FileNotFoundError(f"Negative samples path not found: {args.neg_path}")
    if not os.path.exists(args.pos_path):
        raise FileNotFoundError(f"Positive samples path not found: {args.pos_path}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set model name
    if args.model_name is None:
        args.model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    
    # Prepare dataset
    print("Preparing evaluation dataset...")
    filepaths, labels = prepare_dataset_for_evaluation(args.neg_path, args.pos_path)
    print(f"Found {len(filepaths)} files ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
    
    # Get predictions
    print("Getting predictions...")
    predictions = get_preds(filepaths, model, config, batch_size)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = get_metrics_custom(
        labels, predictions, filepaths, threshold, config,
        make_csv=args.save_csv, model_name=args.model_name, output_dir=args.output_dir
    )
    
    # Generate PR curve
    print("Generating precision-recall curve...")
    pr_results = get_pr_curve(
        labels, predictions, save_figs=args.save_pr_curve, 
        model_name=args.model_name, output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {len(filepaths)} files")
    print(f"Threshold: {threshold}")
    print(f"Best F1 Threshold: {pr_results['best_f1_threshold']:.3f}")
    print(f"Average Precision: {pr_results['avg_precision']:.3f}")
    print(f"Precision at 95% Recall: {pr_results['precision_at_95_recall']:.3f}")
    
    return metrics, pr_results


if __name__ == "__main__":
    main()
