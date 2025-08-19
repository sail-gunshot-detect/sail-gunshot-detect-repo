import os
import argparse
import random
import yaml
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SeparableConv2D, BatchNormalization, Activation, MaxPooling2D, Reshape, Conv1D, MaxPooling1D, GRU, GlobalMaxPooling1D, Dense
import psutil


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_load_wav_tf(desired_sr):
    """Create a load_wav function with specified sample rate."""
    @tf.function
    def load_wav_8k_mono_tf(filename, label):
        """Load and resample WAV file to specified sample rate mono."""
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        # resample wav if sample rate is not as desired
        wav = tf.cond(tf.not_equal(sample_rate, desired_sr),
                      lambda: tfio.audio.resample(wav, rate_in=sample_rate, rate_out=desired_sr),
                      lambda: wav)
        wav = tf.squeeze(wav, axis=-1)
        label = tf.cast(label, dtype=tf.float32)
        return wav, label
    return load_wav_8k_mono_tf


def create_preprocess_mel_tf(sample_rate, mel_spec_length, frame_length, frame_step, mel_bins, fmax, fmin, top_db):
    """Create a mel preprocessing function with specified parameters."""
    @tf.function
    def preprocess_mel_db_tf(wav, label):
        """Convert waveform to mel-spectrogram in dB scale."""
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

        return dbscale_mel_spectrogram, label
    return preprocess_mel_db_tf

@tf.function
def expand_spec_dim_tf(spec, label):
    """Add channel dimension for Conv2D input."""
    spec = tf.expand_dims(spec, axis=-1)
    return spec, label


def set_seed(seed):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


def build_model(config):
    """Build the gunshot detection model."""
    data_config = config['data']
    training_config = config['training']
    
    # Calculate input shape manually: mel_spec_length / (frame_length - frame_step)
    input_height = data_config['mel_spec_length'] // (data_config['frame_length'] - data_config['frame_step'])
    input_width = data_config['mel_bins']
    input_channels = 1
    
    input_shape = (input_height, input_width, input_channels)
    inputs = Input(shape=input_shape)
    x = inputs

    # Conv2D Stack - hardcoded architecture
    x = SeparableConv2D(64, (7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(64, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for Conv1D
    time_steps = x.shape[1]
    freq_bins = x.shape[2]
    channels = x.shape[3]
    x = Reshape((time_steps, freq_bins * channels))(x)

    # Conv1D stack - hardcoded architecture
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)

    x = MaxPooling1D(pool_size=(2))(x)

    # GRU layer - hardcoded architecture
    x = GRU(64, return_sequences=True)(x)

    # Global Max Pooling
    x = GlobalMaxPooling1D()(x)

    # Dense layers - hardcoded architecture
    x = Dense(64)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    learning_rate = training_config['learning_rate']
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(curve='PR', name='AUC_PR')
        ]
    )

    return model


def prepare_data(train_neg_path, train_pos_path, val_neg_path, val_pos_path, seed, config):
    """Prepare training and validation datasets."""
    data_config = config['data']
    training_config = config['training']
    
    # Create configurable functions
    load_wav_tf = create_load_wav_tf(data_config['desired_sr'])
    preprocess_mel_tf = create_preprocess_mel_tf(
        data_config['sample_rate'],
        data_config['mel_spec_length'],
        data_config['frame_length'],
        data_config['frame_step'],
        data_config['mel_bins'],
        data_config['fmax'],
        data_config['fmin'],
        data_config['top_db']
    )
    
    # Build file lists with labels
    val_neg = [[(val_neg_path + '\\' + item), 0] for item in os.listdir(val_neg_path)]
    # Remove clipped gunshot file
    if len(val_neg) > 8:
        del(val_neg[8])
    val_pos = [[(val_pos_path + '\\' + item), 1] for item in os.listdir(val_pos_path)]
    train_neg = [[(train_neg_path + '\\' + item), 0] for item in os.listdir(train_neg_path)]
    train_pos = [[(train_pos_path + '\\' + item), 1] for item in os.listdir(train_pos_path)]

    full_train = train_pos + train_neg
    full_val = val_pos + val_neg

    random.seed(seed)
    random.shuffle(full_train)
    random.shuffle(full_val)

    t_filepaths = [x[0] for x in full_train]
    t_labels = [x[1] for x in full_train]

    v_filepaths = [x[0] for x in full_val]
    v_labels = [x[1] for x in full_val]

    # Create TensorFlow datasets
    train = tf.data.Dataset.from_tensor_slices((t_filepaths, t_labels))
    train = train.map(load_wav_tf, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.map(preprocess_mel_tf, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.map(expand_spec_dim_tf, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.cache()
    train = train.shuffle(buffer_size=training_config['shuffle_buffer_size'], reshuffle_each_iteration=True)

    val = tf.data.Dataset.from_tensor_slices((v_filepaths, v_labels))
    val = val.map(load_wav_tf, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.map(preprocess_mel_tf, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.map(expand_spec_dim_tf, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.cache()

    return train, val


def main():
    parser = argparse.ArgumentParser(description='Train gunshot detection model')
    
    # Configuration argument
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    
    # Path arguments
    parser.add_argument('--train_neg_path', type=str, required=True,
                        help='Path to training negative samples folder')
    parser.add_argument('--train_pos_path', type=str, required=True,
                        help='Path to training positive samples folder')
    parser.add_argument('--val_neg_path', type=str, required=True,
                        help='Path to validation negative samples folder')
    parser.add_argument('--val_pos_path', type=str, required=True,
                        help='Path to validation positive samples folder')
    
    # Training arguments (with config defaults)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    parser.add_argument('--model_name', type=str, default='gunshot_model',
                        help='Name for the saved model')
    parser.add_argument('--model_save_dir', type=str, default='.',
                        help='Directory to save the trained model')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (overrides config)')
    
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.seed is not None:
        config['training']['seed'] = args.seed
    
    # Set environment variables for reproducibility
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    # Set random seed
    set_seed(config['training']['seed'])
    
    # Calculate and print configuration info
    input_height = config['data']['mel_spec_length'] // (config['data']['frame_length'] - config['data']['frame_step'])
    input_width = config['data']['mel_bins']
    
    print(f"Configuration loaded:")
    print(f"  Sample rate: {config['data']['sample_rate']} Hz")
    print(f"  Mel spec length: {config['data']['mel_spec_length']}")
    print(f"  Mel bins: {config['data']['mel_bins']}")
    print(f"  Input shape: ({input_height}, {input_width}, 1) [calculated: mel_spec_length / (frame_length - frame_step)]")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Seed: {config['training']['seed']}")
    
    # Print system info
    mem = psutil.virtual_memory()
    print(f"Total RAM:     {mem.total / 1e9:.2f} GB")
    print(f"Available RAM: {mem.available / 1e9:.2f} GB")
    print(f"Used RAM:      {mem.used / 1e9:.2f} GB")
    print(f"RAM Usage:     {mem.percent}%")
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    # Prepare data
    print("Preparing datasets...")
    train_dataset, val_dataset = prepare_data(
        args.train_neg_path, args.train_pos_path,
        args.val_neg_path, args.val_pos_path, config['training']['seed'], config
    )
    
    # Batch and prefetch
    train_dataset = train_dataset.batch(config['training']['batch_size'])
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(config['training']['batch_size'])
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Build model
    print("Building model...")
    model = build_model(config)
    model.summary()
    
    # Set up callbacks
    model_save_path = os.path.join(args.model_save_dir, f"{args.model_name}.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, save_best_only=True, verbose=1
    )
    
    # Train model
    print(f"Starting training for {config['training']['epochs']} epochs...")
    hist = model.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        validation_data=val_dataset,
        callbacks=[checkpoint]
    )
    
    print(f"Training completed! Best model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
