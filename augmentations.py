import tensorflow as tf
import tensorflow_io as tfio


# Helper functions
def rand_uniform(low, high):
    """Return a random float in the bounds."""
    return tf.random.uniform([], low, high)


def apply_with_p(p, fn, x):
    """Apply fn(x) with probability p, else return x."""
    return tf.cond(tf.less(tf.random.uniform([], 0, 1), p),
                   lambda: fn(x),
                   lambda: x)


# Waveform augmentation functions
@tf.function
def add_gaussian_snr(wav, min_snr=16.0, max_snr=32.0):
    """Add Gaussian noise with specified SNR range."""
    rms_signal = tf.sqrt(tf.reduce_mean(wav**2))
    snr_db = rand_uniform(min_snr, max_snr)
    snr = 10.0**(snr_db / 20.0)
    noise_rms = rms_signal / snr
    noise = tf.random.normal(tf.shape(wav)) * noise_rms
    return wav + noise


@tf.function
def gain(wav, min_db=-8.0, max_db=24.0):
    """Apply random gain (scalar amplification)."""
    db = rand_uniform(min_db, max_db)
    factor = 10.0**(db / 20.0)
    return wav * factor


@tf.function
def naive_pitch_shift(wav, sample_rate=8000, min_semi=-4.0, max_semi=4.0):
    """Apply pitch shift via resampling."""
    semi = rand_uniform(min_semi, max_semi)
    rate = 2 ** (semi / 12.0)
    # compute rate in Hz
    rate_in_hz = tf.cast(sample_rate, tf.int64)
    rate_out_hz = tf.cast(tf.math.round(sample_rate * rate), tf.int64)
    # directly resample: shortens/lengthens + shifts pitch
    shifted = tfio.audio.resample(wav, rate_in=rate_in_hz, rate_out=rate_out_hz)
    return shifted


@tf.function
def shift(wav, min_shift=-0.5, max_shift=0.5):
    """Shift (roll) the signal in time."""
    frac = rand_uniform(min_shift, max_shift)
    shift_amt = tf.cast(frac * tf.cast(tf.shape(wav)[0], tf.float32), tf.int32)
    return tf.roll(wav, shift=shift_amt, axis=0)


@tf.function
def load_wav_8k_mono_tf_nolabel(filename):
    """Load WAV file without label (for background noise loading)."""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    desired_sr = 8000
    # resample wav if sample rate is not as desired
    wav = tf.cond(tf.not_equal(sample_rate, desired_sr),
                  lambda: tfio.audio.resample(wav, rate_in=sample_rate, rate_out=desired_sr),
                  lambda: wav)
    wav = tf.squeeze(wav, axis=-1)
    return wav


@tf.function
def add_background_noise(wav, bg_files, min_snr=5.0, max_snr=24.0):
    """Add background noise at specified SNR range.
    
    Args:
        wav: Input waveform tensor
        bg_files: tf.constant tensor of background file paths
        min_snr: Minimum SNR in dB
        max_snr: Maximum SNR in dB
    """
    # Pick one bg file (at graph build time bg_files must be a tf.constant)
    idx = tf.random.uniform([], 0, tf.shape(bg_files)[0], dtype=tf.int32)
    bg_path = bg_files[idx]
    bg_audio = load_wav_8k_mono_tf_nolabel(bg_path)
    bg_audio = bg_audio[:tf.shape(wav)[0]]
    bg_audio = tf.pad(bg_audio, [[0, tf.maximum(0, tf.shape(wav)[0] - tf.shape(bg_audio)[0])]])
    # Scale bg to desired SNR
    rms_w = tf.sqrt(tf.reduce_mean(wav**2))
    snr_db = rand_uniform(min_snr, max_snr)
    snr = 10.0**(snr_db / 20.0)
    rms_bg = tf.sqrt(tf.reduce_mean(bg_audio**2))
    scale = rms_w / (snr * (rms_bg + 1e-8))
    return wav + bg_audio * scale


@tf.function
def time_mask(wav, min_part=0.02, max_part=0.05, min_masks=1, max_masks=4):
    """Apply time masking to waveform."""
    length = tf.shape(wav)[0]
    num_masks = tf.random.uniform([], minval=min_masks, maxval=max_masks + 1, dtype=tf.int32)
    
    def apply_one_mask(wav, _):
        mask_size = tf.cast(rand_uniform(min_part, max_part) * tf.cast(length, tf.float32), tf.int32)
        start = tf.random.uniform([], 0, length - mask_size, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([start], dtype=wav.dtype),
            tf.zeros([mask_size], dtype=wav.dtype),
            tf.ones([length - start - mask_size], dtype=wav.dtype)
        ], axis=0)
        return wav * mask
    
    # fold over a range to apply masks
    masked = tf.foldl(apply_one_mask,
                      elems=tf.range(num_masks),
                      initializer=wav)
    return masked


# Spectrogram augmentation functions
@tf.function
def freq_mask(spec, min_part=0.02, max_part=0.05, min_masks=1, max_masks=5):
    """Apply frequency masking to spectrogram."""
    freq_len = tf.shape(spec)[1]
    # pick how many masks to apply
    num_masks = tf.random.uniform(
        [], minval=min_masks, maxval=max_masks + 1, dtype=tf.int32)

    def apply_one_mask(s, _):
        # 1) choose mask size in bins
        part = tf.random.uniform([], minval=min_part, maxval=max_part)
        mask_size = tf.cast(part * tf.cast(freq_len, tf.float32), tf.int32)

        # 2) choose start bin
        start = tf.random.uniform([], 0, freq_len - mask_size, dtype=tf.int32)

        # 3) build 1-D mask [freq_len]
        mask1d = tf.concat([
            tf.ones([start], dtype=s.dtype),
            tf.zeros([mask_size], dtype=s.dtype),
            tf.ones([freq_len - start - mask_size], dtype=s.dtype),
        ], axis=0)

        # 4) broadcast to [time, freq]
        mask2d = tf.expand_dims(mask1d, 0)

        # 5) replace masked bins by min value in spec
        min_val = tf.reduce_min(s)
        return s * mask2d + min_val * (1.0 - mask2d)

    # fold over a range to apply masks
    masked = tf.foldl(apply_one_mask,
                      elems=tf.range(num_masks),
                      initializer=spec)
    return masked


# Combined augmentation functions
@tf.function
def augment_waveform_tf(wav, label, bg_files):
    """Apply combined waveform augmentations.
    
    Args:
        wav: Input waveform
        label: Label tensor
        bg_files: tf.constant tensor of background file paths
    """
    x = wav
    x = apply_with_p(0.3, lambda w: add_gaussian_snr(w, 16.0, 32.0), x)
    x = apply_with_p(1.0, lambda w: add_background_noise(w, bg_files, 16, 24), x)
    x = apply_with_p(0.3, lambda w: gain(w, -8.0, 8.0), x)
    x = apply_with_p(0.3, lambda w: shift(w, 0.25, 0.5), x)
    x = apply_with_p(0.3, lambda w: naive_pitch_shift(w, 8000, -2.0, 2.0), x)
    x = apply_with_p(0.3, lambda w: time_mask(w, 0.01, 0.03, 1, 1), x)
    return x, label


@tf.function
def augment_spec_tf(spec, label):
    """Apply combined spectrogram augmentations."""
    x = spec
    x = apply_with_p(0.3, lambda s: freq_mask(s, 0.01, 0.03, 1, 1), x)
    return x, label
