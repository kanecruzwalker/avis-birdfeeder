"""
src/audio/preprocess.py

Converts raw audio into mel spectrograms suitable for BirdNET and our fine-tuned model.

Pipeline:
    WAV file or NumPy array
        → resample to target rate (48 kHz for BirdNET compatibility)
        → convert to mono
        → normalize amplitude to [-1,1]
        → compute mel spectrogram
        → convert to dB scale
        → return as 2-D NumPy array (or save as PNG for visual inspection)

Why mel spectrograms?
    Bird vocalizations have rich frequency structure. Mel spectrograms compress raw
    audio into a 2-D image-like representation that captures time-frequency patterns
    at a perceptually meaningful scale. Making image classification CNNs directly
    applicable to audio data. This is the standard approach in bird audio ML, and
    is the input format expected by BirdNET.

Why dB scale?
    Raw power values span many orders of magnitude. Converting to decibels (log scale)
    compresses this dynamic range into a roughly [-80, 0] dB window, which greatly
    improves numerical stability and model training behavior.

Config keys consumed (configs/thresholds.yaml):
    audio.sample_rate       - target sample rate (default 48000)
    audio.window_seconds    - not used directly here, but sets context for callers

Config keys consumed (configs/paths.yaml):
    datasets.spectrograms   - where preprocess_file() writes output PNGs (optional)

Dependencies:
    librosa >= 0.10     - audio I/O, resampling, mel spectrogram computation
    soundfile           - WAV backend for librosa on Windows (avoids audioread issues)
    numpy               - array operations
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Module-level defaults - all sourced from configs/thresholds.yaml
# Callers should pass values loaded from YAML rather than relying on these.
# They are provided so individual functions remain usable in isolation
# (e.g., during testing with synthetic data).
# -----------------------------------------------------------------------------
_DEFAULT_SR: int = 48_000  # BirdNET expects 48 kHz - do not change without retraining
_DEFAULT_N_MELS: int = 128  # mel filter banks; matches BirdNET input conventions
_DEFAULT_HOP_LENGTH: int = 512  # ~10.7 ms per frame at 48 kHz
_DEFAULT_N_FFT: int = 2048  # ~42.7 ms FFT windows; captures typical bird call structure
_DEFAULT_F_MIN: float = 150.0  # Hz; below this is mostly wind/hum, not bird vocalizaiont
_DEFAULT_F_MAX: float = 15_000.0  # Hz; above this is ultrasonic, not useful for most species


# -------------------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------------------


def load_wav(path: str | Path, target_sr: int = _DEFAULT_SR) -> tuple[np.ndarray, int]:
    """
    Load a WAV file, resample to target_sr, and convert to mono.

    Uses librosa's built-in resampling (resampy by default when installed, otherwise scipy).
    Mono conversion averages all channels, which avoids phase cancellation artifacts from
    simply taking the first channel.

    Args:
        path: Path to the input WAV file. Must exist and be readable.
        target_sr: Target sample rate in Hz. Defaults to 48 000 (BirdNET standard).
                   Pass the value from ''configs/thresholds.yaml -> audio.sample_rate''.

    Returns:
        Tuple of:
            audio (np.ndarray): 1-D float32 array of audio samples.
            sample_rate (int): Actual sample rate after resampling (== target_sr).

    Raises:
        FileNotFoundError: If 'path' does not exist
        RuntimeError: If librosa fails to decode the files
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    logger.debug("Loading WAV: %s (target_sr=%d)", path, target_sr)

    # mono=True averages all channels; res_type = 'kaiser_best' gives high quality
    # resampling at a modest cmpute cost - acceptable for offline preprocessing
    try:
        audio, sr = librosa.load(path, sr=target_sr, mono=True, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"librosa failed to load '{path}': {exc}") from exc

    logger.debug("Loaded %d samples at %d Hz (duration = %.2fs)", len(audio), sr, len(audio) / sr)
    return audio, sr


def normalize(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude so peak absolute value is 1.0.

    Formula: y[n] = x[n] / max(|x|)

    This is peak normalization. Every sample is divided by the loudest
    single sample in the clip. The result is guranteed to be in [-1, 1]
    with the loudest moment sitting exactly at +-1.0.

    Why peak (not RMS)?
        RMS normalization targets average loudness, which would boost
        quiet clips with occasional loud spikes past +- 1.0, causing clipping.
        Peak normalization is clip-safe by construction and preserves the
        internal dynamics of the recording. A loud call followed by a soft trill
        keeps its relative shape.

    Why normalize at all?
        A 3-second clip recorded close to the mic and one recorded far away
        will have very different raw amplitudes. Without normalization the
        mel spectrogram of the quiet clip would sit near -80 dB everywhere,
        making the patterns unrecognizable to the model. Normalizing first
        ensures both clips occupy the same dB range and the model sees
        structure, not volume.

    Args:
        audio: 1-D float32 NumPy array of audio samples.

    Returns:
        Normalized float32 array of the same shape, values in [-1.0, 1.0].
    """

    # max(|x|)    -> find the largest absolute value across all samples
    # np.abs      -> maps every sample to its distance from zero (negative -> positive)
    # then np.max -> finds the single loudest peak in the entire clip
    peak = np.max(np.abs(audio))

    # Guard: if the peak is below 1e-6 (essentially zero), the clip is silent.
    # Dividing by a ner-zero peak would produce NaN or Inf, which would
    # silently corrupt the spectrogram and crash the model downstream.
    # Returning zeros is the correct behavior. A silent clip stays silent
    if peak < 1e-6:
        logger.debug("normalize: near-silent clip (peak=%.2e), returning zeros", peak)
        return np.zeros_like(audio)

    # y[n] = x[n] / peak
    # Each sample is scaled by the same constant factor (1/peak), so the
    # shape of the waveform is preserved exactly. Only the amplitude changes.
    # The loudest sample becomes exactly +- 1.0. All others scale proportionally.
    return (audio / peak).astype(np.float32)


def to_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = _DEFAULT_SR,
    n_mels: int = _DEFAULT_N_MELS,
    hop_length: int = _DEFAULT_HOP_LENGTH,
    n_fft: int = _DEFAULT_N_FFT,
    f_min: float = _DEFAULT_F_MIN,
    f_max: float = _DEFAULT_F_MAX,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Convert a 1-D audio array to a 2-D log-mel spectrogram in dB scale.

    Math overview:
        1. STFT — slide a window of n_fft samples across the audio with a
           step of hop_length samples. At each position, compute the FFT to
           get the frequency content at that moment in time. The result is a
           complex matrix of shape (n_fft/2 + 1, time_frames).

        2. Power spectrum — take |STFT|^2 (squared magnitude), discarding
           phase. Phase encodes timing of wave cycles, which is irrelevant
           for species classification; power encodes which frequencies are
           loud or quiet, which is what matters.

        3. Mel filter bank — multiply the power spectrum by a bank of
           triangular filters spaced on the mel scale. The mel scale is
           logarithmic and mimics how the cochlea perceives pitch: equal
           perceptual steps correspond to larger Hz steps at high frequencies.
           This compresses 1025 raw FFT bins into n_mels=128 perceptually
           meaningful bands. Bird calls sit in a narrow Hz range; mel spacing
           gives them proportionally more bins than low-frequency noise.

        4. dB conversion — apply 10 * log10(power / ref), where ref is the
           maximum power value in the clip. This maps the dynamic range from
           a multiplicative scale (10x louder = 10x the value) to an additive
           dB scale (10x louder = +10 dB). The loudest frame in the clip
           becomes 0 dB; everything quieter becomes negative. Values below
           -top_db are clipped, giving a clean [-80, 0] output range.

    Output shape: ``(n_mels, time_frames)`` where
        ``time_frames = 1 + len(audio) // hop_length``

    Args:
        audio:       1-D float32 NumPy array of audio samples.
        sample_rate: Sample rate of the input audio. Must match the rate used
                     when the audio was loaded (i.e., target_sr from load_wav).
        n_mels:      Number of mel filter banks (height of the spectrogram).
                     128 is the BirdNET convention.
        hop_length:  Number of samples between STFT frames. Controls the
                     temporal resolution of the output. 512 samples at 48 kHz
                     = ~10.7 ms per column — fine enough to resolve bird call
                     onsets and trills.
        n_fft:       FFT window size in samples. Larger = better frequency
                     resolution, worse time resolution. 2048 at 48 kHz =
                     ~42.7 ms window, which captures the periodic structure
                     of most bird vocalizations cleanly.
        f_min:       Lowest frequency included in the mel filter bank (Hz).
                     150 Hz removes subsonic hum and wind rumble without
                     losing any bird vocalization content.
        f_max:       Highest frequency included (Hz). 15 000 Hz covers all
                     common San Diego species — most calls peak below 12 kHz.
                     Dropping ultrasonic content reduces spectrogram noise.
        top_db:      Dynamic range of the dB conversion. Values more than
                     top_db below the loudest frame are clipped to -top_db.
                     80 dB is the librosa default; it preserves all meaningful
                     signal while discarding inaudible noise floor variation.

    Returns:
        2-D float32 NumPy array of shape ``(n_mels, time_frames)`` in dB scale.
        Values are in roughly ``[-top_db, 0]`` with 0 dB at the loudest frame.

    Raises:
        ValueError: If ``audio`` is not 1-D, or has fewer samples than ``n_fft``.
    """
    if audio.ndim != 1:
        raise ValueError(
            f"to_mel_spectrogram expects a 1-D array, got shape {audio.shape}. "
            "Did you forget to convert to mono?"
        )
    if len(audio) < n_fft:
        raise ValueError(
            f"Audio too short: {len(audio)} samples < n_fft={n_fft}. "
            "Pad the audio or use a smaller n_fft."
        )

    # Step 1 — mel-scaled power spectrogram.
    # Internally librosa computes: STFT → |STFT|^2 → mel filter bank.
    # The output is a power matrix (not amplitude, not dB) of shape
    # (n_mels, time_frames), where each value is proportional to acoustic
    # energy in that frequency band at that moment in time.
    # fmin/fmax clip the filter bank to the bird-relevant frequency range,
    # so no mel bins are wasted on wind rumble or ultrasonic content.
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=f_min,
        fmax=f_max,
    )

    # Step 2 — convert power to dB scale.
    # Formula: dB[i,j] = 10 * log10(mel_spec[i,j] / ref)
    # ref=np.max means the reference is the loudest value in this clip,
    # so the output is always relative — the peak frame is 0 dB and
    # everything else is negative. This makes clips comparable regardless
    # of their absolute recording volume.
    # top_db=80.0 clips values below -80 dB, keeping the output range clean
    # at [-80, 0] and preventing near-zero power values from producing
    # extreme negative numbers that would destabilize model training.
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=top_db)

    logger.debug(
        "Spectrogram shape: %s, range: [%.1f, %.1f] dB",
        mel_spec_db.shape,
        mel_spec_db.min(),
        mel_spec_db.max(),
    )

    return mel_spec_db.astype(np.float32)


def preprocess_file(
    path: str | Path,
    target_sr: int = _DEFAULT_SR,
    n_mels: int = _DEFAULT_N_MELS,
    hop_length: int = _DEFAULT_HOP_LENGTH,
    n_fft: int = _DEFAULT_N_FFT,
    f_min: float = _DEFAULT_F_MIN,
    f_max: float = _DEFAULT_F_MAX,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single audio file.

    Convenience wrapper that chains: load_wav → normalize → to_mel_spectrogram.

    This is the primary entry point for the audio pipeline. The agent and
    dataset utilities should call this function rather than the individual
    steps, so that any future changes to the pipeline only require updating
    this function.

    Args:
        path:        Path to input WAV file.
        target_sr:   Target sample rate in Hz (default: 48 000).
                     Pass ``configs['audio']['sample_rate']`` from thresholds.yaml.
        n_mels:      Number of mel filter banks.
        hop_length:  STFT hop length in samples.
        n_fft:       FFT window size in samples.
        f_min:       Minimum frequency for mel filter bank (Hz).
        f_max:       Maximum frequency for mel filter bank (Hz).

    Returns:
        2-D float32 NumPy array of shape ``(n_mels, time_frames)`` in dB scale,
        ready for model input or saving as a .npy file.

    Raises:
        FileNotFoundError: If the WAV file does not exist.
        ValueError:        If the audio is too short for the given n_fft.
    """
    logger.info("Preprocessing: %s", path)

    # Chain the three pipeline steps. Each function is independently testable
    # and has its own validation — errors surface with clear messages pointing
    # to which step failed and why.
    audio, sr = load_wav(path, target_sr=target_sr)
    audio = normalize(audio)
    spectrogram = to_mel_spectrogram(
        audio,
        sample_rate=sr,  # sr is guaranteed == target_sr after load_wav
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        f_min=f_min,
        f_max=f_max,
    )

    logger.info("Preprocessed '%s' → spectrogram shape %s", Path(path).name, spectrogram.shape)
    return spectrogram


def preprocess_array(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: int = _DEFAULT_SR,
    n_mels: int = _DEFAULT_N_MELS,
    hop_length: int = _DEFAULT_HOP_LENGTH,
    n_fft: int = _DEFAULT_N_FFT,
    f_min: float = _DEFAULT_F_MIN,
    f_max: float = _DEFAULT_F_MAX,
) -> np.ndarray:
    """
    Full preprocessing pipeline for an in-memory audio array.

    Used by the live capture path (src.audio.capture) where audio arrives
    as a NumPy array from sounddevice rather than as a file on disk.

    Resamples if ``sample_rate != target_sr``, then normalizes and computes
    the mel spectrogram identically to ``preprocess_file``.

    Args:
        audio:       1-D float32 NumPy array of raw audio samples.
        sample_rate: Sample rate of the incoming array (from capture device).
                     The Fifine USB mic may report 44 100 or 48 000 Hz depending
                     on OS audio settings — this function handles either.
        target_sr:   Target sample rate to resample to before processing.
                     Must match the rate BirdNET was trained on (48 000 Hz).
        n_mels:      Number of mel filter banks.
        hop_length:  STFT hop length in samples.
        n_fft:       FFT window size in samples.
        f_min:       Minimum frequency for mel filter bank (Hz).
        f_max:       Maximum frequency for mel filter bank (Hz).

    Returns:
        2-D float32 NumPy array of shape ``(n_mels, time_frames)`` in dB scale.
    """
    # Resample only when necessary — resampling is computationally expensive
    # and introduces minor interpolation artifacts, so we skip it if the
    # capture device already matches the model's expected rate.
    if sample_rate != target_sr:
        logger.debug("Resampling array %d → %d Hz", sample_rate, target_sr)
        # librosa.resample uses high-quality sinc interpolation by default.
        # orig_sr and target_sr drive the ratio: e.g. 44100→48000 upsamples
        # by a factor of 48000/44100 ≈ 1.088, stretching the array slightly.
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr  # update so to_mel_spectrogram sees the correct rate

    audio = normalize(audio)
    return to_mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        f_min=f_min,
        f_max=f_max,
    )
