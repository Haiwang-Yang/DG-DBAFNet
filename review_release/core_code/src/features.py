import hashlib
import warnings
import wave
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


warnings.filterwarnings(
    "ignore",
    message="Trying to estimate tuning from empty frequency set.",
    category=UserWarning,
)


SAMPLE_RATE = 44100
CLIP_SECONDS = 5.0
FEATURE_SIZE = 128


def stable_key(*parts: str) -> str:
    text = "::".join(str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def wav_duration_seconds(raw: bytes) -> float:
    try:
        with wave.open(BytesIO(raw), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except wave.Error:
        import soundfile as sf

        info = sf.info(BytesIO(raw))
        return float(info.duration)


def load_wav_mono(
    raw: bytes,
    target_sr: int = SAMPLE_RATE,
    start: float = 0.0,
    duration: float = CLIP_SECONDS,
) -> np.ndarray:
    try:
        with wave.open(BytesIO(raw), "rb") as wav:
            src_sr = wav.getframerate()
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            start_frame = max(0, int(start * src_sr))
            frame_count = max(1, int(duration * src_sr))
            wav.setpos(min(start_frame, wav.getnframes()))
            frames = wav.readframes(frame_count)

        if sampwidth == 1:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sampwidth == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
    except (wave.Error, ValueError):
        import soundfile as sf

        src_sr = sf.info(BytesIO(raw)).samplerate
        start_frame = max(0, int(start * src_sr))
        frame_count = max(1, int(duration * src_sr))
        audio, src_sr = sf.read(
            BytesIO(raw),
            start=start_frame,
            frames=frame_count,
            dtype="float32",
            always_2d=True,
        )
        audio = audio.mean(axis=1)

    if src_sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=src_sr, target_sr=target_sr)

    target_len = int(target_sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]

    return audio.astype(np.float32)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), copy=False)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax - xmin < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin)


def _resize_feature(x: np.ndarray, size: int = FEATURE_SIZE) -> torch.Tensor:
    tensor = torch.from_numpy(_minmax(x)).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).squeeze(0)


def extract_stacked_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    size: int = FEATURE_SIZE,
) -> torch.Tensor:
    import librosa

    hop_length = 1024
    n_fft = 2048
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    cqt = np.abs(
        librosa.cqt(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            n_bins=84,
            bins_per_octave=12,
        )
    )
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_chroma=12,
    )

    return torch.stack(
        [
            _resize_feature(mfcc, size),
            _resize_feature(librosa.amplitude_to_db(cqt, ref=np.max), size),
            _resize_feature(chroma, size),
        ],
        dim=0,
    ).float()


def cached_stacked_features(
    raw: bytes,
    cache_dir: Path,
    cache_key: str,
    start: float,
    clip_seconds: float = CLIP_SECONDS,
    target_sr: int = SAMPLE_RATE,
    size: int = FEATURE_SIZE,
) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.npy"
    if cache_path.exists():
        return torch.from_numpy(np.load(cache_path)).float()

    audio = load_wav_mono(raw, target_sr=target_sr, start=start, duration=clip_seconds)
    features = extract_stacked_features(audio, sr=target_sr, size=size)
    np.save(cache_path, features.numpy().astype(np.float32))
    return features
