import csv
import math
import random
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .features import CLIP_SECONDS, FEATURE_SIZE, cached_stacked_features, stable_key, wav_duration_seconds


FAMILY_NAMES = ["wind", "percussion", "plucked_string", "bowed_string"]
FAMILY_TO_ID = {name: idx for idx, name in enumerate(FAMILY_NAMES)}


CTIS_PREFIX_TO_FAMILY = {
    "C": "wind",
    "D": "percussion",
    "T": "plucked_string",
    "L": "bowed_string",
}


CHMUSIC_ID_TO_FAMILY = {
    1: "bowed_string",    # Erhu
    2: "plucked_string",  # Pipa
    3: "plucked_string",  # Sanxian
    4: "wind",            # Dizi
    5: "wind",            # Suona
    6: "bowed_string",    # Zhuiqin
    7: "plucked_string",  # Zhongruan
    8: "plucked_string",  # Liuqin
    9: "plucked_string",  # Guzheng
    10: "plucked_string", # Yangqin
    11: "wind",           # Sheng
}


FOLKMUSIC_FOLDER_TO_FAMILY = {
    "ba": "wind",
    "dizi": "wind",
    "dongxiao": "wind",
    "hulusi": "wind",
    "sheng": "wind",
    "suona": "wind",
    "erhu": "bowed_string",
    "zhuiqin": "bowed_string",
    "guqin": "plucked_string",
    "guzheng": "plucked_string",
    "liuqin": "plucked_string",
    "pipa": "plucked_string",
    "sanxian": "plucked_string",
    "yangqin": "plucked_string",
    "zhongruan": "plucked_string",
}


@dataclass(frozen=True)
class AudioItem:
    dataset: str
    source: str
    member: str
    label: int
    family: str
    duration: float


@dataclass(frozen=True)
class ClipItem:
    audio_index: int
    start: float
    key: str


MANIFEST_FIELDS = (
    "dataset",
    "source",
    "member",
    "label",
    "family",
    "duration",
)


def portable_source(item: AudioItem, data_root: Path) -> str:
    source = Path(item.source).resolve()
    root = data_root.resolve()
    try:
        return source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Audio source {source} is outside data root {root}.") from exc


def write_manifest(path: Path, items: Sequence[AudioItem], data_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "dataset": item.dataset,
                    "source": portable_source(item, data_root),
                    "member": item.member,
                    "label": item.label,
                    "family": item.family,
                    "duration": f"{item.duration:.9f}",
                }
            )


def read_manifest(path: Path, data_root: Path) -> List[AudioItem]:
    items: List[AudioItem] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = (data_root / row["source"]).resolve()
            items.append(
                AudioItem(
                    dataset=row["dataset"],
                    source=str(source),
                    member=row["member"],
                    label=int(row["label"]),
                    family=row["family"],
                    duration=float(row["duration"]),
                )
            )
    if not items:
        raise RuntimeError(f"Manifest contains no audio files: {path}")
    return items


def audio_file_id(item: AudioItem) -> str:
    if item.member:
        identity = item.member.replace("\\", "/")
    else:
        source_parts = Path(item.source).parts
        identity = "/".join(source_parts[-3:])
    return stable_key(item.dataset, identity)


def _read_file(path: str) -> bytes:
    return Path(path).read_bytes()


def _read_zip_member(zip_path: str, member: str) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.read(member)


def _audio_bytes(item: AudioItem) -> bytes:
    if item.member:
        return _read_zip_member(item.source, item.member)
    return _read_file(item.source)


def _safe_duration(raw: bytes) -> Optional[float]:
    try:
        return wav_duration_seconds(raw)
    except Exception:
        return None


def _duration_from_path(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except Exception:
        try:
            return _safe_duration(path.read_bytes())
        except Exception:
            return None


def _duration_from_zip(zf: zipfile.ZipFile, member: str) -> Optional[float]:
    try:
        with zf.open(member, "r") as fp:
            with wave.open(fp, "rb") as wav:
                return wav.getnframes() / float(wav.getframerate())
    except Exception:
        try:
            return _safe_duration(zf.read(member))
        except Exception:
            return None


def collect_ctis(data_root: Path, limit_per_family: Optional[int] = None) -> List[AudioItem]:
    zip_path = data_root / "CTIS" / "data" / "audio.zip"
    if not zip_path.exists():
        return []

    grouped: Dict[str, List[AudioItem]] = {name: [] for name in FAMILY_NAMES}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if limit_per_family and all(len(v) >= limit_per_family for v in grouped.values()):
                break
            if info.is_dir() or not info.filename.lower().endswith(".wav"):
                continue
            parts = info.filename.replace("\\", "/").split("/")
            if len(parts) < 3:
                continue
            code = parts[1]
            family = CTIS_PREFIX_TO_FAMILY.get(code[:1])
            if not family:
                continue
            if limit_per_family and len(grouped[family]) >= limit_per_family:
                continue
            duration = _duration_from_zip(zf, info.filename)
            if not duration:
                continue
            grouped[family].append(
                AudioItem(
                    dataset="CTIS",
                    source=str(zip_path),
                    member=info.filename,
                    label=FAMILY_TO_ID[family],
                    family=family,
                    duration=duration,
                )
            )

    items: List[AudioItem] = []
    for family, family_items in grouped.items():
        family_items.sort(key=lambda x: x.member)
        items.extend(family_items[:limit_per_family] if limit_per_family else family_items)
    return items


def collect_chmusic(data_root: Path, limit_per_family: Optional[int] = None) -> List[AudioItem]:
    music_dir = data_root / "ChMusic" / "ChMusic" / "Musics"
    if not music_dir.exists():
        return []

    grouped: Dict[str, List[AudioItem]] = {name: [] for name in FAMILY_NAMES}
    for path in sorted(music_dir.glob("*.wav")):
        try:
            instrument_id = int(path.stem.split(".")[0])
        except ValueError:
            continue
        family = CHMUSIC_ID_TO_FAMILY.get(instrument_id)
        if not family:
            continue
        if limit_per_family and len(grouped[family]) >= limit_per_family:
            continue
        duration = _duration_from_path(path)
        if not duration:
            continue
        grouped[family].append(
            AudioItem(
                dataset="ChMusic",
                source=str(path),
                member="",
                label=FAMILY_TO_ID[family],
                family=family,
                duration=duration,
            )
        )
    items: List[AudioItem] = []
    for family, family_items in grouped.items():
        family_items.sort(key=lambda x: x.source)
        items.extend(family_items[:limit_per_family] if limit_per_family else family_items)
    return items


def collect_folkmusic(data_root: Path, limit_per_family: Optional[int] = None) -> List[AudioItem]:
    root = data_root / "China traditional music instrument dataset" / "FolkMusic"
    if not root.exists():
        return []

    grouped: Dict[str, List[AudioItem]] = {name: [] for name in FAMILY_NAMES}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        family = FOLKMUSIC_FOLDER_TO_FAMILY.get(folder.name)
        if not family:
            continue
        for path in sorted(folder.glob("*.mp3")):
            duration = _duration_from_path(path)
            if not duration:
                continue
            grouped[family].append(
                AudioItem(
                    dataset="FolkMusic",
                    source=str(path),
                    member="",
                    label=FAMILY_TO_ID[family],
                    family=family,
                    duration=duration,
                )
            )

    items: List[AudioItem] = []
    for family, family_items in grouped.items():
        family_items.sort(key=lambda x: x.source)
        items.extend(family_items[:limit_per_family] if limit_per_family else family_items)
    return items


def collect_audio_items(
    data_root: str,
    datasets: Sequence[str],
    limit_per_family: Optional[int] = None,
) -> List[AudioItem]:
    root = Path(data_root)
    collectors = {
        "ctis": collect_ctis,
        "chmusic": collect_chmusic,
        "folkmusic": collect_folkmusic,
    }
    items: List[AudioItem] = []
    for name in datasets:
        key = name.lower()
        if key not in collectors:
            raise ValueError(f"Unsupported local dataset: {name}")
        items.extend(collectors[key](root, limit_per_family=limit_per_family))
    return items


def split_audio_items(
    items: Sequence[AudioItem],
    split: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
) -> List[AudioItem]:
    if split == "all":
        return list(items)

    by_label: Dict[int, List[AudioItem]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)

    selected: List[AudioItem] = []
    rng = random.Random(seed)
    for label, label_items in by_label.items():
        shuffled = list(label_items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if val_ratio > 0 and n >= 3 else 0
        if split == "train":
            selected.extend(shuffled[:n_train])
        elif split == "validation":
            selected.extend(shuffled[n_train:n_train + n_val])
        elif split == "test":
            selected.extend(shuffled[n_train + n_val:])
        else:
            raise ValueError(f"Unsupported split: {split}")
    return selected


class LocalStackedFeatureDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        datasets: Sequence[str] = ("ctis", "chmusic", "folkmusic"),
        cache_dir: str = "./feature_cache",
        seed: int = 42,
        clip_seconds: float = CLIP_SECONDS,
        feature_size: int = FEATURE_SIZE,
        limit_per_family: Optional[int] = None,
        pseudo_domain_augment: bool = False,
        num_pseudo_domains: int = 4,
        noise_scale_factor: float = 0.05,
        noise_scale_min: float = 0.005,
        noise_scale_max: float = 0.05,
        smoothing_kernel: int = 7,
        smoothing_blend: float = 0.3,
        frequency_mask_ratio: float = 0.1,
        time_mask_ratio: float = 0.1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.0,
        manifest_path: Optional[str] = None,
    ):
        if noise_scale_factor < 0 or noise_scale_min < 0 or noise_scale_max < 0:
            raise ValueError("Noise scale parameters must be non-negative.")
        if noise_scale_min > noise_scale_max:
            raise ValueError("noise_scale_min cannot exceed noise_scale_max.")
        if smoothing_kernel < 1 or smoothing_kernel % 2 == 0:
            raise ValueError("smoothing_kernel must be a positive odd integer.")
        if not 0.0 <= smoothing_blend <= 1.0:
            raise ValueError("smoothing_blend must be in [0, 1].")
        if not 0.0 <= frequency_mask_ratio <= 1.0:
            raise ValueError("frequency_mask_ratio must be in [0, 1].")
        if not 0.0 <= time_mask_ratio <= 1.0:
            raise ValueError("time_mask_ratio must be in [0, 1].")
        root = Path(data_root)
        if manifest_path:
            self.audio_items = read_manifest(Path(manifest_path), root)
        else:
            all_items = collect_audio_items(data_root, datasets, limit_per_family=limit_per_family)
            if not all_items:
                raise RuntimeError(f"No audio items found under {data_root}")
            self.audio_items = split_audio_items(
                all_items,
                split=split,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.clip_seconds = clip_seconds
        self.feature_size = feature_size
        self.pseudo_domain_augment = pseudo_domain_augment
        self.num_pseudo_domains = max(1, num_pseudo_domains)
        self.noise_scale_factor = noise_scale_factor
        self.noise_scale_min = noise_scale_min
        self.noise_scale_max = noise_scale_max
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_blend = smoothing_blend
        self.frequency_mask_ratio = frequency_mask_ratio
        self.time_mask_ratio = time_mask_ratio
        self.clips: List[ClipItem] = []
        for index, item in enumerate(self.audio_items):
            count = max(1, math.ceil(item.duration / clip_seconds))
            for clip_index in range(count):
                start = clip_index * clip_seconds
                self.clips.append(
                    ClipItem(
                        audio_index=index,
                        start=start,
                        key=stable_key(
                            item.dataset,
                            item.source,
                            item.member,
                            str(start),
                            str(clip_seconds),
                            str(feature_size),
                        ),
                    )
                )

    def __len__(self) -> int:
        return len(self.clips)

    def _pseudo_domain(self, index: int) -> int:
        if not self.pseudo_domain_augment:
            return 0
        return index % self.num_pseudo_domains

    def _apply_pseudo_domain(self, features: torch.Tensor, domain: int) -> torch.Tensor:
        if domain == 0:
            return features

        x = features.clone()
        if domain == 1:
            scale = torch.clamp(
                x.std() * self.noise_scale_factor,
                min=self.noise_scale_min,
                max=self.noise_scale_max,
            )
            return torch.clamp(x + torch.randn_like(x) * scale, 0.0, 1.0)

        if domain == 2:
            padding = self.smoothing_kernel // 2
            smoothed = F.avg_pool2d(
                x.unsqueeze(0),
                kernel_size=(1, self.smoothing_kernel),
                stride=1,
                padding=(0, padding),
            ).squeeze(0)
            return torch.clamp(
                (1.0 - self.smoothing_blend) * x
                + self.smoothing_blend * smoothed,
                0.0,
                1.0,
            )

        if domain == 3:
            _, height, width = x.shape
            fill = float(x.mean())
            if self.frequency_mask_ratio > 0:
                freq_width = max(1, round(height * self.frequency_mask_ratio))
                freq_start = torch.randint(
                    0,
                    max(1, height - freq_width + 1),
                    (1,),
                ).item()
                x[:, freq_start:freq_start + freq_width, :] = fill
            if self.time_mask_ratio > 0:
                time_width = max(1, round(width * self.time_mask_ratio))
                time_start = torch.randint(
                    0,
                    max(1, width - time_width + 1),
                    (1,),
                ).item()
                x[:, :, time_start:time_start + time_width] = fill
            return x

        return x

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        clip = self.clips[index]
        item = self.audio_items[clip.audio_index]
        raw = _audio_bytes(item)
        features = cached_stacked_features(
            raw,
            cache_dir=self.cache_dir,
            cache_key=clip.key,
            start=clip.start,
            clip_seconds=self.clip_seconds,
            size=self.feature_size,
        )
        domain = self._pseudo_domain(index)
        features = self._apply_pseudo_domain(features, domain)
        return {
            "input": features,
            "label": torch.tensor(item.label, dtype=torch.long),
            "domain": torch.tensor(domain, dtype=torch.long),
            "dataset": item.dataset,
            "file_id": audio_file_id(item),
            "source": item.source,
            "member": item.member,
            "clip_start": torch.tensor(clip.start, dtype=torch.float32),
            "sample_id": stable_key(audio_file_id(item), str(clip.start)),
        }


def describe_items(items: Sequence[AudioItem]) -> Dict[str, int]:
    counts = {name: 0 for name in FAMILY_NAMES}
    for item in items:
        counts[item.family] += 1
    return counts
