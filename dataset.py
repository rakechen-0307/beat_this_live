import sys
import json
import math
import random
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soxr
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from beat_this.preprocessing import LogMelSpect

try:
    import soundfile as sf
except ImportError:
    sf = None
import torchaudio


def _disable_stft_centering(spect_module: torch.nn.Module) -> None:
    """Match BeatThisLiveActivation's strictly causal STFT (center=False)."""
    for m in spect_module.modules():
        if hasattr(m, "center"):
            m.center = False


def load_beats_tsv(tsv_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Read '<time_sec>\\t<beat_pos>' where beat_pos == 1 indicates a downbeat."""
    times, positions = [], []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            times.append(float(parts[0]))
            positions.append(int(float(parts[1])) if len(parts) >= 2 else 0)
    return np.asarray(times, dtype=np.float64), np.asarray(positions, dtype=np.int32)


def _probe_wav_info(wav_path: pathlib.Path) -> tuple[int, int]:
    if sf is not None:
        info = sf.info(str(wav_path))
        return int(info.frames), int(info.samplerate)
    info = torchaudio.info(str(wav_path))
    return int(info.num_frames), int(info.sample_rate)


def read_wav_segment(
    wav_path: pathlib.Path, start_sample: int = 0, num_samples: Optional[int] = None
) -> tuple[np.ndarray, int]:
    """Fast mono float32 windowed WAV read using soundfile (with torchaudio fallback)."""
    if sf is not None:
        stop = None if num_samples is None else (start_sample + num_samples)
        data, sr = sf.read(str(wav_path), start=start_sample, stop=stop, dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        return np.ascontiguousarray(mono, dtype=np.float32), int(sr)

    kwargs = {"frame_offset": start_sample}
    if num_samples is not None:
        kwargs["num_frames"] = num_samples
    wav, sr = torchaudio.load(str(wav_path), **kwargs)
    return np.ascontiguousarray(wav.mean(dim=0).numpy(), dtype=np.float32), int(sr)


def compute_active_mask_50fps(
    x22: np.ndarray,
    n_frames: int,
    n_fft: int = 1024,
    hop_22k: int = 441,
    fps: float = 50.0,
    silence_db: float = -45.0,
    hold_sec: float = 1.5,
) -> np.ndarray:
    """
    Build a frame-level float32 activity mask (1.0 = active or within hold_sec of active
    guitar audio, 0.0 = sustained multi-bar silence / tacet).
    Aligned with causal LogMelSpect(center=False) padded with n_fft//2 zeros on both sides.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    pad = n_fft // 2
    x_pad = np.pad(x22.astype(np.float32, copy=False), (pad, pad), mode="constant")
    needed = (n_frames - 1) * hop_22k + n_fft
    if x_pad.size < needed:
        x_pad = np.pad(x_pad, (0, needed - x_pad.size), mode="constant")

    # Vectorized frame RMS via cumulative sum of squares (O(N), no copies)
    sq = np.concatenate([[0.0], np.cumsum(x_pad.astype(np.float64) ** 2)])
    starts = np.arange(n_frames, dtype=np.int64) * hop_22k
    frame_energy = np.maximum((sq[starts + n_fft] - sq[starts]) / float(n_fft), 1e-12)
    rms_db = 10.0 * np.log10(frame_energy)

    active = rms_db > float(silence_db)
    hold_frames = max(0, int(round(hold_sec * fps)))
    if hold_frames > 0 and active.any():
        # Hold active=True for up to hold_frames after the last non-silent frame
        out = np.zeros(n_frames, dtype=bool)
        rem = 0
        for i in range(n_frames):
            if active[i]:
                rem = hold_frames
                out[i] = True
            elif rem > 0:
                rem -= 1
                out[i] = True
        return out.astype(np.float32)

    return active.astype(np.float32)


def compute_causal_logmel(
    x22: np.ndarray, spect: LogMelSpect, n_fft: int = 1024
) -> torch.Tensor:
    """
    Compute 50 FPS LogMel spectrogram with exact parity to BeatThisLiveActivation
    (center=False + n_fft//2 zero padding so frame m is centered at m * 441 / 22050 s).
    """
    pad = np.zeros(n_fft // 2, dtype=np.float32)
    x_in = np.concatenate([pad, np.ascontiguousarray(x22, dtype=np.float32), pad])
    with torch.inference_mode():
        mel = spect(torch.from_numpy(x_in))  # (T, 128)
    return mel.float()


@dataclass
class PerformanceItem:
    dataset_name: str
    track_id: str
    guitar_paths: list[pathlib.Path]
    beats_path: pathlib.Path
    length_samples: int
    sr: int
    weight: float


def discover_dataset_items(datasets_cfg: dict, split: str) -> list[PerformanceItem]:
    """Scan processed/{split} across Slakh2100, GuitarSet, and EGDB-PG."""
    items: list[PerformanceItem] = []

    for ds_name, ds_info in datasets_cfg.items():
        root = pathlib.Path(ds_info["root"])
        weight = float(ds_info.get("weight", 1.0))
        split_dir = root / split
        if not split_dir.exists():
            print(f"[WARN] Split directory not found for {ds_name}/{split}: {split_dir}")
            continue

        meta_path = root / "metadata" / f"{split}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        track_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        n_kept = 0
        for td in track_dirs:
            beats_path = td / "beats.tsv"
            if not beats_path.exists():
                continue
            guitar_paths = sorted(td.glob("guitar_*.wav"))
            if not guitar_paths:
                continue

            m = meta.get(td.name, {})
            if "length" in m and "sr" in m:
                length_samples = int(m["length"])
                sr = int(m["sr"])
            else:
                length_samples, sr = _probe_wav_info(guitar_paths[0])

            if length_samples < sr:  # skip degenerate <1 s clips
                continue

            items.append(
                PerformanceItem(
                    dataset_name=ds_name,
                    track_id=td.name,
                    guitar_paths=guitar_paths,
                    beats_path=beats_path,
                    length_samples=length_samples,
                    sr=sr,
                    weight=weight,
                )
            )
            n_kept += 1

        print(f"[DATA] {split:10s} | {ds_name:10s} | {n_kept:4d} performances (weight={weight:.2f})")

    if not items:
        raise RuntimeError(f"No valid guitar performances with beats.tsv found for split '{split}'.")
    return items


class MultiGuitarBeatTrainDataset(Dataset):
    """
    Training dataset over Slakh2100 + GuitarSet + EGDB-PG solo-guitar stems.
    Returns fixed-length (train_length, 128) causal LogMel spectrograms,
    label_shift-aligned binary beat/downbeat targets, and silence-aware loss masks.
    """

    def __init__(
        self,
        items: list[PerformanceItem],
        label_shift: int,
        data_cfg: dict,
        silence_cfg: dict,
        augment_cfg: dict,
    ):
        self.items = items
        self.label_shift = int(label_shift)
        self.fps = float(data_cfg.get("fps", 50.0))
        self.sr_22k = int(data_cfg.get("sr_22k", 22050))
        self.n_fft = int(data_cfg.get("n_fft", 1024))
        self.hop_22k = int(data_cfg.get("hop_22k", 441))
        self.soxr_quality = str(data_cfg.get("soxr_quality", "LQ"))
        self.train_length = int(data_cfg.get("train_length", 1536))
        self.target_sec = self.train_length / self.fps

        self.sil_enabled = bool(silence_cfg.get("enabled", True))
        self.silence_db = float(silence_cfg.get("silence_db", -45.0))
        self.hold_sec = float(silence_cfg.get("hold_sec", 1.5))
        self.edge_ignore = int(silence_cfg.get("edge_ignore_frames", 25))
        self.min_active_frac = float(silence_cfg.get("min_active_frac", 0.20))

        self.aug = dict(augment_cfg or {})
        self.aug_enabled = bool(self.aug.get("enabled", True))

        # Lazily instantiated per DataLoader worker (safe on Windows spawn)
        self._spect: Optional[LogMelSpect] = None
        self._mel_silence: Optional[torch.Tensor] = None

    def _ensure_spect(self) -> None:
        if self._spect is None:
            self._spect = LogMelSpect()
            _disable_stft_centering(self._spect)
            with torch.no_grad():
                probe = self._spect(torch.zeros(self.n_fft, dtype=torch.float32))
            self._mel_silence = probe[-1].detach().float()  # (128,)

    @property
    def mel_silence(self) -> torch.Tensor:
        self._ensure_spect()
        return self._mel_silence

    def __len__(self) -> int:
        return len(self.items)

    def build_weighted_sampler(self, num_samples: int) -> WeightedRandomSampler:
        """Balance per-epoch sampling strictly according to dataset weights."""
        by_ds: dict[str, int] = {}
        for it in self.items:
            by_ds[it.dataset_name] = by_ds.get(it.dataset_name, 0) + 1
        sample_weights = [it.weight / float(by_ds[it.dataset_name]) for it in self.items]
        return WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(num_samples),
            replacement=True,
        )

    def _apply_waveform_augment(self, x22: np.ndarray) -> np.ndarray:
        if not self.aug_enabled or x22.size == 0:
            return x22

        # 1. Random gain (in dB)
        gain_db = random.uniform(
            float(self.aug.get("gain_db_min", -12.0)),
            float(self.aug.get("gain_db_max", 6.0)),
        )
        x = x22 * float(10.0 ** (gain_db / 20.0))

        # 2. 1-pole spectral tilt (mic/pickup tone variation)
        if random.random() < float(self.aug.get("tilt_prob", 0.5)):
            alpha = random.uniform(
                float(self.aug.get("tilt_alpha_min", -0.6)),
                float(self.aug.get("tilt_alpha_max", 0.6)),
            )
            prev = np.empty_like(x)
            prev[0] = 0.0
            prev[1:] = x[:-1]
            x = x - alpha * prev

        # 3. Mild amp/pedal soft-clipping (tanh overdrive, RMS-matched)
        if random.random() < float(self.aug.get("overdrive_prob", 0.35)):
            rms_before = float(np.sqrt(np.mean(x ** 2) + 1e-10))
            if rms_before > 1e-4:
                drive = random.uniform(
                    float(self.aug.get("overdrive_min", 1.5)),
                    float(self.aug.get("overdrive_max", 6.0)),
                )
                y = np.tanh(drive * (x / (3.0 * rms_before + 1e-8)))
                rms_after = float(np.sqrt(np.mean(y ** 2) + 1e-10))
                x = y * (rms_before / rms_after)

        # 4. Low-level room / audio-interface noise floor
        if random.random() < float(self.aug.get("noise_prob", 0.5)):
            sig_rms = float(np.sqrt(np.mean(x ** 2) + 1e-10))
            if sig_rms > 1e-4:
                snr_db = random.uniform(
                    float(self.aug.get("noise_snr_min", 25.0)),
                    float(self.aug.get("noise_snr_max", 45.0)),
                )
                noise_rms = sig_rms / float(10.0 ** (snr_db / 20.0))
                x = x + np.random.normal(0.0, noise_rms, size=x.shape).astype(np.float32)

        return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)

    def _apply_spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        if not self.aug_enabled:
            return mel
        if random.random() < float(self.aug.get("spec_freq_mask_prob", 0.30)):
            max_w = int(self.aug.get("spec_freq_mask_width", 16))
            n_bins = mel.shape[-1]
            w = random.randint(1, min(max_w, n_bins - 1))
            f0 = random.randint(0, n_bins - w)
            mel = mel.clone()
            mel[:, f0 : f0 + w] = self._mel_silence[f0 : f0 + w]
        return mel

    def __getitem__(self, idx: int) -> dict:
        self._ensure_spect()
        item = self.items[idx]
        wav_path = random.choice(item.guitar_paths)
        beats, positions = load_beats_tsv(item.beats_path)
        has_downbeats = bool((positions == 1).any())

        # Choose speed perturbation (single-pass soxr resampling gives exact beat scaling)
        if self.aug_enabled and random.random() < float(self.aug.get("speed_prob", 0.70)):
            speed = random.uniform(
                float(self.aug.get("speed_min", 0.88)),
                float(self.aug.get("speed_max", 1.12)),
            )
        else:
            speed = 1.0

        # Add small read buffer so frame count after soxr always covers train_length
        raw_needed_samples = int(math.ceil((self.target_sec + 0.10) * speed * item.sr))
        max_start = max(0, item.length_samples - raw_needed_samples)

        # Retry random crop offsets if we land inside a long silent guitar tacet (Slakh2100)
        best_crop = None
        best_active_frac = -1.0
        n_tries = 6 if max_start > 0 else 1

        for _ in range(n_tries):
            start_sample = random.randint(0, max_start) if max_start > 0 else 0
            x_raw, sr = read_wav_segment(
                wav_path,
                start_sample=start_sample,
                num_samples=min(raw_needed_samples, item.length_samples - start_sample),
            )
            in_rate = max(1000, int(round(sr * speed)))
            actual_speed = in_rate / float(sr)
            x22 = soxr.resample(x_raw, in_rate, self.sr_22k, quality=self.soxr_quality).astype(np.float32)

            max_22k_samples = self.train_length * self.hop_22k
            if x22.size > max_22k_samples:
                x22 = x22[:max_22k_samples]

            n_valid = min(self.train_length, 1 + x22.size // self.hop_22k)
            active_mask = compute_active_mask_50fps(
                x22,
                n_frames=n_valid,
                n_fft=self.n_fft,
                hop_22k=self.hop_22k,
                fps=self.fps,
                silence_db=self.silence_db,
                hold_sec=self.hold_sec,
            )
            act_frac = float(active_mask.mean()) if n_valid > 0 else 0.0
            if act_frac > best_active_frac:
                best_active_frac = act_frac
                best_crop = (start_sample, sr, actual_speed, x22, n_valid, active_mask)
            if act_frac >= self.min_active_frac:
                break

        start_sample, sr, actual_speed, x22, n_valid, active_mask = best_crop
        x22_aug = self._apply_waveform_augment(x22)

        mel = compute_causal_logmel(x22_aug, self._spect, n_fft=self.n_fft)  # (T_mel, 128)
        n_valid = min(self.train_length, mel.shape[0])
        mel = mel[:n_valid]
        active_mask = active_mask[:n_valid]

        # Pad short clips (e.g. <30.72 s GuitarSet/EGDB excerpts) with mel_silence on the right
        if n_valid < self.train_length:
            pad_len = self.train_length - n_valid
            pad_mel = self._mel_silence.unsqueeze(0).expand(pad_len, -1)
            mel = torch.cat([mel, pad_mel], dim=0)

        mel = self._apply_spec_augment(mel)

        # Build shifted beat & downbeat binary target sequences
        # Model output frame m predicts beat at time (m - label_shift) / fps = (m + 3) / 50.0
        # => ground-truth beat at crop-relative time t_rel lands at m = round(t_rel * fps) + label_shift
        t_start_sec = start_sample / float(sr)
        rel_beats = (beats - t_start_sec) / actual_speed
        beat_frames = np.rint(rel_beats * self.fps).astype(np.int64) + self.label_shift

        beat_target = np.zeros(self.train_length, dtype=np.float32)
        db_target = np.zeros(self.train_length, dtype=np.float32)

        valid_b = (beat_frames >= 0) & (beat_frames < n_valid)
        if valid_b.any():
            beat_target[beat_frames[valid_b]] = 1.0
            if has_downbeats:
                db_sel = valid_b & (positions == 1)
                if db_sel.any():
                    db_target[beat_frames[db_sel]] = 1.0

        # Build loss mask: valid frames * silence mask * edge ignore (on mid-song crops)
        mask = np.zeros(self.train_length, dtype=np.float32)
        mask[:n_valid] = active_mask if self.sil_enabled else 1.0
        if start_sample > 0 and self.edge_ignore > 0:
            mask[: min(self.edge_ignore, n_valid)] = 0.0
        # Exclude tail frames whose shifted target falls past the end of the clip
        lead = max(0, -self.label_shift)
        if lead > 0 and n_valid > lead:
            mask[n_valid - lead : n_valid] = 0.0

        db_mask = mask.copy() if has_downbeats else np.zeros_like(mask)

        return {
            "mel": mel,                                          # (T, 128)
            "beat_target": torch.from_numpy(beat_target),        # (T,)
            "downbeat_target": torch.from_numpy(db_target),      # (T,)
            "beat_mask": torch.from_numpy(mask),                 # (T,)
            "downbeat_mask": torch.from_numpy(db_mask),          # (T,)
            "dataset": item.dataset_name,
        }


class MultiGuitarBeatValDataset(Dataset):
    """
    Full-track deterministic validation dataset (no cropping, no augmentation).
    Matches BeatThisLiveOffline(causal_stft=True) and live BeatThisLiveActivation.
    """

    def __init__(
        self,
        items: list[PerformanceItem],
        data_cfg: dict,
        eval_cfg: dict,
        max_renders_per_track: Optional[int] = 2,
    ):
        self.fps = float(data_cfg.get("fps", 50.0))
        self.sr_22k = int(data_cfg.get("sr_22k", 22050))
        self.n_fft = int(data_cfg.get("n_fft", 1024))
        self.hop_22k = int(data_cfg.get("hop_22k", 441))
        self.soxr_quality = str(data_cfg.get("soxr_quality", "LQ"))
        self.sil_db = float(eval_cfg.get("active_eval_silence_db", -45.0))
        self.hold_sec = float(eval_cfg.get("active_eval_hold_sec", 1.5))

        self.entries: list[tuple[str, str, pathlib.Path, pathlib.Path]] = []
        for it in items:
            paths = it.guitar_paths
            if max_renders_per_track is not None and len(paths) > max_renders_per_track:
                # Pick evenly spaced renders (e.g. cleanest + dirtiest preset on EGDB)
                sel = np.linspace(0, len(paths) - 1, int(max_renders_per_track), dtype=int)
                paths = [paths[i] for i in sel]
            for gp in paths:
                self.entries.append((it.dataset_name, f"{it.track_id}/{gp.stem}", gp, it.beats_path))

        self._spect: Optional[LogMelSpect] = None
        self._mel_silence: Optional[LogMelSpect] = None

    def _ensure_spect(self) -> None:
        if self._spect is None:
            self._spect = LogMelSpect()
            _disable_stft_centering(self._spect)
            with torch.no_grad():
                probe = self._spect(torch.zeros(self.n_fft, dtype=torch.float32))
            self._mel_silence = probe[-1].detach().float()  # (128,)

    @property
    def mel_silence(self) -> torch.Tensor:
        self._ensure_spect()
        return self._mel_silence

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        self._ensure_spect()
        ds_name, track_key, wav_path, beats_path = self.entries[idx]
        x_raw, sr = read_wav_segment(wav_path)
        x22 = soxr.resample(x_raw, sr, self.sr_22k, quality=self.soxr_quality).astype(np.float32)
        mel = compute_causal_logmel(x22, self._spect, n_fft=self.n_fft)  # (T, 128)
        active_mask = compute_active_mask_50fps(
            x22,
            n_frames=mel.shape[0],
            n_fft=self.n_fft,
            hop_22k=self.hop_22k,
            fps=self.fps,
            silence_db=self.sil_db,
            hold_sec=self.hold_sec,
        )
        beats, positions = load_beats_tsv(beats_path)
        downbeats = beats[positions == 1]

        return {
            "dataset": ds_name,
            "track_key": track_key,
            "mel": mel,
            "active_mask": active_mask,
            "beats": beats,
            "downbeats": downbeats,
        }