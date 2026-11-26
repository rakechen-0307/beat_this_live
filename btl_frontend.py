"""
btl_frontend.py -- Beat This! Live (50 FPS) -> 30 FPS frontend for RealTimeBeatTracker.

Provides:
  1. BeatThisLiveOffline: whole-stem causal extraction at 50 FPS, aligned by each checkpoint's
     `label_shift` and resampled onto the 30 FPS tracker grid t(n) = n/30 + target_lead_ms/1000.
  2. BeatThisLiveActivation: strictly causal 30 FPS (48 kHz, H=1600) drop-in replacement for
     BeatActivation in RealTimeBeatTracker, batching the 1-2 new 50 FPS mel frames produced per
     30 FPS audio hop into a single model forward pass using a zero-copy double-length pinned
     ring buffer, automatic soxr group-delay compensation, and optional fixed-shape CUDA graph capture.
  3. CausalFluxBTLActivation: strictly causal live-streaming fusion of BeatActivation (spectral
     flux) and BeatThisLiveActivation supporting:
       - "boost" (recommended): btl[n] * (1 + flux_boost * min(flux[n] / ema_flux[n], cap))
       - "btl_only": passes BTL straight through.
       - "mul" (legacy): flux[n] * (eps + max_{k in [-past, 0]} btl[n + k]).
"""
from contextlib import contextmanager
import os
import pathlib
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _base in (_here.parent, *_here.parents, Path.cwd()):
    _cand = _base / "beat_this_live"
    if (_cand / "beat_this").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Optional

import numpy as np
import soxr
import torch

from beat_this.inference import load_model
from beat_this.preprocessing import LogMelSpect

# Measured algorithmic filter latency (ms) of soxr.ResampleStream (48 kHz -> 22.05 kHz)
SOXR_LATENCY_MS_48K_TO_22K = {
    "QQ": 0.0,
    "LQ": 6.8,
    "HQ": 18.6,
    "VHQ": 27.9,
    "MQ": 35.9,
}


@contextmanager
def _posix_path_compat():
    """Allow Linux-saved checkpoints containing pickled PosixPath objects to load on Windows."""
    if os.name != "nt":
        yield
        return
    orig_posix = pathlib.PosixPath
    local_mod = getattr(pathlib, "_local", None)
    orig_local = getattr(local_mod, "PosixPath", None) if local_mod is not None else None
    pathlib.PosixPath = pathlib.WindowsPath
    if orig_local is not None:
        local_mod.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = orig_posix
        if orig_local is not None:
            local_mod.PosixPath = orig_local


def _load_btl_model(ckpt_path: str, device: str):
    with _posix_path_compat():
        return load_model(str(ckpt_path), device=device, return_hparams=True)


def _hp_get(hp: Any, keys: tuple[str, ...], default: Any) -> Any:
    for k in keys:
        if isinstance(hp, dict) and k in hp and hp[k] is not None:
            return hp[k]
        if hasattr(hp, k) and getattr(hp, k) is not None:
            return getattr(hp, k)
    return default


@dataclass(frozen=True)
class BTLCheckpointSpec:
    """Timing and context metadata read from a Beat This! Live checkpoint.

    Note: Beat This! Live's `lookahead` (sclm_lookahead_size) belongs solely to its
    SCLM peak-picking postprocessor, which is bypassed here in favor of PLP.
    Only `label_shift` affects the timing of raw frame activations.
    """
    ckpt_path: str
    fps: float = 50.0
    label_shift: int = 0        # negative in CPJKU checkpoints: -1 (Live/Large), -3 (Zero)
    sw_window: int = 512        # 512 for Live/Large, 256 for Zero
    n_fft: int = 1024
    hop_22k: int = 441
    sr_22k: int = 22050

    @property
    def stft_half_window_sec(self) -> float:
        return (self.n_fft // 2) / float(self.sr_22k)  # 512 / 22050 = 23.22 ms

    @property
    def lead_frames(self) -> int:
        """How many 50 FPS frames ahead raw logit `m` predicts (+1 Live/Large, +3 Zero, <0 past)."""
        return -self.label_shift

    def causal_margin_ms_at_30fps(
        self,
        target_lead_ms: float = 0.0,
        resampler_latency_ms: float = 6.8,
        method: str = "max",
        arbitrary_phase: bool = False,
    ) -> float:
        """Worst-case causal margin (ms) at 30 FPS step n (t_now = (n+1)/30)."""
        if arbitrary_phase:
            fut = (1000.0 / self.fps) if method == "linear" else (0.5 * 1000.0 / 30.0)
        else:
            fut = (4.0 / self.fps - 2.0 / 30.0) * 1000.0  # 13.333 ms exact at 0-phase mod 20 ms
        needed_past_target_ms = (
            fut
            - (self.lead_frames / self.fps) * 1000.0
            + self.stft_half_window_sec * 1000.0
            + resampler_latency_ms
        )
        return (1000.0 / 30.0) - target_lead_ms - needed_past_target_ms


def inspect_btl_hparams(ckpt_path: str, hp: Any) -> BTLCheckpointSpec:
    raw_shift = _hp_get(hp, ("label_shift",), None)
    if raw_shift is None:
        raise KeyError(
            f"'label_shift' not found in checkpoint hparams for {ckpt_path}; "
            f"available keys: {list(hp.keys()) if isinstance(hp, dict) else dir(hp)}"
        )
    return BTLCheckpointSpec(
        ckpt_path=str(ckpt_path),
        fps=float(_hp_get(hp, ("fps", "frame_rate"), 50.0)),
        label_shift=int(raw_shift),
        sw_window=int(_hp_get(hp, ("sw_attention_window_size", "attention_window", "window_size"), 512)),
    )


def _disable_stft_centering(spect_module: torch.nn.Module) -> None:
    """Switch internal torchaudio STFT/MelSpectrogram modules to strictly causal (center=False)."""
    for m in spect_module.modules():
        if hasattr(m, "center"):
            m.center = False


def resample_activation_50_to_30(
    act50: np.ndarray,
    n_frames_30: int,
    spec: BTLCheckpointSpec,
    fps_out: float = 30.0,
    align_label_shift: bool = True,
    target_lead_ms: float = 0.0,
    extra_shift_ms: float = 0.0,
    method: str = "max",
    act_power: float = 1.0,
) -> np.ndarray:
    """
    Map a 50 FPS activation array `act50` (frame m centered at m / spec.fps) onto the
    30 FPS grid t_target(n) = n / fps_out + target_lead_ms / 1000.0.
    """
    a50 = np.asarray(act50, dtype=np.float64)
    if a50.size == 0 or n_frames_30 <= 0:
        return np.zeros(max(n_frames_30, 0), dtype=np.float32)

    if act_power != 1.0:
        a50 = np.power(np.clip(a50, 0.0, None), act_power)

    lead = spec.lead_frames if align_label_shift else 0
    m = np.arange(a50.size, dtype=np.float64)
    t50 = (m + lead) / spec.fps + (extra_shift_ms / 1000.0)
    t30 = np.arange(n_frames_30, dtype=np.float64) / fps_out + (target_lead_ms / 1000.0)

    if method == "linear":
        a30 = np.interp(t30, t50, a50, left=0.0, right=0.0)
    elif method in ("max", "avg"):
        half_bin = 0.5 / fps_out
        lo = np.searchsorted(t50, t30 - half_bin, side="left")
        hi = np.searchsorted(t50, t30 + half_bin, side="right")
        a30 = np.zeros(n_frames_30, dtype=np.float64)
        for n in range(n_frames_30):
            if hi[n] > lo[n]:
                seg = a50[lo[n]:hi[n]]
                a30[n] = float(np.max(seg) if method == "max" else np.mean(seg))
            elif 0 <= lo[n] < a50.size:
                a30[n] = float(np.interp(t30[n], t50, a50, left=0.0, right=0.0))
    else:
        raise ValueError(f"Unknown resample method '{method}'; expected 'linear', 'max', or 'avg'.")

    return a30.astype(np.float32)


class BeatThisLiveOffline:
    """Whole-stem causal extractor for offline benchmarking and pre-encoding."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        align_label_shift: bool = True,
        causal_stft: bool = False,
        target_lead_ms: float = 0.0,
        extra_shift_ms: float = 0.0,
        resample_method: str = "max",
        downbeat_weight: float = 0.0,
        act_power: float = 1.0,
        soxr_quality: str = "LQ",
        use_amp: bool = False,
    ):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = str(device)
        self.align_label_shift = bool(align_label_shift)
        self.causal_stft = bool(causal_stft)
        self.target_lead_ms = float(target_lead_ms)
        self.extra_shift_ms = float(extra_shift_ms)
        self.resample_method = str(resample_method)
        self.downbeat_weight = float(downbeat_weight)
        self.act_power = float(act_power)
        self.soxr_quality = str(soxr_quality)
        self.use_amp = bool(use_amp)

        self.model, self.hparams = _load_btl_model(str(ckpt_path), device=self.device)
        self.model.eval()
        self.spec = inspect_btl_hparams(ckpt_path, self.hparams)

        self.spect = LogMelSpect()
        if self.causal_stft:
            _disable_stft_centering(self.spect)

    @torch.inference_mode()
    def extract(self, mono: np.ndarray, sr: int = 48000, hop_out: int = 1600) -> dict:
        fps_out = float(sr) / float(hop_out)
        n_frames_out = int(math.ceil(len(mono) / hop_out))

        x22 = soxr.resample(
            np.asarray(mono, dtype=np.float32), sr, self.spec.sr_22k, quality=self.soxr_quality
        )
        if self.causal_stft:
            pad = np.zeros(self.spec.n_fft // 2, dtype=np.float32)
            x_in = np.concatenate([pad, x22, pad])
            spect = self.spect(torch.from_numpy(np.ascontiguousarray(x_in)))
        else:
            spect = self.spect(torch.from_numpy(np.ascontiguousarray(x22)))

        amp_ok = self.use_amp and str(self.device).startswith("cuda")
        amp_dtype = torch.bfloat16 if (amp_ok and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.amp.autocast(device_type="cuda" if amp_ok else "cpu", dtype=amp_dtype, enabled=amp_ok):
            out = self.model(spect.unsqueeze(0).to(self.device))

        beat50 = torch.sigmoid(out["beat"][0]).float().cpu().numpy()
        db50 = torch.sigmoid(out["downbeat"][0]).float().cpu().numpy()
        comb50 = beat50 + self.downbeat_weight * db50

        act_out = resample_activation_50_to_30(
            comb50,
            n_frames_30=n_frames_out,
            spec=self.spec,
            fps_out=fps_out,
            align_label_shift=self.align_label_shift,
            target_lead_ms=self.target_lead_ms,
            extra_shift_ms=self.extra_shift_ms,
            method=self.resample_method,
            act_power=self.act_power,
        )
        db_out = resample_activation_50_to_30(
            db50,
            n_frames_30=n_frames_out,
            spec=self.spec,
            fps_out=fps_out,
            align_label_shift=self.align_label_shift,
            target_lead_ms=self.target_lead_ms,
            extra_shift_ms=self.extra_shift_ms,
            method=self.resample_method,
            act_power=1.0,
        )
        return dict(
            act=act_out,
            downbeat=db_out,
            beat50=beat50.astype(np.float32),
            label_shift=np.int32(self.spec.label_shift),
            sw_window=np.int32(self.spec.sw_window),
            fps_out=np.float32(fps_out),
        )


@dataclass
class BeatThisLiveActivation:
    """
    Strictly causal 30 FPS (48 kHz, H=1600) drop-in replacement for BeatActivation
    in RealTimeBeatTracker.

    Internally runs Beat This! Live at 50 FPS (22.05 kHz, hop=441, n_fft=1024, center=False)
    using a double-length pinned ring buffer (left-padded with mel_silence at startup so
    fixed-shape CUDA graph replay works from block 0 with zero per-step allocations or
    overlapping-memory copies).
    """
    ckpt_path: str
    samplerate: int = 48000
    H: int = 1600
    device: str = "cuda"
    use_amp: bool = False
    use_cuda_graph: bool = False
    conv_rf_margin: int = 32
    context_frames: Optional[int] = None
    align_label_shift: bool = True
    target_lead_ms: float = 0.0
    extra_shift_ms: float = 0.0
    compensate_resampler_latency: bool = True
    resample_method: str = "max"
    downbeat_weight: float = 0.0
    act_power: float = 1.0
    soxr_quality: str = "LQ"

    activation_frame: np.ndarray = field(default_factory=lambda: np.array([0.0]), init=False, repr=False)
    downbeat_frame: float = field(default=0.0, init=False, repr=False)
    n_clamped_steps: int = field(default=0, init=False, repr=False)
    _torch_dev: torch.device = field(init=False, repr=False)
    _graph: Optional[Any] = field(default=None, init=False, repr=False)
    _ctx_gpu: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _beat_out_gpu: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _db_out_gpu: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _mel_silence: torch.Tensor = field(init=False, repr=False)
    _mel_ring_buf: torch.Tensor = field(init=False, repr=False)
    _ring_pos: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self._torch_dev = torch.device(self.device)
        self.fps_out = float(self.samplerate) / float(self.H)

        self.model, self.hparams = _load_btl_model(str(self.ckpt_path), device=self.device)
        self.model.eval()
        self.spec = inspect_btl_hparams(self.ckpt_path, self.hparams)

        if self.context_frames is None:
            # +2 headroom so even the first frame of a 2-frame 50 FPS batch has full window + conv RF
            self.context_frames = self.spec.sw_window + self.conv_rf_margin + 2

        self.spect = LogMelSpect()
        _disable_stft_centering(self.spect)
        with torch.no_grad():
            probe = self.spect(torch.zeros(self.spec.n_fft, dtype=torch.float32))
        if probe.shape[0] != 1:
            raise RuntimeError(
                f"Expected causal LogMelSpect(center=False) to return 1 frame for {self.spec.n_fft} "
                f"samples, got shape {tuple(probe.shape)}."
            )
        self._mel_silence = probe[-1].detach().float().clone()
        n_mels = int(self._mel_silence.numel())

        # Double-length contiguous ring buffer: writing frame m to [pos] and [pos + C] makes
        # [pos : pos + C] a zero-copy chronological view ready for direct async H2D transfer.
        pin_mem = (self._torch_dev.type == "cuda")
        self._mel_ring_buf = torch.empty(
            (2 * self.context_frames, n_mels), dtype=torch.float32, pin_memory=pin_mem
        )
        self._mel_ring_buf.copy_(self._mel_silence.view(1, -1))
        self._ring_pos = 0

        self._resampler = soxr.ResampleStream(
            self.samplerate, self.spec.sr_22k, num_channels=1, dtype="float32", quality=self.soxr_quality
        )
        # Pre-pad n_fft // 2 zeros before t=0 so frame m window is centered at sample m * 441
        self._pending = np.zeros(self.spec.n_fft // 2, dtype=np.float32)
        self._pred_hist: deque[tuple[float, float, float]] = deque(maxlen=32)  # (t_pred, act, db)
        self._frame_idx_50 = 0
        self._step_out = 0

        if self.use_cuda_graph and self._torch_dev.type == "cuda":
            self._capture_cuda_graph()

    @torch.no_grad()
    def _capture_cuda_graph(self) -> None:
        n_mels = int(self._mel_silence.numel())
        amp_ok = self.use_amp and self._torch_dev.type == "cuda"
        amp_dtype = torch.bfloat16 if (amp_ok and torch.cuda.is_bf16_supported()) else torch.float16
        need_db = (self.downbeat_weight != 0.0)

        try:
            with torch.cuda.device(self._torch_dev):
                self._ctx_gpu = (
                    self._mel_silence.view(1, 1, n_mels)
                    .expand(1, self.context_frames, n_mels)
                    .contiguous()
                    .to(self._torch_dev)
                )
                self._beat_out_gpu = torch.empty(self.context_frames, dtype=torch.float32, device=self._torch_dev)
                self._db_out_gpu = (
                    torch.empty(self.context_frames, dtype=torch.float32, device=self._torch_dev)
                    if need_db else None
                )

                def _body():
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_ok, cache_enabled=False):
                        out = self.model(self._ctx_gpu)
                    self._beat_out_gpu.copy_(torch.sigmoid(out["beat"][0]).float())
                    if self._db_out_gpu is not None:
                        self._db_out_gpu.copy_(torch.sigmoid(out["downbeat"][0]).float())

                side = torch.cuda.Stream(device=self._torch_dev)
                side.wait_stream(torch.cuda.current_stream(self._torch_dev))
                with torch.cuda.stream(side):
                    for _ in range(3):
                        _body()
                torch.cuda.current_stream(self._torch_dev).wait_stream(side)
                torch.cuda.synchronize(self._torch_dev)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    _body()
                torch.cuda.synchronize(self._torch_dev)
                self._graph = g
        except Exception as e:
            self._graph = None
            self._ctx_gpu = None
            self._beat_out_gpu = None
            self._db_out_gpu = None
            print(f"[WARN] BTL CUDA graph capture failed ({type(e).__name__}: {e}); falling back to eager.")
            try:
                torch.cuda.synchronize(self._torch_dev)
            except Exception:
                pass

    @property
    def resampler_latency_ms(self) -> float:
        return float(SOXR_LATENCY_MS_48K_TO_22K.get(self.soxr_quality.upper(), 6.8))

    def reset(self) -> None:
        self._resampler = soxr.ResampleStream(
            self.samplerate, self.spec.sr_22k, num_channels=1, dtype="float32", quality=self.soxr_quality
        )
        self._pending = np.zeros(self.spec.n_fft // 2, dtype=np.float32)
        self._mel_ring_buf.copy_(self._mel_silence.view(1, -1))
        self._ring_pos = 0
        self._pred_hist.clear()
        self._frame_idx_50 = 0
        self._step_out = 0
        self.n_clamped_steps = 0
        self.activation_frame = np.array([0.0], dtype=np.float64)
        self.downbeat_frame = 0.0

    @torch.no_grad()
    def process(self, audio_frame: np.ndarray) -> np.ndarray:
        """Consume one 30 FPS audio frame (1600 samples at 48 kHz) and return [beat_activation]."""
        x22 = self._resampler.resample_chunk(np.asarray(audio_frame, dtype=np.float32)).reshape(-1)
        if x22.size:
            self._pending = np.concatenate([self._pending, x22])

        new_frame_indices = []
        ctx_len = self.context_frames
        while len(self._pending) >= self.spec.n_fft:
            win = self._pending[:self.spec.n_fft].copy()
            self._pending = self._pending[self.spec.hop_22k:]
            mel_1 = self.spect(torch.from_numpy(win))[-1].float()
            pos = self._ring_pos
            self._mel_ring_buf[pos].copy_(mel_1)
            self._mel_ring_buf[pos + ctx_len].copy_(mel_1)
            self._ring_pos = (pos + 1) % ctx_len
            new_frame_indices.append(self._frame_idx_50)
            self._frame_idx_50 += 1

        if new_frame_indices:
            k = len(new_frame_indices)
            ctx_view = self._mel_ring_buf[self._ring_pos : self._ring_pos + ctx_len]

            if self._graph is not None:
                with torch.cuda.device(self._torch_dev):
                    self._ctx_gpu[0].copy_(ctx_view, non_blocking=True)
                    self._graph.replay()
                    beats = self._beat_out_gpu[-k:].cpu().numpy()
                    if self._db_out_gpu is not None:
                        dbs = self._db_out_gpu[-k:].cpu().numpy()
                    else:
                        dbs = np.zeros(k, dtype=np.float32)
            else:
                spect_ctx = ctx_view.unsqueeze(0).to(
                    self._torch_dev, non_blocking=(self._torch_dev.type == "cuda")
                )
                amp_ok = self.use_amp and self._torch_dev.type == "cuda"
                amp_dtype = torch.bfloat16 if (amp_ok and torch.cuda.is_bf16_supported()) else torch.float16
                with torch.amp.autocast(device_type="cuda" if amp_ok else "cpu", dtype=amp_dtype, enabled=amp_ok):
                    out = self.model(spect_ctx)
                beats = torch.sigmoid(out["beat"][0, -k:]).float().cpu().numpy()
                if self.downbeat_weight != 0.0:
                    dbs = torch.sigmoid(out["downbeat"][0, -k:]).float().cpu().numpy()
                else:
                    dbs = np.zeros(k, dtype=np.float32)

            lead = self.spec.lead_frames if self.align_label_shift else 0
            resamp_comp = -self.resampler_latency_ms if self.compensate_resampler_latency else 0.0
            shift_sec = (self.extra_shift_ms + resamp_comp) / 1000.0

            for m_idx, p_b, p_db in zip(new_frame_indices, beats, dbs):
                t_pred = (m_idx + lead) / self.spec.fps + shift_sec
                val = float(p_b + self.downbeat_weight * p_db)
                if self.act_power != 1.0:
                    val = float(max(val, 0.0) ** self.act_power)
                self._pred_hist.append((t_pred, val, float(p_db)))

        t_target = self._step_out / self.fps_out + (self.target_lead_ms / 1000.0)
        self._step_out += 1

        act_val, db_val = self._read_history(t_target)
        self.downbeat_frame = db_val
        self.activation_frame = np.array([act_val], dtype=np.float64)
        return self.activation_frame

    def _read_history(self, t_target: float) -> tuple[float, float]:
        if not self._pred_hist:
            return 0.0, 0.0

        ts = np.fromiter((h[0] for h in self._pred_hist), dtype=np.float64)
        vs = np.fromiter((h[1] for h in self._pred_hist), dtype=np.float64)
        ds = np.fromiter((h[2] for h in self._pred_hist), dtype=np.float64)

        if t_target < ts[0] - 0.5 / self.spec.fps:
            return 0.0, 0.0
        if t_target > ts[-1] + 1e-9:
            self.n_clamped_steps += 1
            return float(vs[-1]), float(ds[-1])

        if self.resample_method in ("max", "avg"):
            half_bin = 0.5 / self.fps_out
            if ts[-1] + (1.0 / self.spec.fps) <= t_target + half_bin + 1e-9:
                self.n_clamped_steps += 1
                return float(vs[-1]), float(ds[-1])
            mask = (ts >= t_target - half_bin) & (ts <= t_target + half_bin)
            if mask.any():
                if self.resample_method == "max":
                    return float(np.max(vs[mask])), float(np.max(ds[mask]))
                return float(np.mean(vs[mask])), float(np.mean(ds[mask]))

        return float(np.interp(t_target, ts, vs)), float(np.interp(t_target, ts, ds))


@dataclass
class CausalFluxBTLActivation:
    """
    Strictly causal live-streaming fusion of spectral flux and Beat This! Live.

    Modes:
      - "boost" (recommended):
            act[n] = btl[n] * (1.0 + flux_boost * min(flux[n] / ema_flux[n], flux_norm_cap))
      - "btl_only":
            act[n] = btl[n]
      - "mul" (legacy):
            act[n] = flux[n] * (eps + max_{k in [-past_frames, 0]} btl[n + k])
    """
    btl_activation: BeatThisLiveActivation
    flux_activation: Any
    fusion_mode: str = "boost"
    flux_boost: float = 0.6
    flux_ema_alpha: float = 0.01
    flux_norm_cap: float = 2.0
    eps: float = 0.05
    past_frames: int = 2
    fut_frames: int = 0

    activation_frame: np.ndarray = field(default_factory=lambda: np.array([0.0]), init=False, repr=False)

    def __post_init__(self):
        if self.fusion_mode not in ("boost", "btl_only", "mul"):
            raise ValueError(f"Unknown fusion_mode '{self.fusion_mode}'; expected 'boost', 'btl_only', or 'mul'.")
        if self.past_frames < 0:
            raise ValueError("past_frames must be >= 0.")
        if self.fut_frames != 0:
            raise ValueError("fut_frames must be 0 for zero-delay causal parity with offline fusion.")
        self._btl_hist: deque[float] = deque(maxlen=self.past_frames + 1)
        self._flux_ema: float = 0.0

    def reset(self) -> None:
        if hasattr(self.btl_activation, "reset"):
            self.btl_activation.reset()
        if hasattr(self.flux_activation, "reset"):
            self.flux_activation.reset()
        self._btl_hist.clear()
        self._flux_ema = 0.0
        self.activation_frame = np.array([0.0], dtype=np.float64)

    def process(self, audio_frame: np.ndarray) -> np.ndarray:
        b_out = self.btl_activation.process(audio_frame)
        b_val = max(float(b_out[0]) if np.asarray(b_out).size else 0.0, 0.0)

        if self.fusion_mode == "btl_only":
            self.activation_frame = np.array([b_val], dtype=np.float64)
            return self.activation_frame

        f_out = self.flux_activation.process(audio_frame)
        f_val = max(float(f_out[0]) if np.asarray(f_out).size else 0.0, 0.0)

        if self.fusion_mode == "boost":
            if f_val > 1e-6:
                if self._flux_ema <= 0.0:
                    self._flux_ema = f_val
                else:
                    self._flux_ema = (1.0 - self.flux_ema_alpha) * self._flux_ema + self.flux_ema_alpha * f_val
                f_norm = min(f_val / max(self._flux_ema, 1e-6), self.flux_norm_cap)
            else:
                f_norm = 0.0
            fused = b_val * (1.0 + self.flux_boost * f_norm)
            self.activation_frame = np.array([fused], dtype=np.float64)
            return self.activation_frame

        # Legacy multiplicative gate ("mul")
        self._btl_hist.append(b_val)
        btl_gate = max(self._btl_hist) if self._btl_hist else 0.0
        fused = f_val * (self.eps + btl_gate)
        self.activation_frame = np.array([fused], dtype=np.float64)
        return self.activation_frame