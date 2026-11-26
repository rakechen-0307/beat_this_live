"""
compare_diag.py  -- diagnostic evaluation of the real-time PLP beat tracker on guitar stems.

Aligned with live inference defaults in streaming.py:
  - Beat This! Live + non-destructive causal Flux boost (btl_fusion_mode='boost', btl_flux_boost=0.6)
  - Causal STFT enabled by default (--btl_centered_stft to disable)
  - Calibrated timing alignment (btl_extra_shift_ms=-6.0, phase_lead_ms=0.0, harmonic_weight=0.0, energy_threshold=0.015)
  - Updated PLP lock-in settings (lock_conf_threshold=1.8, acquire_min_active_sec=2.0, lock_sec=2.0, lock_tol_bpm=3.0)
  - Supports Slakh2100, GuitarSet, and EGDB (prefers beats.tsv, falls back to all_src.mid).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import math
import hashlib
import pathlib
import argparse
import warnings
import concurrent.futures

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pretty_midi
import mir_eval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter, maximum_filter1d
from tqdm import tqdm

from realtimeplp_new import BeatActivation, RealTimeBeatTracker

warnings.filterwarnings("ignore", module="mir_eval")

# === Configuration (defaults, all overridable on the command line) ===
DEFAULT_SPLIT = "test"
DEFAULT_PROCESSED = "/data/ssd0/rakec/slakh2100/processed"
DEFAULT_OUT_DIR = "/data/hdd0/rakec/test/realtimeplp_eval_diag"

MIN_GRID_SCORE = 1.3
SKIP_STATUSES = ("bad_midi", "no_midi", "bad_tempo", "tempo_glitch")

DURATION_SEC = 30
STRIDE_SEC = 15
TRIM_SEC = 5.0
LAG_MAX_FRAMES = 6
MIN_EVENT_GAP_FRAMES = 6
GT_MIN_BEAT_GAP_S = 0.15
LEVELS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
PLV_OK = 0.5
N_PLOT_WORST = 30
N_PLOT_BEST = 20
N_PLOT_PER_CAUSE = 8
CACHE_VERSION = "diag_v3"

# --- diagnostics ---
TOLS = (0.05, 0.07, 0.10, 0.15, 0.20)
SHIFTS_MS = np.arange(-150, 151, 10)
OFFSET_CAP_S = 0.25
RESUME_SEC = 4.0
REGIONS = ("silent", "resume", "steady")
ONSET_REL_THR = 0.15
ORACLE_SIGMA_S = 0.02
JUMP_BPM = 6.0
CAUSES = ["ok", "mostly_silent", "never_acquired", "offbeat_phase", "double_time", "half_time",
          "tempo_wrong", "constant_latency", "timing_jitter", "phase_lost"]
REQ_KEYS = ("sin", "cos", "tempo", "act", "flag", "active", "lock", "fps", "num_frames")

DEFAULT_BEAT_CFG = {
    "features": ["beat_sin", "beat_cos", "tempo_norm", "conf", "lock_score"],
    "tempo_scale": 180.0,
    "tempo_clip_bpm": 360.0,
    "low_bpm": 90,
    "high_bpm": 179,
    "lookahead": 7,
    "n_time": 10,
    "silence_db": -40.0,            # matches live streaming.py (use --silence_db -45.0 for clean DI/Slakh)
    "hold_frames": 36,
    "conf_threshold": 1.6,
    "lock_conf_threshold": 1.8,     # updated inference lock confidence threshold
    "switch_ratio": 1.3,
    "energy_threshold": 0.015,
    "lock_enable": True,
    "lock_sec": 2.0,                # updated inference lock duration
    "lock_tol_bpm": 3.0,            # updated inference lock tolerance
    "lose_rho": 0.5,
    "lose_conf": 1.15,
    "lose_err": 0.4,
    "phase_gate": 0.25,
    "reacquire_sec": 3.0,
    "pll_tau_sec": 1.5,
    "pll_conf_hi": 3.0,
    "switch_persist_sec": 1.5,
    "harmonic_weight": 0.0,
    "acquire_min_active_sec": 2.0,  # updated inference tempogram warm-up before lock counting
    "relock_sec": 2.5,
    "relock_coh": 0.55,
    "relock_min_w": 0.15,
    "resume_gate_sec": 3.0,
    "silence_reopen_sec": 1.5,
    "resume_gain": 3.0,
    "phase_lead_ms": 0.0,           # BTL runs default to 0.0 ms (matching streaming.py); oracle stays 0.0
    "hop": 1600,
}

# Parameters passed into RealTimeBeatTracker.from_args via **kwargs
LOCK_KEYS = [
    "lock_enable", "lock_sec", "lock_tol_bpm", "lose_rho", "lose_conf", "lose_err", "phase_gate",
    "reacquire_sec", "pll_tau_sec", "pll_conf_hi", "switch_persist_sec", "harmonic_weight",
    "acquire_min_active_sec", "relock_sec", "relock_coh", "relock_min_w",
    "resume_gate_sec", "silence_reopen_sec", "resume_gain",
]

# Full set of tracker settings hashed into cache filenames
TRACKER_KEYS = [
    "hop", "n_time", "low_bpm", "high_bpm", "lookahead", "silence_db",
    "hold_frames", "conf_threshold", "lock_conf_threshold", "switch_ratio",
    "energy_threshold", "phase_lead_ms",
] + LOCK_KEYS


def fkey(t):
    return f"F{int(round(t * 1000))}"


def oracle_names(cfg):
    return [f"orc{j:g}" for j in cfg["oracle_jitters"]]


def compare_names(cfg):
    names = []
    if cfg.get("btl_ckpt") and not cfg.get("btl_primary"):
        names.append("btl")
    names.extend(oracle_names(cfg))
    return names


def btl_mode_label(cfg, short=False):
    mode = cfg.get("btl_fusion_mode", "none")
    ckpt_stem = pathlib.Path(cfg["btl_ckpt"]).stem if cfg.get("btl_ckpt") else "btl"
    if mode == "boost":
        return f"BTL+FluxBoost [{ckpt_stem}]" if short else f"Beat This! Live + Flux boost ({ckpt_stem})"
    if mode == "mul":
        return f"Flux x BTL [{ckpt_stem}]" if short else f"Flux x Beat This! Live ({ckpt_stem})"
    return f"Beat This! Live [{ckpt_stem}]" if short else f"Beat This! Live ({ckpt_stem})"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--processed", type=str, default=DEFAULT_PROCESSED)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_tracks", type=int, default=None)
    p.add_argument("--recompute", action="store_true", help="ignore cached PLP tracker outputs")
    p.add_argument("--no_skip_bad_gt", action="store_true")
    p.add_argument("--no_render", action="store_true")
    p.add_argument("--lookahead", type=int, default=None, help="override PLP tracker lookahead (30FPS frames)")
    p.add_argument("--no_lock", action="store_true", help="disable lock mode (A/B comparison)")
    p.add_argument("--low_bpm", type=int, default=None, help="lower edge of the one-octave tempo range (default 90)")
    p.add_argument("--silence_db", type=float, default=None, help="silence gate threshold in dB (default -40.0)")
    p.add_argument("--lock_conf_threshold", type=float, default=None, help="peak/mean ratio required to lock (default 1.8)")
    p.add_argument("--acquire_min_active_sec", type=float, default=None, help="active warm-up seconds before lock counting (default 2.0)")
    p.add_argument("--lock_sec", type=float, default=None, help="steady tempo seconds required to lock (default 2.0)")
    p.add_argument("--lock_tol_bpm", type=float, default=None, help="tempo consistency band for locking in BPM (default 3.0)")
    p.add_argument("--phase_lead_ms", type=float, default=None,
                   help="downstream phase-lead compensation in ms (default 0.0 for BTL, 0.0 for flux)")
    p.add_argument("--pk_offset_frames", type=float, default=None)
    p.add_argument("--flag_offset_frames", type=float, default=None)
    p.add_argument("--no_oracle", action="store_true", help="skip the oracle-activation runs")
    p.add_argument("--oracle_jitter_ms", type=float, nargs="+", default=[0.0, 30.0])
    p.add_argument("--ok_f", type=float, default=0.6, help="F70 above which a window counts as 'ok'")

    # Beat This! Live frontend settings (defaults match streaming.py inference)
    p.add_argument("--btl_ckpt", type=str, default=None, help="path to Beat This! Live .ckpt")
    p.add_argument("--btl_device", type=str, default="cuda", help="device for Beat This! Live inference")
    p.add_argument("--btl_primary", action="store_true",
                   help="run Beat This! Live as the primary run (full taxonomy, onset peak stats, renders)")
    p.add_argument("--btl_streaming", action="store_true",
                   help="use strictly causal BeatThisLiveActivation (streaming) instead of BeatThisLiveOffline")
    p.add_argument("--btl_causal_stft", action="store_true",
                   help="enable causal (center=False) STFT in offline mode (already on by default)")
    p.add_argument("--btl_centered_stft", action="store_true",
                   help="disable causal STFT in offline mode (legacy centered STFT)")
    p.add_argument("--btl_use_amp", action="store_true", help="enable bf16/fp16 autocast for BTL inference")
    p.add_argument("--btl_use_cuda_graph", action="store_true", help="enable CUDA graph in --btl_streaming mode")
    p.add_argument("--btl_soxr_quality", type=str, default="LQ", choices=["QQ", "LQ", "HQ", "VHQ", "MQ"])
    p.add_argument("--btl_context_frames", type=int, default=None)
    p.add_argument("--btl_no_shift_align", action="store_true")
    p.add_argument("--btl_target_lead_ms", type=float, default=0.0)
    p.add_argument("--btl_extra_shift_ms", type=float, default=-6.0,
                   help="empirical calibration shift in ms (default -6.0 matching streaming.py)")
    p.add_argument("--btl_resample_method", type=str, default="max", choices=["linear", "max", "avg"])
    p.add_argument("--btl_downbeat_weight", type=float, default=0.0)
    p.add_argument("--btl_act_power", type=float, default=1.0)
    p.add_argument("--btl_energy_threshold", type=float, default=None)
    p.add_argument("--btl_harmonic_weight", type=float, default=None)
    p.add_argument("--btl_recompute_act", action="store_true")

    p.add_argument("--btl_fusion_mode", type=str, default="boost", choices=["none", "boost", "mul"],
                   help="causal fusion with spectral flux (default 'boost' matching streaming.py)")
    p.add_argument("--btl_flux_boost", type=float, default=0.6,
                   help="boost weight for --btl_fusion_mode boost (default 0.6 matching streaming.py)")
    p.add_argument("--btl_flux_ema_alpha", type=float, default=0.01)
    p.add_argument("--btl_flux_norm_cap", type=float, default=2.0)
    p.add_argument("--btl_fuse_flux", action="store_true",
                   help="legacy alias for --btl_fusion_mode mul")
    p.add_argument("--btl_fuse_eps", type=float, default=0.05)
    p.add_argument("--btl_fuse_past", type=int, default=2)
    p.add_argument("--btl_fuse_fut", type=int, default=0)

    p.add_argument("--min_window_active_frac", type=float, default=0.1)
    p.add_argument("--min_track_active_frac", type=float, default=0.1)
    p.add_argument("--eval_active_only", action="store_true")
    p.add_argument("--plot_exclude_silent", action="store_true")
    return p.parse_args()


def make_paths(args):
    processed = pathlib.Path(args.processed)
    out = pathlib.Path(args.out_dir)
    return dict(
        data=processed / args.split,
        latent=processed / "latents" / args.split,
        report=processed / "metadata" / f"beat_check_{args.split}.json",
        out=out,
        cache=out / "cache",
    )


def load_cfg(args):
    cfg = dict(DEFAULT_BEAT_CFG)

    cfg["hop"] = 1600
    if args.low_bpm is not None:
        cfg["low_bpm"] = int(args.low_bpm)
    want_high = 2 * int(cfg["low_bpm"]) - 1
    if int(cfg["high_bpm"]) != want_high:
        print(f"[INFO] tempo range forced to ONE octave: [{cfg['low_bpm']}, {want_high}] "
              f"(config had high_bpm={cfg['high_bpm']}).")
    cfg["high_bpm"] = want_high
    if args.no_lock:
        cfg["lock_enable"] = False
    if args.silence_db is not None:
        cfg["silence_db"] = float(args.silence_db)
    if args.lock_conf_threshold is not None:
        cfg["lock_conf_threshold"] = float(args.lock_conf_threshold)
    if args.acquire_min_active_sec is not None:
        cfg["acquire_min_active_sec"] = float(args.acquire_min_active_sec)
    if args.lock_sec is not None:
        cfg["lock_sec"] = float(args.lock_sec)
    if args.lock_tol_bpm is not None:
        cfg["lock_tol_bpm"] = float(args.lock_tol_bpm)

    cfg["lookahead"] = int(args.lookahead) if args.lookahead is not None else int(cfg["lookahead"])
    cfg["pk_offset_frames"] = float(args.pk_offset_frames) if args.pk_offset_frames is not None \
        else cfg["lookahead"] + 1.0
    cfg["flag_offset_frames"] = float(args.flag_offset_frames) if args.flag_offset_frames is not None \
        else cfg["lookahead"] + 1.0
    cfg["oracle_jitters"] = [] if args.no_oracle else [float(j) for j in args.oracle_jitter_ms]
    cfg["ok_f"] = float(args.ok_f)

    # Beat This! Live settings (matching streaming.py defaults)
    cfg["btl_ckpt"] = args.btl_ckpt
    cfg["btl_primary"] = bool(args.btl_primary and args.btl_ckpt)
    cfg["btl_streaming"] = bool(args.btl_streaming)
    cfg["btl_causal_stft"] = not bool(args.btl_centered_stft)
    cfg["btl_use_amp"] = bool(args.btl_use_amp)
    cfg["btl_use_cuda_graph"] = bool(args.btl_use_cuda_graph)
    cfg["btl_soxr_quality"] = str(args.btl_soxr_quality)
    cfg["btl_context_frames"] = int(args.btl_context_frames) if args.btl_context_frames is not None else None
    cfg["btl_align_shift"] = not bool(args.btl_no_shift_align)
    cfg["btl_target_lead_ms"] = float(args.btl_target_lead_ms)
    cfg["btl_extra_shift_ms"] = float(args.btl_extra_shift_ms)
    cfg["btl_resample_method"] = str(args.btl_resample_method)
    cfg["btl_downbeat_weight"] = float(args.btl_downbeat_weight)
    cfg["btl_act_power"] = float(args.btl_act_power)
    cfg["btl_energy_threshold"] = args.btl_energy_threshold
    cfg["btl_harmonic_weight"] = args.btl_harmonic_weight

    if args.btl_fusion_mode is not None:
        fusion_mode = str(args.btl_fusion_mode)
    elif args.btl_fuse_flux:
        fusion_mode = "mul"
    else:
        fusion_mode = "boost"
    cfg["btl_fusion_mode"] = fusion_mode
    cfg["btl_fuse_flux"] = (fusion_mode != "none")
    cfg["btl_flux_boost"] = float(args.btl_flux_boost)
    cfg["btl_flux_ema_alpha"] = float(args.btl_flux_ema_alpha)
    cfg["btl_flux_norm_cap"] = float(args.btl_flux_norm_cap)
    cfg["btl_fuse_eps"] = float(args.btl_fuse_eps)
    cfg["btl_fuse_past"] = int(args.btl_fuse_past)
    cfg["btl_fuse_fut"] = int(args.btl_fuse_fut)

    # Phase lead defaults to 0.0 ms for BTL runs (matching streaming.py), 0.0 ms for pure spectral flux
    cfg["phase_lead_ms_override"] = args.phase_lead_ms
    cfg["phase_lead_ms"] = float(args.phase_lead_ms) if args.phase_lead_ms is not None else 0.0

    cfg["min_window_active_frac"] = float(args.min_window_active_frac)
    cfg["min_track_active_frac"] = float(args.min_track_active_frac)
    cfg["eval_active_only"] = bool(args.eval_active_only)
    cfg["plot_exclude_silent"] = bool(args.plot_exclude_silent)

    return cfg


def btl_tracker_cfg(cfg):
    """Return tracker config used for BTL runs (matching streaming.py inference defaults)."""
    c = dict(cfg)
    prob_scale = cfg.get("btl_ckpt") and cfg.get("btl_fusion_mode", "boost") in ("none", "boost")
    if cfg.get("btl_energy_threshold") is not None:
        c["energy_threshold"] = float(cfg["btl_energy_threshold"])
    elif prob_scale:
        c["energy_threshold"] = min(float(c["energy_threshold"]), 0.015)

    if cfg.get("btl_harmonic_weight") is not None:
        c["harmonic_weight"] = float(cfg["btl_harmonic_weight"])
    elif prob_scale:
        c["harmonic_weight"] = 0.0

    if cfg.get("phase_lead_ms_override") is not None:
        c["phase_lead_ms"] = float(cfg["phase_lead_ms_override"])
    elif cfg.get("btl_ckpt"):
        c["phase_lead_ms"] = 0.0
    return c


def btl_act_sig(cfg):
    ckpt_name = pathlib.Path(cfg["btl_ckpt"]).stem
    key = json.dumps({
        "ckpt": ckpt_name,
        "hop": cfg["hop"],
        "streaming": cfg["btl_streaming"],
        "comp_resamp": True if cfg["btl_streaming"] else False,
        "causal_stft": cfg["btl_causal_stft"] if not cfg["btl_streaming"] else True,
        "amp": cfg["btl_use_amp"],
        "soxr": cfg["btl_soxr_quality"],
        "ctx": cfg["btl_context_frames"],
        "align": cfg["btl_align_shift"],
        "lead_ms": cfg["btl_target_lead_ms"],
        "extra_ms": cfg["btl_extra_shift_ms"],
        "method": cfg["btl_resample_method"],
        "dbw": cfg["btl_downbeat_weight"],
        "pow": cfg["btl_act_power"],
    }, sort_keys=True)
    return f"{ckpt_name}_{hashlib.md5(key.encode()).hexdigest()[:8]}"


def btl_run_sig(cfg):
    base = btl_act_sig(cfg)
    mode = cfg.get("btl_fusion_mode", "boost")
    if mode == "none":
        return f"btl_{base}"
    if mode == "boost":
        fkey_str = json.dumps({
            "mode": "boost",
            "boost": cfg["btl_flux_boost"],
            "alpha": cfg["btl_flux_ema_alpha"],
            "cap": cfg["btl_flux_norm_cap"],
        }, sort_keys=True)
        return f"btlboost_{base}_{hashlib.md5(fkey_str.encode()).hexdigest()[:6]}"
    fkey_str = json.dumps({
        "mode": "mul",
        "eps": cfg["btl_fuse_eps"],
        "past": cfg["btl_fuse_past"],
        "fut": cfg["btl_fuse_fut"],
    }, sort_keys=True)
    return f"btlfuse_{base}_{hashlib.md5(fkey_str.encode()).hexdigest()[:6]}"


def btl_act_cache_path(paths, track, stem, cfg):
    return paths["cache"] / f"{track}_{stem}_btlact_{btl_act_sig(cfg)}.npz"


def precompute_btl_activations(items, cfg, paths, device, recompute=False):
    if not cfg.get("btl_ckpt"):
        return
    todo = [(t, s) for (t, s, *_rest) in items
            if recompute or not btl_act_cache_path(paths, t, s, cfg).exists()]
    if not todo:
        print(f"[INFO] Using cached Beat This! Live activations ({btl_act_sig(cfg)}).")
        return

    from btl_frontend import BeatThisLiveOffline, BeatThisLiveActivation, SOXR_LATENCY_MS_48K_TO_22K

    hop = int(cfg["hop"])
    if cfg["btl_streaming"]:
        fe = BeatThisLiveActivation(
            ckpt_path=cfg["btl_ckpt"],
            samplerate=48000,
            H=hop,
            device=device,
            use_amp=cfg["btl_use_amp"],
            use_cuda_graph=cfg["btl_use_cuda_graph"],
            context_frames=cfg["btl_context_frames"],
            align_label_shift=cfg["btl_align_shift"],
            target_lead_ms=cfg["btl_target_lead_ms"],
            extra_shift_ms=cfg["btl_extra_shift_ms"],
            compensate_resampler_latency=True,
            resample_method=cfg["btl_resample_method"],
            downbeat_weight=cfg["btl_downbeat_weight"],
            act_power=cfg["btl_act_power"],
            soxr_quality=cfg["btl_soxr_quality"],
        )
        spec = fe.spec
        resamp_ms = fe.resampler_latency_ms
    else:
        fe = BeatThisLiveOffline(
            ckpt_path=cfg["btl_ckpt"],
            device=device,
            align_label_shift=cfg["btl_align_shift"],
            causal_stft=cfg["btl_causal_stft"],
            target_lead_ms=cfg["btl_target_lead_ms"],
            extra_shift_ms=cfg["btl_extra_shift_ms"],
            resample_method=cfg["btl_resample_method"],
            downbeat_weight=cfg["btl_downbeat_weight"],
            act_power=cfg["btl_act_power"],
            soxr_quality=cfg["btl_soxr_quality"],
            use_amp=cfg["btl_use_amp"],
        )
        spec = fe.spec
        resamp_ms = float(SOXR_LATENCY_MS_48K_TO_22K.get(cfg["btl_soxr_quality"].upper(), 6.8))

    margin_ms = spec.causal_margin_ms_at_30fps(
        target_lead_ms=cfg["btl_target_lead_ms"],
        resampler_latency_ms=resamp_ms,
        method=cfg["btl_resample_method"],
    ) + min(cfg["btl_extra_shift_ms"], 0.0)
    mode_str = "streaming" if cfg["btl_streaming"] else ("offline-causal-stft" if cfg["btl_causal_stft"] else "offline")
    print(f"[INFO] Beat This! Live ({mode_str}, soxr={cfg['btl_soxr_quality']}, amp={cfg['btl_use_amp']}) spec: "
          f"label_shift={spec.label_shift} (raw pred lead={spec.lead_frames * 20:+.0f} ms, SCLM bypassed), "
          f"sw_window={spec.sw_window}, extra_shift={cfg['btl_extra_shift_ms']:+.1f} ms, "
          f"30FPS eff_causal_margin={margin_ms:+.2f} ms")
    if margin_ms < 0:
        print(f"[WARNING] 30FPS eff_causal_margin is NEGATIVE ({margin_ms:+.2f} ms); streaming will clamp!")

    total_clamped = 0
    for track, stem in tqdm(todo, ncols=140, desc=f"BTL ({mode_str})"):
        y, sr = sf.read(str(paths["data"] / track / f"{stem}.wav"), dtype="float32", always_2d=True)
        if sr != 48000:
            raise ValueError(f"Expected 48000 Hz audio, got {sr} for {track}/{stem}")
        mono = y.mean(axis=1)

        if cfg["btl_streaming"]:
            fe.reset()
            n_frames = int(math.ceil(len(mono) / hop))
            x = np.pad(mono, (0, max(0, n_frames * hop - len(mono))))
            act = np.zeros(n_frames, dtype=np.float32)
            db = np.zeros(n_frames, dtype=np.float32)
            for n in range(n_frames):
                a = fe.process(x[n * hop:(n + 1) * hop])
                act[n] = float(a[0]) if a.size else 0.0
                db[n] = float(fe.downbeat_frame)
            total_clamped += int(fe.n_clamped_steps)
            res = dict(
                act=act,
                downbeat=db,
                n_clamped_steps=np.int32(fe.n_clamped_steps),
                label_shift=np.int32(spec.label_shift),
                sw_window=np.int32(spec.sw_window),
                fps_out=np.float32(sr / hop),
            )
        else:
            res = fe.extract(mono, sr=sr, hop_out=hop)

        out_p = btl_act_cache_path(paths, track, stem, cfg)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_p, **res)

    if cfg["btl_streaming"]:
        print(f"[INFO] Beat This! Live streaming finished: total_clamped_steps={total_clamped} across {len(todo)} stems.")


# ----------------------------------------------------------------------------------------------------------
# Causal Flux + Beat This! Fusion (strictly causal, bit-for-bit parity with CausalFluxBTLActivation)
# ----------------------------------------------------------------------------------------------------------
def compute_causal_flux(mono, sr, hop):
    fps = float(sr) / float(hop)
    n_frames = int(math.ceil(len(mono) / hop))
    x = np.pad(mono, (0, max(0, n_frames * hop - len(mono))))
    fa = BeatActivation(N=2 * hop, H=hop, samplerate=sr, M=max(1, round(fps / 3.0)))
    flux = np.zeros(n_frames, dtype=np.float64)
    for n in range(n_frames):
        out = fa.process(x[n * hop:(n + 1) * hop])
        flux[n] = float(out[0]) if np.asarray(out).size else 0.0
    return flux


def fuse_flux_btl_causal(
    flux,
    btl_act,
    mode="boost",
    flux_boost=0.6,
    flux_ema_alpha=0.01,
    flux_norm_cap=2.0,
    eps=0.05,
    past_frames=2,
    fut_frames=0,
):
    flux = np.maximum(np.asarray(flux, dtype=np.float64), 0.0)
    btl = np.maximum(np.asarray(btl_act, dtype=np.float64), 0.0)
    n_len = min(len(flux), len(btl))

    if mode == "none":
        return btl[:n_len].astype(np.float32)

    fused = np.zeros(n_len, dtype=np.float64)
    if mode == "boost":
        ema = 0.0
        for n in range(n_len):
            f_val = float(flux[n])
            if f_val > 1e-6:
                ema = f_val if ema <= 0.0 else (1.0 - flux_ema_alpha) * ema + flux_ema_alpha * f_val
                f_norm = min(f_val / max(ema, 1e-6), flux_norm_cap)
            else:
                f_norm = 0.0
            fused[n] = btl[n] * (1.0 + flux_boost * f_norm)
        return fused.astype(np.float32)

    if fut_frames != 0:
        raise ValueError("btl_fuse_fut must be 0 for strict causal parity with streaming.")
    for n in range(n_len):
        lo = max(0, n - past_frames)
        hi = min(n_len, n + 1)
        gate = float(np.max(btl[lo:hi])) if hi > lo else 0.0
        fused[n] = flux[n] * (eps + gate)
    return fused.astype(np.float32)


# ----------------------------------------------------------------------------------------------------------
# Ground truth (prefers beats.tsv matching pre_encode.py, falls back to all_src.mid)
# ----------------------------------------------------------------------------------------------------------
def _beats_to_pos_bpm(beats, beat_positions, times):
    beats = np.asarray(beats, dtype=np.float64)
    pos_arr = np.asarray(beat_positions, dtype=np.int32)
    if beats.size < 2:
        raise ValueError("fewer than 2 beats in ground truth")
    order = np.argsort(beats)
    beats, pos_arr = beats[order], pos_arr[order]
    keep = np.concatenate([[True], np.diff(beats) > 1e-4])
    beats, pos_arr = beats[keep], pos_arr[keep]

    idx = np.arange(beats.size, dtype=np.float64)
    pos = np.interp(times, beats, idx)

    intervals = np.maximum(np.diff(beats), 1e-6)
    first_iv = max(float(np.median(intervals[:4])), 1e-6)
    last_iv = max(float(np.median(intervals[-4:])), 1e-6)
    pos = np.where(times < beats[0], (times - beats[0]) / first_iv, pos)
    pos = np.where(times > beats[-1], (beats.size - 1) + (times - beats[-1]) / last_iv, pos)

    smooth_iv = median_filter(intervals, size=5, mode="mirror") if intervals.size >= 5 else intervals
    seg = np.clip(np.searchsorted(beats, times, side="right") - 1, 0, smooth_iv.size - 1)
    bpm = 60.0 / smooth_iv[seg]
    return beats, pos_arr, pos, bpm


def load_gt(track_dir, times):
    tsv_path = track_dir / "beats.tsv"
    midi_path = track_dir / "all_src.mid"
    if tsv_path.exists():
        b_times, b_pos = [], []
        for ln in tsv_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            b_times.append(float(parts[0]))
            b_pos.append(int(float(parts[1])) if len(parts) >= 2 else 0)
        return _beats_to_pos_bpm(b_times, b_pos, times)
    if midi_path.exists():
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        beats = np.sort(np.asarray(pm.get_beats(), dtype=np.float64))
        downbeats = np.sort(np.asarray(pm.get_downbeats(), dtype=np.float64))
        b_pos = np.zeros(beats.size, dtype=np.int32)
        if downbeats.size > 0 and beats.size > 0:
            j = np.searchsorted(downbeats, beats)
            lo = downbeats[np.clip(j - 1, 0, downbeats.size - 1)]
            hi = downbeats[np.clip(j, 0, downbeats.size - 1)]
            b_pos[np.minimum(np.abs(beats - lo), np.abs(beats - hi)) <= 0.03] = 1
        return _beats_to_pos_bpm(beats, b_pos, times)
    raise FileNotFoundError(f"Neither beats.tsv nor all_src.mid found in {track_dir}")


def gt_fold_factor(beats, low_bpm):
    if beats.size < 2:
        return 1.0
    mb = 60.0 / max(float(np.median(np.diff(beats))), 1e-6)
    f = 1.0
    while mb * f < low_bpm:
        f *= 2.0
    while mb * f >= 2.0 * low_bpm:
        f *= 0.5
    return f


def folded_beat_times(beats, f, t_max, anchor_idx=0.0):
    idx = np.arange(beats.size, dtype=np.float64)
    step = 1.0 / f
    # Anchor halving (f < 1) on the first downbeat so beat 1 stays aligned with phase 0
    start = anchor_idx % step if f < 1.0 else 0.0
    x = np.arange(start, beats.size - 1 + 1e-9, step)
    t = np.interp(x, idx, beats)
    return t[t <= t_max]


def clean_beats(beats, duration, min_gap=GT_MIN_BEAT_GAP_S):
    b = beats[beats <= duration]
    if b.size == 0:
        return b
    keep = [b[0]]
    for x in b[1:]:
        if x - keep[-1] >= min_gap:
            keep.append(x)
    return np.asarray(keep)


# ----------------------------------------------------------------------------------------------------------
# Tracker runs (real audio, Beat This! Live / fused activation, or oracle activation)
# ----------------------------------------------------------------------------------------------------------
def build_tracker(cfg, sr):
    hop = int(cfg["hop"])
    return RealTimeBeatTracker.from_args(
        N=2 * hop,
        H=hop,
        samplerate=sr,
        N_time=cfg["n_time"],
        Theta=np.arange(cfg["low_bpm"], cfg["high_bpm"] + 1, 1),
        lookahead=cfg["lookahead"],
        silence_db=cfg["silence_db"],
        hold_frames=cfg["hold_frames"],
        conf_threshold=cfg["conf_threshold"],
        lock_conf_threshold=cfg["lock_conf_threshold"],
        switch_ratio=cfg["switch_ratio"],
        energy_threshold=cfg["energy_threshold"],
        **{k: cfg[k] for k in LOCK_KEYS},
    )


def process_with_activation(tracker, frame, a):
    rms = np.sqrt(np.mean(np.asarray(frame, dtype=np.float64) ** 2))
    tracker.last_rms_db = float(20 * np.log10(rms + 1e-12))
    if tracker.last_rms_db > tracker.silence_db:
        tracker._hold = tracker.hold_frames
    else:
        tracker._hold = max(tracker._hold - 1, 0)
    tracker.active = tracker._hold > 0
    a_eff = float(a) if tracker.active else 0.0
    act = np.array([a_eff])
    tracker.activation.activation_frame = act
    tempogram_frame = tracker.tempogram.process(act)
    pulse = tracker.plp.process(tempogram_frame, tracker.active, tracker.tempogram._tempo_buffer,
                                tracker.tempogram.win, tracker.tempogram.harm_frames)
    return tracker.plp.detect_beat(pulse)


def run_tracker(mono, sr, cfg, act_override=None, phase_lead_ms=0.0):
    hop = int(cfg["hop"])
    if sr != 48000:
        raise ValueError(f"tracker expects sr=48000, got {sr}")

    n_model = int(math.ceil(len(mono) / hop))
    x = mono
    if x.shape[0] < n_model * hop:
        x = np.pad(x, (0, n_model * hop - x.shape[0]))

    tracker = build_tracker(cfg, sr)
    cs = tracker.cs
    phase_lead_sec = float(phase_lead_ms) / 1000.0

    sin_ = np.zeros(n_model, dtype=np.float32)
    cos_ = np.zeros(n_model, dtype=np.float32)
    tempo_ = np.zeros(n_model, dtype=np.float32)
    lfo_ = np.zeros(n_model, dtype=np.float32)
    conf_ = np.zeros(n_model, dtype=np.float32)
    rms_ = np.zeros(n_model, dtype=np.float32)
    flag_ = np.zeros(n_model, dtype=bool)
    trusted_ = np.zeros(n_model, dtype=bool)
    active_ = np.zeros(n_model, dtype=bool)
    lock_ = np.zeros(n_model, dtype=bool)
    lscore_ = np.zeros(n_model, dtype=np.float32)
    act_ = np.zeros(n_model, dtype=np.float32)

    for n in range(n_model):
        frame = x[n * hop:(n + 1) * hop]
        if act_override is None:
            flag_[n] = tracker.process(frame)
            lfo_[n] = cs.gamma_lfo
        else:
            flag_[n] = process_with_activation(tracker, frame, act_override[n])

        ph = cs.beat_phase
        if ph is not None and phase_lead_sec != 0.0 and cs.tempo_bpm > 0.0:
            ph = (ph + phase_lead_sec * cs.tempo_bpm / 60.0) % 1.0
        sin_[n] = 0.0 if ph is None else float(np.sin(2.0 * np.pi * ph))
        cos_[n] = 0.0 if ph is None else float(np.cos(2.0 * np.pi * ph))
        tempo_[n] = cs.tempo_bpm
        conf_[n] = tracker.plp.last_confidence
        trusted_[n] = tracker.plp.last_trusted
        active_[n] = tracker.active
        rms_[n] = tracker.last_rms_db
        lock_[n] = cs.lock_state
        lscore_[n] = cs.lock_score
        af = tracker.activation.activation_frame
        act_[n] = af[0] if af.size else 0.0

    sin_ = np.clip(np.nan_to_num(sin_, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    cos_ = np.clip(np.nan_to_num(cos_, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    lfo_ = np.clip(np.nan_to_num(lfo_, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    return dict(sin=sin_, cos=cos_, tempo=tempo_, lfo=lfo_, conf=conf_,
                rms_db=rms_, flag=flag_, trusted=trusted_, active=active_,
                lock=lock_, lock_score=lscore_, act=act_,
                num_frames=n_model, fps=sr / hop)


def make_oracle_activation(gt_beats, n_frames, fps, amp, jitter_ms, seed):
    rng = np.random.default_rng(seed)
    tb = np.asarray(gt_beats, dtype=np.float64)
    if jitter_ms > 0:
        tb = tb + rng.normal(0.0, jitter_ms / 1000.0, tb.size)
    act = np.zeros(n_frames, dtype=np.float64)
    for b in tb:
        c = b * fps
        lo, hi = max(int(np.floor(c - 4)), 0), min(int(np.ceil(c + 4)) + 1, n_frames)
        if hi <= lo:
            continue
        k = np.arange(lo, hi)
        act[lo:hi] += np.exp(-0.5 * (((k - c) / fps) / ORACLE_SIGMA_S) ** 2)
    return amp * act


def cache_path(paths, track, stem, cfg, tag="real"):
    key = json.dumps({**{k: cfg[k] for k in TRACKER_KEYS}, "v": CACHE_VERSION, "tag": tag,
                      "sigma": ORACLE_SIGMA_S}, sort_keys=True, default=str)
    return paths["cache"] / f"{track}_{stem}_{tag}_{hashlib.md5(key.encode()).hexdigest()[:8]}.npz"


def load_or_compute(cache, compute, required, recompute):
    if cache.exists() and not recompute:
        try:
            d = dict(np.load(cache))
            if all(k in d for k in required):
                return d
        except Exception:
            pass
    d = compute()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **d)
    return d


def phase_events(sin, cos, offset_sec, fps, min_gap_frames=MIN_EVENT_GAP_FRAMES):
    s, c = sin.astype(np.float64), cos.astype(np.float64)
    ok = np.hypot(s, c) > 1e-6
    ph = np.arctan2(s, c)
    n = np.where(ok[:-1] & ok[1:] & (ph[:-1] < 0) & (ph[1:] >= 0) & (c[:-1] > 0) & (c[1:] > 0))[0]
    if n.size == 0:
        return np.array([])
    frac = n + (-ph[n]) / np.maximum(ph[n + 1] - ph[n], 1e-9)
    keep = [0]
    for i in range(1, len(frac)):
        if frac[i] - frac[keep[-1]] >= min_gap_frames:
            keep.append(i)
    return (frac[keep] / fps) + offset_sec


def build_run(tr, cfg, fps, lead_sec=0.0):
    R = dict(tr)
    R["pk_times"] = phase_events(tr["sin"], tr["cos"], cfg["pk_offset_frames"] / fps + lead_sec, fps)
    R["flag_times"] = np.nonzero(tr["flag"])[0] / fps + cfg["flag_offset_frames"] / fps + lead_sec
    return R


def onset_peaks(act, fps, lead_sec=0.0):
    a = np.asarray(act, dtype=np.float64)
    if a.size < 3 or a.max() <= 0:
        return np.array([]), np.array([])
    ref = max(float(np.percentile(a, 99)), 1e-9)
    p, _ = signal.find_peaks(a, height=ONSET_REL_THR * ref)
    if p.size == 0:
        return np.array([]), np.array([])
    delta = np.zeros(p.size)
    ok = (p > 0) & (p < a.size - 1)
    y0, y1, y2 = a[p[ok] - 1], a[p[ok]], a[p[ok] + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
    delta[ok] = np.clip(d, -0.5, 0.5)
    return (p + delta) / fps + lead_sec, a[p] / ref


def prepare_track(track, stem, cfg, paths, recompute=False):
    track_dir = paths["data"] / track
    y, sr = sf.read(str(track_dir / f"{stem}.wav"), dtype="float32", always_2d=True)
    mono = y.mean(axis=1).astype(np.float64)

    btl_act = None
    btl_cfg = btl_tracker_cfg(cfg)
    btl_lead_sec = float(cfg.get("btl_target_lead_ms", 0.0)) / 1000.0
    if cfg.get("btl_ckpt"):
        btl_p = btl_act_cache_path(paths, track, stem, cfg)
        if not btl_p.exists():
            raise FileNotFoundError(f"BTL activation cache missing: {btl_p}")
        btl_act = np.load(btl_p)["act"]
        mode = cfg.get("btl_fusion_mode", "boost")
        if mode != "none":
            flux = compute_causal_flux(mono, sr, int(cfg["hop"]))
            btl_act = fuse_flux_btl_causal(
                flux,
                btl_act,
                mode=mode,
                flux_boost=cfg["btl_flux_boost"],
                flux_ema_alpha=cfg["btl_flux_ema_alpha"],
                flux_norm_cap=cfg["btl_flux_norm_cap"],
                eps=cfg["btl_fuse_eps"],
                past_frames=cfg["btl_fuse_past"],
                fut_frames=cfg["btl_fuse_fut"],
            )

    if cfg.get("btl_primary"):
        btl_tag = btl_run_sig(cfg)
        tr = load_or_compute(
            cache_path(paths, track, stem, btl_cfg, btl_tag),
            lambda: run_tracker(mono, sr, btl_cfg, act_override=btl_act,
                                phase_lead_ms=btl_cfg["phase_lead_ms"]),
            REQ_KEYS, recompute,
        )
        primary_lead_sec = btl_lead_sec
    else:
        tr = load_or_compute(
            cache_path(paths, track, stem, cfg, "real"),
            lambda: run_tracker(mono, sr, cfg, phase_lead_ms=cfg["phase_lead_ms"]),
            REQ_KEYS, recompute,
        )
        primary_lead_sec = 0.0

    fps = float(tr["fps"])
    num_frames = int(tr["num_frames"])
    frame_times = np.arange(num_frames) / fps + cfg["pk_offset_frames"] / fps + primary_lead_sec
    duration = len(mono) / sr

    beats, beat_positions, pos, bpm = load_gt(track_dir, frame_times)
    fold = gt_fold_factor(beats, cfg["low_bpm"])
    db_indices = np.flatnonzero(beat_positions == 1)
    db0 = float(db_indices[0]) if db_indices.size > 0 else 0.0
    pos = (pos - db0) * fold if fold < 1.0 else pos * fold
    bpm = bpm * fold
    gt_beats = clean_beats(folded_beat_times(beats, fold, duration, anchor_idx=db0), duration)
    ang = 2.0 * np.pi * pos

    runs = {"real": build_run(tr, cfg, fps, lead_sec=primary_lead_sec)}

    if btl_act is not None and not cfg.get("btl_primary"):
        btl_tag = btl_run_sig(cfg)
        tr_btl = load_or_compute(
            cache_path(paths, track, stem, btl_cfg, btl_tag),
            lambda: run_tracker(mono, sr, btl_cfg, act_override=btl_act,
                                phase_lead_ms=btl_cfg["phase_lead_ms"]),
            REQ_KEYS, recompute,
        )
        runs["btl"] = build_run(tr_btl, cfg, fps, lead_sec=btl_lead_sec)

    # Oracle runs stay strictly unshifted (phase_lead_ms=0.0) so oracle measures pure back-end bias
    orc_cfg = dict(cfg)
    orc_cfg["phase_lead_ms"] = 0.0
    if cfg.get("btl_primary") and cfg.get("btl_fusion_mode", "boost") in ("none", "boost"):
        amp = 1.0
    else:
        amp = float(np.percentile(tr["act"], 95))
        amp = amp if amp > 0 else 1.0
    seed0 = int(hashlib.md5(f"{track}/{stem}".encode()).hexdigest()[:8], 16)
    for j in cfg["oracle_jitters"]:
        name = f"orc{j:g}"

        def compute(j=j):
            act = make_oracle_activation(gt_beats, num_frames, fps, amp, j, seed0)
            return run_tracker(mono, sr, orc_cfg, act_override=act, phase_lead_ms=0.0)

        runs[name] = build_run(
            load_or_compute(cache_path(paths, track, stem, orc_cfg, name), compute, REQ_KEYS, recompute),
            cfg, fps, lead_sec=0.0,
        )

    onsets, _ = onset_peaks(tr["act"], fps, lead_sec=primary_lead_sec)
    return dict(
        mono=mono, sr=sr, duration=duration, gt_fold=fold, fps=fps, num_frames=num_frames,
        frame_times=frame_times, primary_lead_sec=primary_lead_sec,
        beats_raw=beats, gt_beats=gt_beats,
        gt_pos=pos, gt_bpm=bpm, gt_sin=np.sin(ang).astype(np.float32), gt_cos=np.cos(ang).astype(np.float32),
        runs=runs, act=tr["act"], onsets=onsets,
    )


# ----------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------
def fmeas(g, e, tol):
    try:
        return float(mir_eval.beat.f_measure(g, e, f_measure_threshold=tol))
    except Exception:
        return 0.0


def nearest_offsets(ref, x):
    j = np.searchsorted(ref, x)
    lo = ref[np.clip(j - 1, 0, ref.size - 1)]
    hi = ref[np.clip(j, 0, ref.size - 1)]
    near = np.where(np.abs(x - lo) <= np.abs(x - hi), lo, hi)
    return x - near


def nearest_dist(ref, x):
    if ref.size == 0:
        return np.full(np.shape(x), np.inf)
    return np.abs(nearest_offsets(ref, x))


def level_variants(g):
    mid = 0.5 * (g[:-1] + g[1:])
    return {"beat": g, "offbeat": mid, "double": np.sort(np.concatenate([g, mid])),
            "half_even": g[0::2], "half_odd": g[1::2]}


def event_metrics(gt, est, t0, t1, prefix, sweep=False, light=False):
    g = gt[(gt >= t0) & (gt < t1)]
    e = est[(est >= t0) & (est < t1)]
    fk = [fkey(t) for t in TOLS]
    zero_keys = fk + ["Cemgil", "CMLc", "CMLt", "AMLc", "AMLt", "F70_debias", "F70_lvlfree",
                      "Fv_offbeat", "Fv_double", "Fv_half"]
    out = {prefix + k: np.nan for k in zero_keys + ["off_med_ms", "off_mad_ms", "off_p90_ms", "ibi_ratio"]}
    out[prefix + "lvl_best"] = "beat"
    out[prefix + "n_gt"] = int(len(g))
    out[prefix + "n_est"] = int(len(e))
    offs, curve = np.array([]), None
    if len(g) < 4:
        return out, offs, curve
    if len(e) < 2:
        for k in zero_keys:
            out[prefix + k] = 0.0
        out[prefix + "lvl_best"] = "none"
        if sweep:
            curve = np.zeros(len(SHIFTS_MS))
        return out, offs, curve

    for t in TOLS:
        out[prefix + fkey(t)] = fmeas(g, e, t)

    d = nearest_offsets(g, e)
    d = d[np.abs(d) < OFFSET_CAP_S]
    med = 0.0
    if d.size:
        med = float(np.median(d))
        out[prefix + "off_med_ms"] = med * 1000.0
        out[prefix + "off_mad_ms"] = float(np.median(np.abs(d - med))) * 1000.0
        out[prefix + "off_p90_ms"] = float(np.percentile(np.abs(d - med), 90)) * 1000.0
        offs = d * 1000.0
    out[prefix + "F70_debias"] = fmeas(g, e - med, 0.07)
    if sweep:
        curve = np.array([fmeas(g, e + s / 1000.0, 0.07) for s in SHIFTS_MS])

    if not light:
        try:
            out[prefix + "Cemgil"] = float(mir_eval.beat.cemgil(g, e)[0])
            cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(g, e)
            out[prefix + "CMLc"], out[prefix + "CMLt"] = float(cmlc), float(cmlt)
            out[prefix + "AMLc"], out[prefix + "AMLt"] = float(amlc), float(amlt)
        except Exception:
            pass
        v = level_variants(g)
        fv = {k: fmeas(v[k], e, 0.07) for k in v}
        fv["half"] = max(fv["half_even"], fv["half_odd"])
        f_beat = out[prefix + "F70"]
        out[prefix + "Fv_offbeat"], out[prefix + "Fv_double"], out[prefix + "Fv_half"] = \
            fv["offbeat"], fv["double"], fv["half"]
        best, best_f = "beat", f_beat
        for k in ("offbeat", "double", "half"):
            if fv[k] > best_f + 0.1:
                best, best_f = k, fv[k]
        out[prefix + "lvl_best"] = best
        out[prefix + "F70_lvlfree"] = max(f_beat, fv["offbeat"], fv["double"], fv["half"])
        out[prefix + "ibi_ratio"] = float(np.median(np.diff(e)) / max(np.median(np.diff(g)), 1e-6))
    return out, offs, curve


def region_hits(gt, est, active, fps, t0, t1, tol=0.07):
    out = {}
    for r in REGIONS:
        out[f"hit70_{r}"] = np.nan
        out[f"n_{r}"] = 0
    g = gt[(gt >= t0) & (gt < t1)]
    if g.size == 0 or active.size == 0:
        return out
    idx = np.clip(np.rint(g * fps).astype(int), 0, active.size - 1)
    edges = np.flatnonzero(~active[:-1] & active[1:]) + 1
    since = np.full(g.size, np.inf)
    if edges.size:
        k = np.searchsorted(edges, idx, side="right") - 1
        since = np.where(k >= 0, (idx - edges[np.maximum(k, 0)]) / fps, np.inf)
    reg = np.where(~active[idx], "silent", np.where(since < RESUME_SEC, "resume", "steady"))
    hit = nearest_dist(est, g) <= tol
    for r in REGIONS:
        m = reg == r
        out[f"n_{r}"] = int(m.sum())
        if m.any():
            out[f"hit70_{r}"] = float(hit[m].mean())
    return out


def acquisition_metrics(gt, est, tol=0.07, run=4):
    out = {"acq_sec": np.nan, "hit70_post_acq": np.nan}
    if gt.size < run or est.size < 2:
        return out
    hit = nearest_dist(est, gt) <= tol
    c = np.convolve(hit.astype(int), np.ones(run, dtype=int), mode="valid")
    idx = np.flatnonzero(c == run)
    if idx.size:
        out["acq_sec"] = float(gt[idx[0]])
        out["hit70_post_acq"] = float(hit[idx[0]:].mean())
    return out


def frontend_metrics(onsets, act, fps, gt, t0, t1, lead_sec=0.0):
    keys = ["onsets_per_beat", "onset_rec50", "onset_rec100", "onset_prec100",
            "act_off_med_ms", "act_off_mad_ms", "beat_act_ratio"]
    out = {k: np.nan for k in keys}
    g = gt[(gt >= t0) & (gt < t1)]
    o = onsets[(onsets >= t0) & (onsets < t1)]
    offs = np.array([])
    if g.size < 4:
        return out, offs
    out["onsets_per_beat"] = float(o.size / g.size)
    n0 = max(int(np.ceil((t0 - lead_sec) * fps)), 0)
    n1 = min(int(np.floor((t1 - lead_sec) * fps)), len(act))
    if n1 - n0 >= 10:
        idx = np.clip(np.rint((g - lead_sec) * fps).astype(int), 0, len(act) - 1)
        sal = maximum_filter1d(np.asarray(act, dtype=np.float64), 3)[idx]
        out["beat_act_ratio"] = float(sal.mean() / max(float(np.mean(act[n0:n1])), 1e-9))
    if o.size == 0:
        out["onset_rec50"] = out["onset_rec100"] = 0.0
        return out, offs
    dr = nearest_dist(o, g)
    out["onset_rec50"], out["onset_rec100"] = float((dr <= 0.05).mean()), float((dr <= 0.10).mean())
    ds = nearest_offsets(g, o)
    out["onset_prec100"] = float((np.abs(ds) <= 0.10).mean())
    dm = ds[np.abs(ds) < OFFSET_CAP_S]
    if dm.size:
        med = float(np.median(dm))
        out["act_off_med_ms"] = med * 1000.0
        out["act_off_mad_ms"] = float(np.median(np.abs(dm - med))) * 1000.0
        offs = dm * 1000.0
    return out, offs


def _pearson(a, b):
    if a.size < 10 or a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def phase_metrics(sin, cos, pos, gt_bpm):
    out = {"plv": np.nan, "plv_k1": np.nan, "level_k": np.nan, "phase_std_deg": np.nan, "phase_off_ms": np.nan,
           "locked_frac": float((np.hypot(sin, cos) > 1e-6).mean()) if len(sin) else np.nan}
    amp = np.hypot(sin.astype(np.float64), cos.astype(np.float64))
    ok = amp > 1e-6
    if ok.sum() < 30:
        return out
    z = (cos[ok].astype(np.float64) + 1j * sin[ok].astype(np.float64)) / amp[ok]
    p = pos[ok]
    best = None
    for k in LEVELS:
        m = (z * np.exp(-2j * np.pi * k * p)).mean()
        r = float(abs(m))
        if k == 1.0:
            out["plv_k1"] = r
        if best is None or r > best[0]:
            best = (r, k, m)
    r, k, m = best
    out["plv"], out["level_k"] = float(r), float(k)
    out["phase_std_deg"] = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(r, 1e-9)), 0.0))))
    bpm = gt_bpm[ok]
    bpm_med = float(np.median(bpm[bpm > 0])) if (bpm > 0).any() else np.nan
    if np.isfinite(bpm_med):
        out["phase_off_ms"] = float(-np.angle(m) / (2.0 * np.pi) * 60000.0 / (bpm_med * k))
    return out


def _lagged(a, b, d):
    if d >= 0:
        return a[d:], b[:len(b) - d]
    return a[:d], b[-d:]


def feature_corr(sin, cos, pos, k, frame_ms):
    ang = 2.0 * np.pi * k * pos
    gs, gc = np.sin(ang), np.cos(ang)

    def corr(d):
        a1, b1 = _lagged(cos, gc, d)
        a2, b2 = _lagged(sin, gs, d)
        return 0.5 * (_pearson(a1, b1) + _pearson(a2, b2))

    best_c, best_d = -2.0, 0
    for d in range(-LAG_MAX_FRAMES, LAG_MAX_FRAMES + 1):
        c = corr(d)
        if c > best_c:
            best_c, best_d = c, d
    return {"feat_corr0": corr(0), "feat_corr_best": best_c, "feat_lag_ms": best_d * frame_ms}


def tempo_metrics(pred, gt, fps):
    v = (pred > 0) & (gt > 0)
    out = {"tempo_valid_frac": float(v.mean()), "tempo_acc1": np.nan, "tempo_acc2": np.nan,
           "tempo_ratio_med": np.nan, "tempo_pred_med": np.nan,
           "tempo_med_abs_err_pct": np.nan, "tempo_jumps_per_min": np.nan}
    both = (pred[:-1] > 0) & (pred[1:] > 0)
    if both.any() and len(pred) > 1:
        jumps = np.abs(np.diff(pred.astype(np.float64)))[both] > JUMP_BPM
        out["tempo_jumps_per_min"] = float(jumps.sum() / max(len(pred) / fps / 60.0, 1e-6))
    if v.sum() == 0:
        return out
    r = pred[v] / gt[v]
    out["tempo_acc1"] = float((np.abs(r - 1) < 0.04).mean())
    ok2 = np.zeros(r.shape, dtype=bool)
    for k in (1, 2, 0.5, 3, 1 / 3):
        ok2 |= np.abs(r / k - 1) < 0.04
    out["tempo_acc2"] = float(ok2.mean())
    out["tempo_ratio_med"] = float(np.median(r))
    out["tempo_pred_med"] = float(np.median(pred[v]))
    out["tempo_med_abs_err_pct"] = float(np.median(np.abs(r - 1)) * 100.0)
    return out


def classify(row, ok_f, eval_active_only=False):
    f70 = row.get("pk_F70", np.nan)
    if not np.isfinite(f70):
        return "n/a"
    if f70 >= ok_f:
        return "ok"
    if (not eval_active_only) and row["active_frac"] < 0.2:
        return "mostly_silent"
    if not (row["tempo_valid_frac"] >= 0.5) or row["pk_n_est"] < 2:
        return "never_acquired"
    lv = row.get("pk_lvl_best", "beat")
    if lv in ("offbeat", "double", "half") and row.get("pk_F70_lvlfree", 0.0) >= ok_f:
        return {"offbeat": "offbeat_phase", "double": "double_time", "half": "half_time"}[lv]
    if not (row["tempo_acc1"] >= 0.5):
        return "tempo_wrong"
    if row["pk_F70_debias"] >= ok_f:
        return "constant_latency"
    if row["pk_F200"] >= ok_f:
        return "timing_jitter"
    return "phase_lost"


def blame(row, cause, cfg):
    if cause in ("ok", "n/a"):
        return ""
    if cause == "mostly_silent":
        return "input"
    names = oracle_names(cfg)
    if not names:
        return "n/a"
    o = row.get(f"{names[0]}_F70", np.nan)
    if not np.isfinite(o):
        return "n/a"
    return "front-end" if o >= cfg["ok_f"] else "back-end"


def filter_active_events(times, active, fps):
    if times.size == 0 or active.size == 0:
        return times
    idx = np.clip(np.rint(times * fps).astype(int), 0, active.size - 1)
    return times[active[idx]]


def evaluate_range(P, t0, t1, cfg, track_level=False):
    ft, fps = P["frame_times"], P["fps"]
    frame_ms = 1000.0 / fps
    t0e = max(t0, TRIM_SEC)
    f0, f1 = int(np.searchsorted(ft, t0e, "left")), int(np.searchsorted(ft, t1, "left"))
    if f1 - f0 < 30:
        return None
    sl = slice(f0, f1)
    R = P["runs"]["real"]
    active = R["active"]

    active_frac = float(active[sl].mean())
    min_af = cfg["min_track_active_frac"] if track_level else cfg["min_window_active_frac"]
    if active_frac < min_af:
        return None

    gt_ev = filter_active_events(P["gt_beats"], active, fps) if cfg["eval_active_only"] else P["gt_beats"]
    if ((gt_ev >= t0e) & (gt_ev < t1)).sum() < 4:
        return None

    def _ev(times):
        return filter_active_events(times, active, fps) if cfg["eval_active_only"] else times

    row, ex = {}, {"off": {}, "sweep": {}, "off_act": np.array([])}

    m, o, sw = event_metrics(gt_ev, _ev(R["pk_times"]), t0e, t1, "pk_", sweep=track_level)
    row.update(m)
    ex["off"]["pk"], ex["sweep"]["pk"] = o, sw

    m, _, _ = event_metrics(gt_ev, _ev(R["flag_times"]), t0e, t1, "flag_", light=True)
    row.update({k: v for k, v in m.items() if k.split("_", 1)[1] in ("F70", "F200", "off_med_ms", "n_est")})

    act_mask = active[sl] if cfg["eval_active_only"] else np.ones(f1 - f0, dtype=bool)
    gt_pos_m = P["gt_pos"][sl][act_mask]
    gt_bpm_m = P["gt_bpm"][sl][act_mask]

    for name in [n for n in P["runs"] if n != "real"]:
        Ro = P["runs"][name]
        m, o, sw = event_metrics(gt_ev, _ev(Ro["pk_times"]), t0e, t1, f"{name}_", sweep=track_level, light=True)
        row.update(m)
        ex["off"][name], ex["sweep"][name] = o, sw
        row[f"{name}_plv_k1"] = phase_metrics(Ro["sin"][sl][act_mask], Ro["cos"][sl][act_mask],
                                              gt_pos_m, gt_bpm_m)["plv_k1"]
        row[f"{name}_tempo_acc1"] = tempo_metrics(Ro["tempo"][sl][act_mask], gt_bpm_m, fps)["tempo_acc1"]

    pm = phase_metrics(R["sin"][sl][act_mask], R["cos"][sl][act_mask], gt_pos_m, gt_bpm_m)
    row.update(pm)
    k = pm["level_k"] if np.isfinite(pm["level_k"]) else 1.0
    row.update(feature_corr(R["sin"][sl], R["cos"][sl], P["gt_pos"][sl], k, frame_ms))
    row.update(tempo_metrics(R["tempo"][sl][act_mask], gt_bpm_m, fps))
    row["tempo_gt_med"] = float(np.median(gt_bpm_m)) if gt_bpm_m.size else np.nan
    row["gt_fold_log2"] = float(np.log2(P["gt_fold"]))
    row["gt_in_range"] = bool(P["gt_fold"] == 1.0)
    row["active_frac"] = active_frac
    row["trusted_frac"] = float(R["trusted"][sl].mean())

    onsets_ev = _ev(P["onsets"])
    fm, off_act = frontend_metrics(onsets_ev, P["act"], fps, gt_ev, t0e, t1, lead_sec=P["primary_lead_sec"])
    row.update(fm)
    ex["off_act"] = off_act
    row.update(region_hits(P["gt_beats"], R["pk_times"], active, fps, t0e, t1))
    if track_level:
        row.update(acquisition_metrics(gt_ev, _ev(R["pk_times"])))

    row["cause"] = classify(row, cfg["ok_f"], cfg["eval_active_only"])
    row["blame"] = blame(row, row["cause"], cfg)
    return row, ex


def lock_stats(lock, fps):
    lock = np.asarray(lock, dtype=bool)
    first = np.nonzero(lock)[0]
    t0 = int(math.ceil(TRIM_SEC * fps))
    return {"lock_on_sec": float(first[0] / fps) if first.size else np.nan,
            "lock_frac": float(lock[t0:].mean()) if lock.size > t0 else np.nan,
            "n_unlocks": int(np.sum(lock[:-1] & ~lock[1:]))}


def process_item(args):
    track, stem, cfg, paths, recompute = args
    try:
        P = prepare_track(track, stem, cfg, paths, recompute)
        sr = P["sr"]
        windows = []
        for start in range(0, len(P["mono"]) - DURATION_SEC * sr + 1, STRIDE_SEC * sr):
            res = evaluate_range(P, start / sr, start / sr + DURATION_SEC, cfg)
            if res is None:
                continue
            row, _ = res
            row.update(track=track, stem=stem, start_sec=start / sr, level="window")
            windows.append(row)

        track_row, payload = None, None
        res = evaluate_range(P, 0.0, P["duration"], cfg, track_level=True)
        if res is not None:
            row, ex = res
            row.update(track=track, stem=stem, level="track", duration=P["duration"],
                       gt_bpm_raw=60.0 / float(np.median(np.diff(P["beats_raw"]))),
                       **lock_stats(P["runs"]["real"]["lock"], P["fps"]),
                       n_tempo_changes=int(len(np.unique(np.round(np.diff(P["beats_raw"]), 3)))))
            track_row = row
            R = P["runs"]["real"]
            gt_pay = filter_active_events(P["gt_beats"], R["active"], P["fps"]) if cfg["eval_active_only"] else P["gt_beats"]
            pk_pay = filter_active_events(R["pk_times"], R["active"], P["fps"]) if cfg["eval_active_only"] else R["pk_times"]
            payload = dict(gt_beats=gt_pay, gt_pos=P["gt_pos"], gt_bpm=P["gt_bpm"], pk_times=pk_pay,
                           sin=R["sin"], cos=R["cos"], duration=P["duration"], fps=P["fps"],
                           frame_ms=1000.0 / P["fps"], off=ex["off"], sweep=ex["sweep"], off_act=ex["off_act"])
        return {"windows": windows, "track": track_row, "payload": payload}
    except Exception as e:
        return {"skip": f"{track}/{stem}: {type(e).__name__}: {e}"}


def add_chance(track_rows, payloads, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(payloads.keys())
    for r in track_rows:
        r.update({"chance_pk_F70": np.nan, "chance_pk_AMLt": np.nan, "chance_plv": np.nan,
                  "chance_plv_k1": np.nan, "chance_feat_corr_best": np.nan})
        others = [k for k in keys if k[0] != r["track"]]
        if not others:
            continue
        p, o = payloads[(r["track"], r["stem"])], payloads[others[int(rng.integers(len(others)))]]
        dur = min(p["duration"], o["duration"])
        m, _, _ = event_metrics(o["gt_beats"], p["pk_times"], TRIM_SEC, dur, "pk_")
        f0 = int(math.ceil(TRIM_SEC * p["fps"]))
        L = min(len(p["sin"]), len(o["gt_pos"]))
        pm = phase_metrics(p["sin"][f0:L], p["cos"][f0:L], o["gt_pos"][f0:L], o["gt_bpm"][f0:L])
        k = pm["level_k"] if np.isfinite(pm["level_k"]) else 1.0
        fm = feature_corr(p["sin"][f0:L], p["cos"][f0:L], o["gt_pos"][f0:L], k, p["frame_ms"])
        r["chance_pk_F70"], r["chance_pk_AMLt"] = m["pk_F70"], m["pk_AMLt"]
        r["chance_plv"], r["chance_plv_k1"] = pm["plv"], pm["plv_k1"]
        r["chance_feat_corr_best"] = fm["feat_corr_best"]


def aggregate(payloads, cfg):
    names = ["pk"] + compare_names(cfg)
    ps = list(payloads.values())
    off = {}
    for n in names:
        arrs = [p["off"][n] for p in ps if p["off"].get(n) is not None and p["off"][n].size]
        off[n] = np.concatenate(arrs) if arrs else np.array([])
    sweep = {}
    for n in names:
        cs = [p["sweep"][n] for p in ps if p["sweep"].get(n) is not None]
        sweep[n] = np.mean(np.stack(cs), axis=0) if cs else None
    arrs = [p["off_act"] for p in ps if p["off_act"].size]
    return dict(off=off, sweep=sweep, off_act=np.concatenate(arrs) if arrs else np.array([]))


# ----------------------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------------------
def mean_col(df, c):
    return float(df[c].mean()) if (c in df.columns and df[c].notna().any()) else np.nan


def med_col(df, c):
    return float(df[c].median()) if (c in df.columns and df[c].notna().any()) else np.nan


def best_shift(curve):
    top = float(np.max(curve))
    idx = np.flatnonzero(curve >= top - 1e-6)
    sh = SHIFTS_MS[idx]
    return int(sh[np.argmin(np.abs(sh - np.mean(sh)))]), top


def summarize(trk, win, cfg, agg):
    out = []

    def P(s=""):
        print(s)
        out.append(str(s))

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 60)
    fps = 48000.0 / cfg["hop"]
    frame_ms = 1000.0 / fps
    orcs = oracle_names(cfg)
    orc0 = orcs[0] if orcs else None
    has_btl_cmp = "btl_F70" in trk.columns
    primary_label = btl_mode_label(cfg, short=False) if cfg.get("btl_primary") else "spectral flux"

    P(f"=== Track-level summary ({len(trk)} tracks/stems, {len(win)} windows | primary: {primary_label}) ===")
    cols = ["pk_F70", "pk_F200", "pk_AMLt", "pk_CMLt", "flag_F70", "plv", "plv_k1", "phase_std_deg",
            "feat_corr_best", "feat_lag_ms", "tempo_acc1", "tempo_acc2", "locked_frac", "active_frac",
            "lock_on_sec", "lock_frac", "n_unlocks", "acq_sec", "chance_pk_F70", "chance_plv_k1"]
    if has_btl_cmp:
        cols = ["pk_F70", "btl_F70", "btl_F200", "btl_plv_k1", "btl_tempo_acc1"] + cols[1:]
    cols = [c for c in cols if c in trk.columns]
    P(trk[cols].agg(["mean", "median"]).T.round(3).to_string())

    P("\n=== Gap decomposition (mean over tracks) ===")
    rows = [(f"F70  primary tracker ({primary_label})", mean_col(trk, "pk_F70"))]
    sw = agg["sweep"].get("pk")
    if sw is not None:
        sh_best, f_best = best_shift(sw)
        rows.append((f"F70  best GLOBAL shift ({sh_best:+d} ms on all predictions)", f_best))
    rows += [("F70  per-track de-biased (median offset removed)", mean_col(trk, "pk_F70_debias")),
             ("F70  at any metrical level (beat/off-beat/x2/x0.5)", mean_col(trk, "pk_F70_lvlfree")),
             ("F100 primary tracker as is", mean_col(trk, "pk_F100")),
             ("F200 primary tracker as is", mean_col(trk, "pk_F200"))]
    if has_btl_cmp:
        tag_lbl = btl_mode_label(cfg, short=True)
        rows.append((f"F70  {tag_lbl}", mean_col(trk, "btl_F70")))
        sw_btl = agg["sweep"].get("btl")
        if sw_btl is not None:
            sh_b, f_b = best_shift(sw_btl)
            rows.append((f"F70  {tag_lbl} best GLOBAL shift ({sh_b:+d} ms)", f_b))
        rows.append((f"F70  {tag_lbl} per-track de-biased", mean_col(trk, "btl_F70_debias")))
        rows.append((f"F200 {tag_lbl}", mean_col(trk, "btl_F200")))
    for n in orcs:
        rows.append((f"F70  ORACLE activation [{n}]  (back-end only)", mean_col(trk, f"{n}_F70")))
        rows.append((f"F200 ORACLE activation [{n}]", mean_col(trk, f"{n}_F200")))
    for lab, v in rows:
        P(f"  {lab:<62s} {v:6.3f}")

    real_off, real_mad = med_col(trk, "pk_off_med_ms"), med_col(trk, "pk_off_mad_ms")
    P("\n=== Timing: bias vs jitter (signed, > 0 = tracker late) ===")
    P(f"  primary : per-track median offset (bias)   {real_off:+7.1f} ms | per-track MAD (jitter) {real_mad:6.1f} ms "
      f"| p90 |dev| {med_col(trk, 'pk_off_p90_ms'):6.1f} ms")
    if "pk_off_med_ms" in trk.columns:
        P(f"            spread of the per-track bias across tracks (IQR): "
          f"{trk['pk_off_med_ms'].quantile(0.75) - trk['pk_off_med_ms'].quantile(0.25):.1f} ms")
    if "flag_off_med_ms" in trk.columns:
        P(f"  detect_beat events: median offset {med_col(trk, 'flag_off_med_ms'):+.1f} ms, "
          f"F70 {mean_col(trk, 'flag_F70'):.3f} (quantised to the {frame_ms:.1f} ms frame grid)")
    if has_btl_cmp:
        b_off, b_mad = med_col(trk, "btl_off_med_ms"), med_col(trk, "btl_off_mad_ms")
        P(f"  btl     : per-track median offset (bias)   {b_off:+7.1f} ms | per-track MAD (jitter) {b_mad:6.1f} ms "
          f"| p90 |dev| {med_col(trk, 'btl_off_p90_ms'):6.1f} ms")
    if orc0:
        o_off, o_mad = med_col(trk, f"{orc0}_off_med_ms"), med_col(trk, f"{orc0}_off_mad_ms")
        P(f"  oracle  : median offset (back-end / time convention) {o_off:+7.1f} ms | MAD {o_mad:6.1f} ms")
        if np.isfinite(real_off) and np.isfinite(o_off):
            P(f"  => primary front-end latency (primary - oracle){real_off - o_off:+7.1f} ms")
            if has_btl_cmp and np.isfinite(med_col(trk, "btl_off_med_ms")):
                P(f"  => BTL front-end latency (btl - oracle)        {med_col(trk, 'btl_off_med_ms') - o_off:+7.1f} ms")
            P(f"  => back-end / time-convention bias (oracle)    {o_off:+7.1f} ms "
              f"(pk_offset_frames now {cfg['pk_offset_frames']:.2f}; a pure relabel would be "
              f"{cfg['pk_offset_frames'] - o_off / frame_ms:.2f})")
    if np.isfinite(real_off):
        P(f"  Residual phase lead to reach 0 ms: add {real_off:+.1f} ms to phase_lead_ms")

    P(f"\n=== Front end ({primary_label} peaks vs GT beats, no tracker involved) ===")
    for lab, c in [("onsets per GT beat", "onsets_per_beat"), ("GT beats with an onset within 50 ms", "onset_rec50"),
                   ("GT beats with an onset within 100 ms", "onset_rec100"),
                   ("onsets within 100 ms of a GT beat (precision)", "onset_prec100"),
                   ("activation at beats / mean activation", "beat_act_ratio"),
                   ("onset offset to nearest beat, median (ms)", "act_off_med_ms"),
                   ("onset offset MAD (ms)", "act_off_mad_ms")]:
        P(f"  {lab:<50s} mean {mean_col(trk, c):8.3f}   median {med_col(trk, c):8.3f}")

    P("\n=== GT-beat hit rate (<= 70 ms) by input region ===")
    for r in REGIONS:
        c, n = f"hit70_{r}", f"n_{r}"
        if c in trk.columns:
            tot = int(trk[n].sum())
            w = trk[n].to_numpy(dtype=float)
            v = trk[c].to_numpy(dtype=float)
            m = np.isfinite(v) & (w > 0)
            wavg = float(np.sum(v[m] * w[m]) / max(np.sum(w[m]), 1.0)) if m.any() else np.nan
            P(f"  {r:<8s} beats={tot:6d}   hit rate (beat-weighted) {wavg:6.3f}   (track mean {mean_col(trk, c):.3f})")
    P(f"  time to first 4 consecutive hits: median {med_col(trk, 'acq_sec'):.1f} s | "
      f"hit rate after acquisition {mean_col(trk, 'hit70_post_acq'):.3f}")

    P("\n=== Tempo ===")
    P(f"  acc1 {mean_col(trk, 'tempo_acc1'):.3f} | acc2 (octave-tolerant) {mean_col(trk, 'tempo_acc2'):.3f} | "
      f"median |err| {med_col(trk, 'tempo_med_abs_err_pct'):.2f} % | jumps/min {mean_col(trk, 'tempo_jumps_per_min'):.2f}")

    r = trk["pk_ibi_ratio"].dropna() if "pk_ibi_ratio" in trk.columns else pd.Series(dtype=float)
    if len(r):
        P("\nMetrical level of the tracker events (median tracker interval / GT interval):")
        P(f"  same level    : {(np.abs(r - 1) < 0.1).mean():.1%}")
        P(f"  half-time     : {(np.abs(r - 2) < 0.2).mean():.1%}   (tracker slower than GT)")
        P(f"  double-time   : {(np.abs(r - 0.5) < 0.1).mean():.1%}   (tracker faster than GT)")
        for v in ("beat", "offbeat", "double", "half"):
            c = "pk_F70" if v == "beat" else f"pk_Fv_{v}"
            P(f"  mean F70 vs GT variant '{v}': {mean_col(trk, c):.3f}")

    lk = trk["level_k"].dropna()
    if len(lk):
        P("\nBest metrical level of the tracker PHASE vs the unfolded GT (tracker beat rate = k x GT beat rate):")
        for k in sorted(LEVELS):
            P(f"  k = {k:<5.3g}: {np.isclose(lk, k).mean():.1%}")
        P(f"  phase locked (PLV > {PLV_OK}) at the native level k=1 : {(trk['plv_k1'] > PLV_OK).mean():.1%}")
        P(f"  phase locked (PLV > {PLV_OK}) at any level            : {(trk['plv'] > PLV_OK).mean():.1%}")

    if "gt_fold_log2" in trk.columns and trk["gt_fold_log2"].notna().any():
        grp = np.where(trk["gt_fold_log2"] > 0, "folded UP (slow song, no info lost)",
                       np.where(trk["gt_fold_log2"] < 0, "folded DOWN (fast song, phase ambiguous)",
                                "native (in range)"))
        P(f"\nBy GT fold into the tracker's octave [{cfg['low_bpm']}, {cfg['high_bpm']}]:")
        keep = [c for c in ["pk_F70", "btl_F70", "pk_F200", "plv_k1", "tempo_acc1", "tempo_acc2", "lock_frac"]
                if c in trk.columns]
        P(trk.groupby(grp)[keep].agg(["mean", "count"]).round(3).to_string())

    for name, df in (("windows", win), ("tracks", trk)):
        if not len(df) or "cause" not in df.columns:
            continue
        P(f"\n=== Failure taxonomy ({name}, ok = F70 >= {cfg['ok_f']}) ===")
        g = df.groupby("cause").agg(n=("cause", "size"), F70=("pk_F70", "mean"), F200=("pk_F200", "mean"),
                                    F70_debias=("pk_F70_debias", "mean"))
        g["share"] = g["n"] / g["n"].sum()
        g = g.reindex([c for c in CAUSES if c in g.index])
        P(g.round(3).to_string())
        bad = df[~df["cause"].isin(["ok", "n/a"])]
        if len(bad):
            P(f"\n  non-ok {name} by cause x blame (front-end = oracle F70 >= {cfg['ok_f']}, back-end = oracle also fails):")
            P(pd.crosstab(bad["cause"], bad["blame"]).to_string())

    P("\n=== Reading guide (automatic hints, check against the plots) ===")
    r70 = mean_col(trk, "pk_F70")
    if has_btl_cmp:
        b70 = mean_col(trk, "btl_F70")
        P(f"  * Beat This! Live lifts F70 {r70:.3f} -> {b70:.3f} ({b70 - r70:+.3f}) on the exact same 30 FPS PLP back-end.")
    if orc0:
        o70 = mean_col(trk, f"{orc0}_F70")
        if o70 - r70 > 0.15:
            P(f"  * Perfect onsets lift F70 {r70:.2f} -> {o70:.2f}: the activation / onset side is the main limiter.")
        elif o70 < 0.8:
            P(f"  * Even perfect onsets only reach F70 {o70:.2f}: the back-end loses accuracy by itself.")
        else:
            P(f"  * Oracle F70 {o70:.2f} is close to the primary {r70:.2f}: the onsets are not what limits F70.")
        if len(orcs) > 1:
            P(f"  * Oracle with {orcs[1]} ms-style jitter: F70 {mean_col(trk, f'{orcs[1]}_F70'):.2f}.")
    if sw is not None and float(np.max(sw)) - r70 > 0.05:
        P(f"  * A single global shift of {best_shift(sw)[0]:+d} ms lifts F70 {r70:.2f} -> {float(np.max(sw)):.2f}.")
    return "\n".join(out)


def summary_plot(trk, win, offs_pk, path, frame_ms, cfg):
    fig, axs = plt.subplots(3, 3, figsize=(17, 13))
    b = np.linspace(0, 1, 21)
    axs[0, 0].hist(trk["pk_F70"].dropna(), b, alpha=0.7, label="primary tracker vs GT")
    if "btl_F70" in trk.columns:
        axs[0, 0].hist(trk["btl_F70"].dropna(), b, alpha=0.55, label="Beat This! Live vs GT")
    axs[0, 0].hist(trk["chance_pk_F70"].dropna(), b, alpha=0.45, label="chance (other track's GT)")
    axs[0, 0].set_title("Beat F-measure (+-70 ms), phase-wrap events, native level"); axs[0, 0].legend()

    axs[0, 1].hist(trk["plv"].dropna(), b, alpha=0.7, label="best level")
    axs[0, 1].hist(trk["plv_k1"].dropna(), b, alpha=0.5, label="native level (k=1)")
    axs[0, 1].set_title("Phase-locking value of (sin, cos) vs GT phase"); axs[0, 1].legend()

    if offs_pk.size:
        axs[0, 2].hist(offs_pk, np.linspace(-150, 150, 61))
    axs[0, 2].axvline(0, color="k", lw=0.8)
    axs[0, 2].set_title("Signed timing offset of predicted beats (ms, > 0 = late)")

    lr = np.log2(trk["pk_ibi_ratio"].dropna())
    axs[1, 0].hist(lr, np.linspace(-2, 2, 41))
    axs[1, 0].set_title("log2(tracker beat interval / GT)  (0 ok, +1 half-time, -1 double-time)")

    if len(win):
        axs[1, 1].scatter(win["active_frac"], win["pk_F70"], s=6, alpha=0.3)
    axs[1, 1].set_xlabel("fraction of window with active guitar"); axs[1, 1].set_ylabel("F70")

    axs[1, 2].hist(trk["phase_off_ms"].dropna(), np.linspace(-250, 250, 51))
    axs[1, 2].axvline(0, color="k", lw=0.8)
    axs[1, 2].set_title("Phase timing offset at the best level (ms, > 0 = late)")

    g = trk[["tempo_gt_med", "tempo_pred_med"]].dropna()
    if len(g):
        axs[2, 0].scatter(g["tempo_gt_med"], g["tempo_pred_med"], s=8, alpha=0.4)
        xs = np.array([20.0, 460.0])
        for kk, ls in ((1.0, "-"), (2.0, "--"), (0.5, "--")):
            axs[2, 0].plot(xs, xs * kk, color="k", lw=0.7, ls=ls)
        axs[2, 0].axvspan(cfg["low_bpm"], cfg["high_bpm"], color="green", alpha=0.08)
        axs[2, 0].axhspan(cfg["low_bpm"], cfg["high_bpm"], color="green", alpha=0.08)
        axs[2, 0].set_xscale("log"); axs[2, 0].set_yscale("log")
    axs[2, 0].set_title("Tempo (lines: k = 1, 2, 0.5; green: tracker range)")

    lk = trk["level_k"].dropna()
    ks = sorted(LEVELS)
    axs[2, 1].bar([f"{k:.3g}" for k in ks], [np.isclose(lk, k).mean() if len(lk) else 0 for k in ks])
    axs[2, 1].set_title("Best metrical level k (tracker rate = k x GT rate)")

    if len(win):
        axs[2, 2].hist(win["feat_lag_ms"].dropna(),
                       np.arange(-LAG_MAX_FRAMES - 0.5, LAG_MAX_FRAMES + 1.5) * frame_ms)
    axs[2, 2].set_title("Best lag of tracker sin/cos vs GT (ms, > 0 = late)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def diagnosis_plot(trk, win, agg, path, cfg):
    orcs = oracle_names(cfg)
    names = ["pk"] + compare_names(cfg)
    pk_lab = btl_mode_label(cfg, short=True) if cfg.get("btl_primary") else "spectral flux"
    btl_lab = btl_mode_label(cfg, short=True)
    labels = {"pk": pk_lab, "btl": btl_lab, **{n: f"oracle ({n})" for n in orcs}}
    colors = {"pk": "tab:red", "btl": "tab:green",
              **{n: c for n, c in zip(orcs, ["tab:blue", "tab:cyan", "tab:purple"])}}
    fig, axs = plt.subplots(3, 4, figsize=(22, 14))

    ms = [int(round(t * 1000)) for t in TOLS]
    for n in names:
        ys = [mean_col(trk, f"{n}_{fkey(t)}") for t in TOLS]
        axs[0, 0].plot(ms, ys, "o-", color=colors[n], label=labels[n])
    axs[0, 0].axvline(70, color="k", lw=0.6, ls=":")
    axs[0, 0].set_xlabel("tolerance (ms)"); axs[0, 0].set_ylabel("mean F")
    axs[0, 0].set_title("F-measure vs tolerance"); axs[0, 0].legend()

    for n in names:
        if agg["sweep"].get(n) is not None:
            axs[0, 1].plot(SHIFTS_MS, agg["sweep"][n], color=colors[n], label=labels[n])
    axs[0, 1].axvline(0, color="k", lw=0.6)
    axs[0, 1].set_xlabel("shift added to predicted beats (ms)"); axs[0, 1].set_ylabel("mean F70")
    axs[0, 1].set_title("Global shift sweep"); axs[0, 1].legend()

    bins = np.linspace(-150, 150, 61)
    hist_names = ["pk"] + (["btl"] if "btl" in names else []) + (orcs[:1] if orcs else [])
    for n in hist_names:
        if agg["off"].get(n) is not None and agg["off"][n].size:
            axs[0, 2].hist(agg["off"][n], bins, alpha=0.45, density=True, color=colors[n], label=labels[n])
    axs[0, 2].axvline(0, color="k", lw=0.8); axs[0, 2].legend()
    axs[0, 2].set_title("Signed offset of predicted beats (ms, > 0 = late)")

    if "pk_F70_debias" in trk.columns:
        axs[0, 3].scatter(trk["pk_F70"], trk["pk_F70_debias"], s=10, alpha=0.5)
        axs[0, 3].plot([0, 1], [0, 1], "k", lw=0.6)
        axs[0, 3].set_xlabel("F70"); axs[0, 3].set_ylabel("F70 after removing per-track median offset")
        axs[0, 3].set_title("Above diagonal = constant bias")

    if len(win):
        cnt = win["cause"].value_counts().reindex([c for c in CAUSES if c in set(win["cause"])]).fillna(0)
        axs[1, 0].bar(range(len(cnt)), cnt.values, color=plt.cm.tab10(np.arange(len(cnt)) % 10))
        axs[1, 0].set_xticks(range(len(cnt))); axs[1, 0].set_xticklabels(cnt.index, rotation=40, ha="right")
        axs[1, 0].set_title("Failure cause (windows)")

    if agg["off_act"].size:
        axs[1, 1].hist(agg["off_act"], np.linspace(-150, 150, 61))
    axs[1, 1].axvline(0, color="k", lw=0.8)
    axs[1, 1].set_title("Front end only: activation peak - nearest GT beat (ms)")

    for c, lab in (("onset_rec50", "GT beats with onset <=50ms"), ("onset_prec100", "onsets within 100ms of a beat")):
        if c in trk.columns:
            axs[1, 2].hist(trk[c].dropna(), np.linspace(0, 1, 21), alpha=0.55, label=lab)
    axs[1, 2].legend(); axs[1, 2].set_title("Onset recall / precision per track")

    hv = [mean_col(trk, f"hit70_{r}") for r in REGIONS]
    axs[1, 3].bar(REGIONS, [0 if not np.isfinite(v) else v for v in hv])
    axs[1, 3].set_ylim(0, 1); axs[1, 3].set_title("GT beat hit rate (<=70 ms) by region")

    if "btl_F70" in trk.columns:
        axs[2, 0].scatter(trk["pk_F70"], trk["btl_F70"], s=12, alpha=0.55, color="tab:green")
        axs[2, 0].plot([0, 1], [0, 1], "k", lw=0.6)
        axs[2, 0].set_xlabel("F70 spectral flux"); axs[2, 0].set_ylabel(f"F70 {btl_lab}")
        axs[2, 0].set_title(f"Above diagonal = {btl_lab} improves over spectral flux")
    elif orcs and f"{orcs[0]}_F70" in trk.columns:
        o = orcs[0]
        axs[2, 0].scatter(trk["pk_F70"], trk[f"{o}_F70"], s=10, alpha=0.5)
        axs[2, 0].plot([0, 1], [0, 1], "k", lw=0.6)
        axs[2, 0].set_xlabel(f"F70 ({pk_lab})"); axs[2, 0].set_ylabel(f"F70 oracle ({o})")
        axs[2, 0].set_title("Above diagonal = front-end loses; flat at low = back-end")

    vv = [("beat", "pk_F70"), ("offbeat", "pk_Fv_offbeat"), ("double", "pk_Fv_double"), ("half", "pk_Fv_half")]
    axs[2, 1].bar([a for a, _ in vv], [0 if not np.isfinite(mean_col(trk, c)) else mean_col(trk, c) for _, c in vv])
    axs[2, 1].set_ylim(0, 1); axs[2, 1].set_title("Mean F70 against GT metrical variants")

    if "tempo_acc1" in trk.columns:
        axs[2, 2].hist(trk["tempo_acc1"].dropna(), np.linspace(0, 1, 21), alpha=0.6, label="acc1")
        axs[2, 2].hist(trk["tempo_acc2"].dropna(), np.linspace(0, 1, 21), alpha=0.5, label="acc2 (octave-tolerant)")
        axs[2, 2].legend(); axs[2, 2].set_title("Tempo accuracy per track")

    if len(win):
        axs[2, 3].scatter(win["pk_off_med_ms"], win["pk_off_mad_ms"], s=6, alpha=0.3,
                          c=(win["pk_F70"].fillna(0)).values, cmap="viridis")
        axs[2, 3].set_xlabel("window median offset = bias (ms)"); axs[2, 3].set_ylabel("window MAD = jitter (ms)")
        axs[2, 3].set_title("Bias vs jitter per window (colour = F70)")
        axs[2, 3].set_xlim(-150, 150)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def _fmt(x):
    return f"{x:.2f}" if (x is not None and np.isfinite(x)) else "nan"


def render_window(row, cfg, paths, tag, rank):
    P = prepare_track(row["track"], row["stem"], cfg, paths, recompute=False)
    R = P["runs"]["real"]
    btl = P["runs"].get("btl")
    orc_names = [n for n in P["runs"] if n not in ("real", "btl")]
    orc = P["runs"][orc_names[0]] if orc_names else None
    sr, fps = P["sr"], P["fps"]
    t0 = float(row["start_sec"])
    t1 = t0 + DURATION_SEC
    crop = P["mono"][int(round(t0 * sr)):int(round(t1 * sr))]
    t_axis = t0 + np.arange(len(crop)) / sr
    ft = P["frame_times"]
    m = (ft >= t0) & (ft < t1)
    sel = lambda a: a[(a >= t0) & (a < t1)]
    gt_in, pk_in, on_in = sel(P["gt_beats"]), sel(R["pk_times"]), sel(P["onsets"])
    btl_in = sel(btl["pk_times"]) if btl is not None else np.array([])
    orc_in = sel(orc["pk_times"]) if orc is not None else np.array([])

    fig, axs = plt.subplots(5, 1, figsize=(16, 14), sharex=True, height_ratios=[1.4, 1, 1, 0.8, 0.9])

    axs[0].plot(t_axis, crop / (np.abs(crop).max() + 1e-9), color="lightgray", label="guitar waveform")
    axs[0].vlines(gt_in, 0, 1.1, color="green", lw=2, label="GT beats (folded)")
    pk_lab = f"{btl_mode_label(cfg, short=True)} beats" if cfg.get("btl_primary") else "tracker beats"
    axs[0].vlines(pk_in, -1.1, -0.55, color="red", lw=2, ls="--", label=pk_lab)
    if btl is not None:
        axs[0].vlines(btl_in, -0.55, -0.15, color="tab:green", lw=2, ls="-.",
                      label=f"{btl_mode_label(cfg, short=True)} beats")
    if orc is not None:
        y_lo = -0.15 if btl is not None else -0.5
        axs[0].vlines(orc_in, y_lo, 0, color="tab:blue", lw=2, ls=":", label=f"oracle beats ({orc_names[0]})")
    axs[0].set_ylim(-1.1, 1.1); axs[0].legend(loc="upper right", ncol=3)
    btl_str = f" btl F70={_fmt(row.get('btl_F70', np.nan))} |" if btl is not None else ""
    axs[0].set_title(f"{row['track']}/{row['stem']} start={t0:.0f}s | cause={row['cause']} ({row['blame']}) | "
                     f"F70={_fmt(row['pk_F70'])} F200={_fmt(row['pk_F200'])} debiased={_fmt(row['pk_F70_debias'])} "
                     f"off={_fmt(row['pk_off_med_ms'])}ms mad={_fmt(row['pk_off_mad_ms'])}ms |{btl_str} oracle F70="
                     f"{_fmt(row.get(f'{orc_names[0]}_F70', np.nan) if orc_names else np.nan)} | "
                     f"PLV={_fmt(row['plv'])} (native {_fmt(row['plv_k1'])}) active={_fmt(row['active_frac'])}")

    axs[1].plot(ft[m], P["gt_cos"][m], color="green", label="GT cos")
    axs[1].plot(ft[m], R["cos"][m], color="red", ls="--", label="primary cos")
    axs[1].plot(ft[m], P["gt_sin"][m], color="green", lw=0.8, ls=":", alpha=0.7, label="GT sin")
    axs[1].plot(ft[m], R["sin"][m], color="red", lw=0.8, ls=":", alpha=0.7, label="primary sin")
    if btl is not None:
        axs[1].plot(ft[m], btl["cos"][m], color="tab:green", lw=1.0, alpha=0.85, label="BTL cos")
    if orc is not None:
        axs[1].plot(ft[m], orc["cos"][m], color="tab:blue", lw=0.9, alpha=0.8, label="oracle cos")
    axs[1].fill_between(ft[m], -1.1, 1.1, where=R["lock"][m], color="tab:blue", alpha=0.07, step="post",
                        label="locked (PLL)")
    axs[1].fill_between(ft[m], -1.1, -1.0, where=~R["active"][m], color="black", alpha=0.5, step="post",
                        label="silence gate")
    axs[1].set_ylim(-1.1, 1.1); axs[1].legend(loc="upper right", ncol=4); axs[1].set_ylabel("sin / cos")

    axs[2].plot(ft[m], P["gt_bpm"][m], color="green", label="GT tempo (folded)")
    axs[2].plot(ft[m], np.where(R["tempo"][m] > 0, R["tempo"][m], np.nan), color="red", ls="--", label="primary tempo")
    if btl is not None:
        axs[2].plot(ft[m], np.where(btl["tempo"][m] > 0, btl["tempo"][m], np.nan), color="tab:green", lw=1.0,
                    label="BTL tempo")
    if orc is not None:
        axs[2].plot(ft[m], np.where(orc["tempo"][m] > 0, orc["tempo"][m], np.nan), color="tab:blue", lw=0.9,
                    label="oracle tempo")
    axs[2].set_ylabel("BPM"); axs[2].legend(loc="upper right")

    axs[3].plot(ft[m], R["conf"][m], color="orange", lw=1.2, label="conf (Peak/Mean Ratio)")
    axs[3].axhline(cfg["conf_threshold"], color="orange", ls="--", alpha=0.5, label="conf threshold")
    axs[3].axhline(cfg["lock_conf_threshold"], color="red", ls=":", alpha=0.6, label="lock conf threshold")
    axs[3].set_ylabel("conf"); axs[3].legend(loc="upper left")
    ax3b = axs[3].twinx()
    ax3b.plot(ft[m], R["lock_score"][m], color="tab:blue", lw=1.2, label="lock score (0 to 1)")
    ax3b.set_ylim(0, 1.05); ax3b.set_ylabel("lock score"); ax3b.legend(loc="upper right")

    t30 = np.arange(len(P["act"])) / fps + P["primary_lead_sec"]
    m30 = (t30 >= t0) & (t30 < t1)
    a = np.asarray(P["act"], dtype=np.float64)
    ref = max(float(np.percentile(a, 99)), 1e-9)
    act_lab = f"beat activation ({btl_mode_label(cfg, short=True)}, / p99)" if cfg.get("btl_primary") \
        else "beat activation (spectral flux, / p99)"
    axs[4].plot(t30[m30], a[m30] / ref, color="purple", lw=0.8, label=act_lab)
    axs[4].vlines(gt_in, 0, 1.0, color="green", lw=1, alpha=0.6, label="GT beats")
    if on_in.size:
        axs[4].plot(on_in, np.interp(on_in, t30, a / ref), "rv", ms=5, label="picked onsets")
    axs[4].set_ylim(0, 1.5); axs[4].set_ylabel("activation"); axs[4].set_xlabel("Time (seconds)")
    axs[4].legend(loc="upper right"); axs[4].set_xlim(t0, t1)

    plt.tight_layout()
    base = f"{tag}{rank:02d}_{row['track']}_{row['stem']}_start{int(t0)}s"
    (paths["out"] / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(paths["out"] / "plots" / f"{base}.png", dpi=120)
    plt.close(fig)

    c_gt = librosa.clicks(times=gt_in - t0, sr=sr, length=len(crop), click_freq=900.0, click_duration=0.05)
    c_pr = librosa.clicks(times=pk_in - t0, sr=sr, length=len(crop), click_freq=900.0, click_duration=0.05)
    mix_gt = crop + 0.5 * c_gt
    mix_pr = crop + 0.5 * c_pr
    sf.write(paths["out"] / "plots" / f"{base}_gt.wav", (mix_gt / max(1.0, float(np.abs(mix_gt).max()))).astype(np.float32), sr)
    sf.write(paths["out"] / "plots" / f"{base}_pr.wav", (mix_pr / max(1.0, float(np.abs(mix_pr).max()))).astype(np.float32), sr)


def main():
    args = parse_args()
    paths = make_paths(args)
    paths["out"].mkdir(parents=True, exist_ok=True)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args)
    eff_cfg = btl_tracker_cfg(cfg) if cfg.get("btl_primary") else cfg
    print(f"[INFO] Effective primary tracker settings: { {k: eff_cfg[k] for k in TRACKER_KEYS} } "
          f"btl_fusion={cfg['btl_fusion_mode']} (boost={cfg['btl_flux_boost']}, "
          f"extra_shift_ms={cfg['btl_extra_shift_ms']:+.1f})")
    print(f"[INFO] time convention: pk_offset_frames={cfg['pk_offset_frames']}, "
          f"flag_offset_frames={cfg['flag_offset_frames']}, compare runs={compare_names(cfg) or 'off'}")

    grid_scores, bad = {}, set()
    if paths["report"].exists():
        for k, v in json.load(open(paths["report"])).items():
            gs = v.get("grid_score")
            grid_scores[k] = gs
            if v.get("status") in SKIP_STATUSES or (gs is not None and gs < MIN_GRID_SCORE):
                bad.add(k)

    tracks = sorted(t.name for t in paths["data"].iterdir() if t.is_dir())
    if not args.no_skip_bad_gt:
        tracks = [t for t in tracks if t not in bad]
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]

    items = [(t, gp.stem, cfg, paths, args.recompute) for t in tracks
             for gp in sorted((paths["data"] / t).glob("guitar_*.wav"))]
    print(f"[INFO] Evaluating {len(items)} guitar stems ({args.num_workers} workers).")

    if cfg.get("btl_ckpt"):
        precompute_btl_activations(items, cfg, paths, args.btl_device, recompute=args.btl_recompute_act)

    win_rows, trk_rows, payloads, skipped = [], [], {}, []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = [ex.submit(process_item, it) for it in items]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), ncols=140):
            res = fut.result()
            if "skip" in res:
                skipped.append(res["skip"])
                continue
            win_rows += res["windows"]
            if res["track"] is not None:
                trk_rows.append(res["track"])
                payloads[(res["track"]["track"], res["track"]["stem"])] = res["payload"]

    if skipped:
        (paths["out"] / "skipped.txt").write_text("\n".join(skipped))
        print(f"[INFO] {len(skipped)} stems skipped (see skipped.txt), e.g. {skipped[0]}")
    if not trk_rows:
        return

    add_chance(trk_rows, payloads)
    trk = pd.DataFrame(trk_rows)
    win = pd.DataFrame(win_rows)
    for df in (trk, win):
        if len(df):
            df["grid_score"] = df["track"].map(grid_scores)
    trk.to_csv(paths["out"] / "tracks.csv", index=False)
    win.to_csv(paths["out"] / "windows.csv", index=False)

    agg = aggregate(payloads, cfg)
    sweep_df = pd.DataFrame({"shift_ms": SHIFTS_MS, **{f"F70_{n}": v for n, v in agg["sweep"].items() if v is not None}})
    sweep_df.to_csv(paths["out"] / "shift_sweep.csv", index=False)
    frame_ms = next(iter(payloads.values()))["frame_ms"]

    text = summarize(trk, win, cfg, agg)
    (paths["out"] / "summary.txt").write_text(text)
    summary_plot(trk, win, agg["off"]["pk"], paths["out"] / "summary.png", frame_ms, cfg)
    diagnosis_plot(trk, win, agg, paths["out"] / "diagnosis.png", cfg)

    if not args.no_render and len(win):
        w = win.dropna(subset=["pk_F70"])
        if cfg["plot_exclude_silent"]:
            w = w[w["cause"] != "mostly_silent"]
        w = w.sort_values(["pk_F70", "feat_corr_best"])
        jobs = [("best", w.tail(N_PLOT_BEST).iloc[::-1]), ("worst", w.head(N_PLOT_WORST))]
        for cause, g in w.groupby("cause"):
            if cause not in ("ok", "n/a"):
                jobs.append((f"cause-{cause}-", g.sample(min(N_PLOT_PER_CAUSE, len(g)), random_state=0)))
        for tag, df in jobs:
            for rank, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"render {tag}", ncols=140)):
                render_window(r, cfg, paths, tag, rank)


if __name__ == "__main__":
    main()