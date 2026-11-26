import os
import copy
import json
import math
import time
import random
import pathlib
import argparse
from contextlib import contextmanager
from collections import defaultdict
from typing import Optional

import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch._dynamo
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.cache_size_limit = 64

try:
    import wandb
except ImportError:
    wandb = None

torch.set_float32_matmul_precision("high")

from beat_this.inference import load_model
from beat_this.model.loss import ShiftTolerantBCELoss

from dataset import (
    discover_dataset_items,
    MultiGuitarBeatTrainDataset,
    MultiGuitarBeatValDataset,
    load_beats_tsv,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Beat This! Live Zero on solo-guitar datasets")
    p.add_argument("--config_path", type=str, default="configs/btl_zero_ft.yaml", help="Path to config YAML")
    p.add_argument("--resume", type=str, default=None, help="Path to train_state.pt to resume training")
    p.add_argument("--eval_only", action="store_true", help="Run validation only and exit")
    p.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default=None, help="Override wandb.project")
    p.add_argument("--wandb_name", type=str, default=None, help="Override wandb.name")
    return p.parse_args()


@contextmanager
def _posix_path_compat():
    """Allow Linux-saved Beat This! Live checkpoints containing PosixPath to load on Windows."""
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


def sanitize_hparams(obj):
    """Recursively replace pathlib.Path objects with str so saved .ckpt loads natively on Windows."""
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, dict):
        return {k: sanitize_hparams(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_hparams(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_hparams(v) for v in obj)
    return obj


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


class ModelEMA:
    """Exponential Moving Average of parameters + exact sync of BatchNorm buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for ep, mp in zip(self.ema.parameters(), model.parameters()):
            ep.data.mul_(d).add_(mp.data, alpha=1.0 - d)
        for eb, mb in zip(self.ema.buffers(), model.buffers()):
            eb.copy_(mb)


def init_wandb_run(cfg, args, out_dir: pathlib.Path, resume_run_id: Optional[str] = None):
    """Initialize Weights & Biases run from cfg.wandb + CLI overrides."""
    wb_cfg = OmegaConf.to_container(cfg.get("wandb", {}), resolve=True) or {}
    enabled = bool(wb_cfg.get("enabled", False)) and (not args.no_wandb) and (not args.eval_only)
    mode = str(wb_cfg.get("mode", "online"))
    if not enabled or mode == "disabled":
        return None

    if wandb is None:
        print("[WARN] wandb is enabled in config but not installed (`pip install wandb`); continuing without W&B.")
        return None

    project = args.wandb_project or wb_cfg.get("project", "liveband-beat-tracker")
    entity = wb_cfg.get("entity", None)
    name = args.wandb_name or wb_cfg.get("name", out_dir.name)

    init_kwargs = dict(
        project=project,
        entity=entity,
        name=name,
        mode=mode,
        dir=str(out_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    if resume_run_id is not None:
        init_kwargs["id"] = resume_run_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(**init_kwargs)
    print(f"[INFO] W&B logging enabled: project={project}, run={run.name} (id={run.id}, mode={mode})")
    return run


def flatten_val_metrics_for_wandb(val_summary: dict, epoch: int, best_score: Optional[float] = None) -> dict:
    """Convert nested validation summary dict into flat W&B metric keys."""
    logs = {"epoch": epoch}
    for ds, metrics in val_summary.items():
        for k, v in metrics.items():
            if k == "n_renders":
                continue
            logs[f"val/{ds}/{k}"] = float(v)
    if best_score is not None and best_score >= 0.0:
        logs["val/best_macro_beat_f70_active"] = float(best_score)
    return logs


def resolve_pos_weights(train_items, hparams: dict, loss_cfg: dict, fps: float) -> tuple[float, float]:
    mode = str(loss_cfg.get("pos_weight_mode", "from_ckpt"))
    fb_beat = float(loss_cfg.get("beat_pos_weight", 15.0))
    fb_db = float(loss_cfg.get("downbeat_pos_weight", 60.0))

    if mode == "from_ckpt":
        pw = hparams.get("pos_weights", None)
        if isinstance(pw, dict) and "beat" in pw and "downbeat" in pw:
            return float(pw["beat"]), float(pw["downbeat"])
        print("[WARN] 'pos_weights' not found in checkpoint hyper_parameters; falling back to fixed weights.")
        return fb_beat, fb_db

    if mode == "auto":
        tot_frames, tot_beats, tot_dbs = 0.0, 0.0, 0.0
        for it in train_items:
            bt, bp = load_beats_tsv(it.beats_path)
            tot_frames += (it.length_samples / float(it.sr)) * fps
            tot_beats += float(len(bt))
            tot_dbs += float((bp == 1).sum())
        pw_b = max(1.0, round(tot_frames / max(tot_beats, 1.0)))
        pw_d = max(1.0, round(tot_frames / max(tot_dbs, 1.0)))
        return float(pw_b), float(pw_d)

    return fb_beat, fb_db


def pick_peaks_minimal(probs: np.ndarray, fps: float, label_shift: int, threshold: float = 0.5, pool_size: int = 7):
    """
    Replicates Beat This!'s minimal local-max peak picker on 1D sigmoid probabilities,
    then shifts frame indices by -label_shift (+3 for Zero) to get physical beat timestamps.
    """
    if probs.size == 0:
        return np.zeros(0, dtype=np.float64)
    t = torch.from_numpy(probs).float().view(1, 1, -1)
    pad = pool_size // 2
    pooled = F.max_pool1d(t, kernel_size=pool_size, stride=1, padding=pad).view(-1).numpy()
    idx = np.where((probs >= threshold) & (probs >= pooled - 1e-7))[0]
    if idx.size == 0:
        return np.zeros(0, dtype=np.float64)
    lead = -int(label_shift)
    return (idx.astype(np.float64) + lead) / float(fps)


def f_measure(ref: np.ndarray, est: np.ndarray, tol: float = 0.07) -> float:
    """Standard 1-to-1 greedy beat F-measure within +/- tol seconds."""
    ref = np.sort(np.asarray(ref, dtype=np.float64))
    est = np.sort(np.asarray(est, dtype=np.float64))
    if ref.size == 0 and est.size == 0:
        return 1.0
    if ref.size == 0 or est.size == 0:
        return 0.0
    i = j = tp = 0
    while i < ref.size and j < est.size:
        diff = est[j] - ref[i]
        if abs(diff) <= tol:
            tp += 1
            i += 1
            j += 1
        elif diff < -tol:
            j += 1
        else:
            i += 1
    return float((2.0 * tp) / (ref.size + est.size))


def filter_times_by_active_mask(times: np.ndarray, active_mask: np.ndarray, fps: float) -> np.ndarray:
    if times.size == 0 or active_mask.size == 0:
        return np.zeros(0, dtype=np.float64)
    idx = np.clip(np.rint(times * fps).astype(np.int64), 0, active_mask.size - 1)
    return times[active_mask[idx] > 0.5]


@torch.no_grad()
def forward_chunked_fixed_len(
    model: nn.Module,
    mel: torch.Tensor,
    mel_silence: torch.Tensor,
    device: torch.device,
    chunk_len: int = 1536,
    overlap: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Run causal inference strictly on fixed (1, chunk_len, 128) tensors so
    torch.compile(flex_attention) compiles only once for validation."""
    T = mel.shape[0]
    stride = chunk_len - overlap
    assert stride > 0, "chunk_len must be greater than overlap"

    p_beat = np.empty(T, dtype=np.float32)
    p_db = np.empty(T, dtype=np.float32)
    start = 0

    while start < T:
        end = min(start + chunk_len, T)
        valid_len = end - start
        chunk = mel[start:end]
        if valid_len < chunk_len:
            pad = mel_silence.unsqueeze(0).expand(chunk_len - valid_len, -1)
            chunk = torch.cat([chunk, pad], dim=0)

        out = model(chunk.unsqueeze(0).to(device))
        b = torch.sigmoid(out["beat"][0, :valid_len]).float().cpu().numpy()
        d = torch.sigmoid(out["downbeat"][0, :valid_len]).float().cpu().numpy()

        if start == 0:
            p_beat[0:valid_len] = b
            p_db[0:valid_len] = d
        else:
            p_beat[start + overlap : end] = b[overlap:valid_len]
            p_db[start + overlap : end] = d[overlap:valid_len]

        if end == T:
            break
        start += stride

    return p_beat, p_db


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_ds: MultiGuitarBeatValDataset,
    device: torch.device,
    label_shift: int,
    eval_cfg: dict,
    chunk_len: int = 1536,
    overlap: int = 512,
):
    model.eval()
    fps = val_ds.fps
    tol = float(eval_cfg.get("tol_sec", 0.07))
    thr = float(eval_cfg.get("peak_threshold", 0.5))
    pool = int(eval_cfg.get("peak_pool_size", 7))
    mel_silence = val_ds.mel_silence

    per_ds = defaultdict(lambda: {"b_act": [], "db_act": [], "b_all": [], "db_all": []})

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda b: b[0])
    for sample in tqdm(val_loader, desc="Validation", ncols=120, leave=False):
        p_beat, p_db = forward_chunked_fixed_len(
            model,
            sample["mel"],
            mel_silence=mel_silence,
            device=device,
            chunk_len=chunk_len,
            overlap=overlap,
        )

        est_b = pick_peaks_minimal(p_beat, fps=fps, label_shift=label_shift, threshold=thr, pool_size=pool)
        est_db = pick_peaks_minimal(p_db, fps=fps, label_shift=label_shift, threshold=thr, pool_size=pool)

        ref_b = sample["beats"]
        ref_db = sample["downbeats"]
        act_mask = sample["active_mask"]
        ds = sample["dataset"]

        per_ds[ds]["b_all"].append(f_measure(ref_b, est_b, tol=tol))
        if ref_db.size > 0:
            per_ds[ds]["db_all"].append(f_measure(ref_db, est_db, tol=tol))

        ref_b_act = filter_times_by_active_mask(ref_b, act_mask, fps)
        est_b_act = filter_times_by_active_mask(est_b, act_mask, fps)
        if ref_b_act.size > 0:
            per_ds[ds]["b_act"].append(f_measure(ref_b_act, est_b_act, tol=tol))

        ref_db_act = filter_times_by_active_mask(ref_db, act_mask, fps)
        est_db_act = filter_times_by_active_mask(est_db, act_mask, fps)
        if ref_db_act.size > 0:
            per_ds[ds]["db_act"].append(f_measure(ref_db_act, est_db_act, tol=tol))

    summary = {}
    macro_b_act, macro_db_act, macro_b_all = [], [], []
    for ds, m in sorted(per_ds.items()):
        b_act = float(np.mean(m["b_act"])) if m["b_act"] else 0.0
        db_act = float(np.mean(m["db_act"])) if m["db_act"] else 0.0
        b_all = float(np.mean(m["b_all"])) if m["b_all"] else 0.0
        db_all = float(np.mean(m["db_all"])) if m["db_all"] else 0.0
        summary[ds] = {
            "beat_f70_active": round(b_act, 4),
            "downbeat_f70_active": round(db_act, 4),
            "beat_f70_all": round(b_all, 4),
            "downbeat_f70_all": round(db_all, 4),
            "n_renders": len(m["b_all"]),
        }
        macro_b_act.append(b_act)
        macro_db_act.append(db_act)
        macro_b_all.append(b_all)

    summary["macro"] = {
        "beat_f70_active": round(float(np.mean(macro_b_act)), 4) if macro_b_act else 0.0,
        "downbeat_f70_active": round(float(np.mean(macro_db_act)), 4) if macro_db_act else 0.0,
        "beat_f70_all": round(float(np.mean(macro_b_all)), 4) if macro_b_all else 0.0,
    }
    return summary


def format_val_summary(tag: str, summary: dict) -> str:
    parts = [
        f"{ds}: B_act={v['beat_f70_active']:.4f} DB_act={v['downbeat_f70_active']:.4f} (B_all={v['beat_f70_all']:.4f})"
        for ds, v in summary.items()
        if ds != "macro"
    ]
    mac = summary["macro"]
    return (
        f"[{tag}] MACRO B_act={mac['beat_f70_active']:.4f} DB_act={mac['downbeat_f70_active']:.4f} "
        f"(B_all={mac['beat_f70_all']:.4f}) | " + " | ".join(parts)
    )


def save_btl_checkpoint(
    model: nn.Module,
    clean_hparams: dict,
    out_path: pathlib.Path,
    extra_meta: dict,
):
    """Save in exact CPJKU/beat_this_live format for direct loading in btl_frontend.py & streaming.py."""
    state_dict = {f"model.{k}": v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "state_dict": state_dict,
        "hyper_parameters": clean_hparams,
        "ft_meta": extra_meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))


def save_train_state(
    out_path: pathlib.Path,
    model: nn.Module,
    ema_obj: Optional[ModelEMA],
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_score: float,
    wandb_run_id: Optional[str] = None,
):
    payload = {
        "model": model.state_dict(),
        "ema": ema_obj.ema.state_dict() if ema_obj is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "wandb_run_id": wandb_run_id,
    }
    torch.save(payload, str(out_path))


def build_cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int, lr: float, min_lr: float):
    min_ratio = float(min_lr) / max(float(lr), 1e-12)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        prog = min(max(prog, 0.0), 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_scheduler_wsd(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    tail_ratio: float = 0.05,
    decay_start_ratio: float = 0.8,
    tail_start_ratio: float = 0.9,
):
    decay_start = int(decay_start_ratio * total_steps)
    tail_start = int(tail_start_ratio * total_steps)

    decay_start = max(warmup_steps, decay_start)
    tail_start = max(decay_start, tail_start)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        elif current_step < decay_start:
            return 1.0
        elif current_step < tail_start:
            decay_length = max(1, tail_start - decay_start)
            progress = (current_step - decay_start) / decay_length
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return tail_ratio + (1.0 - tail_ratio) * cosine_decay
        else:
            return tail_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer, train_cfg: dict, total_steps: int):
    sched_type = str(train_cfg.get("scheduler", "cosine")).lower()
    warmup_steps = int(train_cfg.get("warmup_steps", 200))
    lr = float(train_cfg["lr"])
    min_lr = float(train_cfg.get("min_lr", 1e-6))

    if sched_type == "wsd":
        tail_ratio = train_cfg.get("tail_ratio", None)
        tail_ratio = (min_lr / max(lr, 1e-12)) if tail_ratio is None else float(tail_ratio)
        return get_scheduler_wsd(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            tail_ratio=tail_ratio,
            decay_start_ratio=float(train_cfg.get("decay_start_ratio", 0.8)),
            tail_start_ratio=float(train_cfg.get("tail_start_ratio", 0.9)),
        )
    elif sched_type == "cosine":
        return build_cosine_warmup_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            lr=lr,
            min_lr=min_lr,
        )
    raise ValueError(f"Unknown train.scheduler '{sched_type}'; expected 'wsd' or 'cosine'.")


def set_stem_bn_eval(model: nn.Module):
    for _, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.eval()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    seed_everything(int(cfg.get("seed", 42)))

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(out_dir / "config.yaml"))

    gpu = cfg.get("gpu", 0)
    device = torch.device(f"cuda:{gpu}" if (gpu is not None and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. Load pretrained beat_this_live_zero.ckpt (Windows-safe)
    ckpt_path = str(cfg.model.pretrained_ckpt)
    print(f"[INFO] Loading pretrained Beat This! Live checkpoint: {ckpt_path}")
    with _posix_path_compat():
        model, raw_hparams = load_model(ckpt_path, device=str(device), return_hparams=True)
    clean_hparams = sanitize_hparams(dict(raw_hparams))

    label_shift = int(clean_hparams["label_shift"])
    sw_window = int(clean_hparams.get("sw_attention_window_size", 256))
    fps = float(clean_hparams.get("fps", 50.0))
    print(f"[INFO] Pretrained spec: label_shift={label_shift}, sw_window={sw_window}, fps={fps}")

    if bool(cfg.model.get("freeze_stem", False)) and hasattr(model, "frontend"):
        for p in model.frontend.parameters():
            p.requires_grad_(False)
        print("[INFO] Froze model.frontend stem parameters.")

    # 2. Build datasets & loaders
    datasets_cfg = OmegaConf.to_container(cfg.data.datasets, resolve=True)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    silence_cfg = OmegaConf.to_container(cfg.silence_mask, resolve=True)
    augment_cfg = OmegaConf.to_container(cfg.augment, resolve=True)
    eval_cfg = OmegaConf.to_container(cfg.eval, resolve=True)

    train_items = discover_dataset_items(datasets_cfg, split="train")
    val_items = discover_dataset_items(datasets_cfg, split="validation")

    train_ds = MultiGuitarBeatTrainDataset(
        items=train_items,
        label_shift=label_shift,
        data_cfg=data_cfg,
        silence_cfg=silence_cfg,
        augment_cfg=augment_cfg,
    )
    val_ds = MultiGuitarBeatValDataset(
        items=val_items,
        data_cfg=data_cfg,
        eval_cfg=eval_cfg,
        max_renders_per_track=cfg.data.get("val_max_renders_per_track", 2),
    )

    sampler = train_ds.build_weighted_sampler(num_samples=int(cfg.data.samples_per_epoch))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        sampler=sampler,
        num_workers=int(cfg.data.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=(int(cfg.data.num_workers) > 0),
    )

    # 3. Losses (ShiftTolerantBCELoss matching CPJKU/beat_this_live)
    loss_cfg = OmegaConf.to_container(cfg.train.loss, resolve=True)
    pw_beat, pw_db = resolve_pos_weights(train_items, clean_hparams, loss_cfg, fps=fps)
    tol = int(loss_cfg.get("tolerance", 3))
    db_weight = float(loss_cfg.get("downbeat_weight", 1.0))
    beat_loss_fn = ShiftTolerantBCELoss(pos_weight=pw_beat, tolerance=tol).to(device)
    db_loss_fn = ShiftTolerantBCELoss(pos_weight=pw_db, tolerance=tol).to(device)
    print(f"[INFO] ShiftTolerantBCELoss: tolerance={tol}, pos_weights=(beat={pw_beat:.2f}, downbeat={pw_db:.2f})")

    # 4. Optimizer, Scheduler, EMA
    epochs = int(cfg.train.epochs)
    grad_accum = max(1, int(cfg.train.get("grad_accum", 1)))
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = epochs * steps_per_epoch

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    scheduler = build_scheduler(optimizer, train_cfg=train_cfg, total_steps=total_steps)

    ema_decay = float(cfg.train.get("ema_decay", 0.0))
    ema_obj = ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None

    start_epoch = 1
    global_step = 0
    best_score = -1.0
    resume_wandb_id = None
    history_path = out_dir / "history.json"
    history = []

    if args.resume is not None:
        print(f"[INFO] Resuming training state from {args.resume}")
        st = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        if ema_obj is not None and st.get("ema") is not None:
            ema_obj.ema.load_state_dict(st["ema"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        start_epoch = int(st["epoch"]) + 1
        global_step = int(st["global_step"])
        best_score = float(st["best_score"])
        resume_wandb_id = st.get("wandb_run_id", None)
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"[INFO] Resumed at epoch {start_epoch}, step {global_step}, best_score={best_score:.4f}")

    # 5. Initialize Weights & Biases (reuses run ID when resuming)
    wb_run = init_wandb_run(cfg, args, out_dir=out_dir, resume_run_id=resume_wandb_id)
    wb_run_id = wb_run.id if wb_run is not None else resume_wandb_id
    log_every_steps = max(1, int(cfg.get("wandb", {}).get("log_every_steps", 20)))

    try:
        # 6. Baseline validation (Epoch 0 on fresh runs or --eval_only)
        eval_model = ema_obj.ema if ema_obj is not None else model
        if args.resume is None or args.eval_only:
            base_val = evaluate(eval_model, val_ds, device, label_shift=label_shift, eval_cfg=eval_cfg)
            print(format_val_summary("VAL epoch=00 (pretrained)", base_val))
            if wb_run is not None:
                wb_run.log(flatten_val_metrics_for_wandb(base_val, epoch=0), step=global_step)
            if args.eval_only:
                return
            history.append({"epoch": 0, "global_step": global_step, "val": base_val})
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        use_amp = bool(cfg.train.get("use_amp", False)) and (device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
        freeze_bn = bool(cfg.model.get("freeze_stem_bn", False))
        grad_clip = float(cfg.train.get("grad_clip", 1.0))
        eval_every = int(cfg.train.get("eval_every", 1))
        save_every = int(cfg.train.get("save_every", 5))

        # 7. Training loop
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            model.train()
            if freeze_bn:
                set_stem_bn_eval(model)

            optimizer.zero_grad(set_to_none=True)
            ep_loss = ep_b_loss = ep_d_loss = 0.0
            ep_batches = 0

            # Rolling accumulators for step-level W&B logs
            win_loss = win_b_loss = win_d_loss = win_mask_frac = 0.0
            win_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d}", ncols=130)
            for step_in_ep, batch in enumerate(pbar, start=1):
                mel = batch["mel"].to(device, non_blocking=True)
                b_tgt = batch["beat_target"].to(device, non_blocking=True)
                d_tgt = batch["downbeat_target"].to(device, non_blocking=True)
                b_mask = batch["beat_mask"].to(device, non_blocking=True)
                d_mask = batch["downbeat_mask"].to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(mel)
                    loss_b = beat_loss_fn(out["beat"], b_tgt, b_mask)
                    if d_mask.sum() > 0:
                        loss_d = db_loss_fn(out["downbeat"], d_tgt, d_mask)
                    else:
                        loss_d = torch.zeros_like(loss_b)
                    loss = loss_b + db_weight * loss_d
                    loss_scaled = loss / float(grad_accum)

                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                lv = float(loss.item())
                lb = float(loss_b.item())
                ld = float(loss_d.item())
                mf = float(b_mask.mean().item())

                ep_loss += lv
                ep_b_loss += lb
                ep_d_loss += ld
                ep_batches += 1

                win_loss += lv
                win_b_loss += lb
                win_d_loss += ld
                win_mask_frac += mf
                win_batches += 1

                if step_in_ep % grad_accum == 0 or step_in_ep == len(train_loader):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)

                    max_norm = grad_clip if grad_clip > 0 else float("inf")
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm).item())

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if ema_obj is not None:
                        ema_obj.update(model)

                    lr_now = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(
                        loss=f"{ep_loss / ep_batches:.4f}",
                        b=f"{ep_b_loss / ep_batches:.4f}",
                        db=f"{ep_d_loss / ep_batches:.4f}",
                        lr=f"{lr_now:.2e}",
                    )

                    if wb_run is not None and (global_step % log_every_steps == 0):
                        wb_run.log(
                            {
                                "epoch": epoch,
                                "train/loss": win_loss / max(win_batches, 1),
                                "train/beat_loss": win_b_loss / max(win_batches, 1),
                                "train/downbeat_loss": win_d_loss / max(win_batches, 1),
                                "train/active_mask_frac": win_mask_frac / max(win_batches, 1),
                                "train/grad_norm": grad_norm,
                                "train/lr": lr_now,
                            },
                            step=global_step,
                        )
                        win_loss = win_b_loss = win_d_loss = win_mask_frac = 0.0
                        win_batches = 0

            ep_sec = round(time.time() - t0, 1)
            ep_rec = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": round(ep_loss / max(ep_batches, 1), 5),
                "train_beat_loss": round(ep_b_loss / max(ep_batches, 1), 5),
                "train_downbeat_loss": round(ep_d_loss / max(ep_batches, 1), 5),
                "lr": optimizer.param_groups[0]["lr"],
                "sec": ep_sec,
            }

            wb_epoch_logs = {
                "epoch": epoch,
                "train/epoch_loss": ep_rec["train_loss"],
                "train/epoch_beat_loss": ep_rec["train_beat_loss"],
                "train/epoch_downbeat_loss": ep_rec["train_downbeat_loss"],
                "train/epoch_sec": ep_sec,
                "train/lr": ep_rec["lr"],
            }

            export_model = ema_obj.ema if ema_obj is not None else model

            if epoch % eval_every == 0 or epoch == epochs:
                val_summary = evaluate(export_model, val_ds, device, label_shift=label_shift, eval_cfg=eval_cfg)
                ep_rec["val"] = val_summary
                print(format_val_summary(f"VAL epoch={epoch:02d}", val_summary))

                score = float(val_summary["macro"]["beat_f70_active"])
                if score > best_score:
                    best_score = score
                    save_btl_checkpoint(
                        export_model,
                        clean_hparams,
                        out_dir / "best.ckpt",
                        extra_meta={"epoch": epoch, "global_step": global_step, "val": val_summary},
                    )
                    print(f"[INFO] New best checkpoint saved -> {out_dir / 'best.ckpt'} (macro B_act={best_score:.4f})")
                    if wb_run is not None:
                        wb_run.summary["best_epoch"] = epoch
                        wb_run.summary["best_macro_beat_f70_active"] = best_score
                        for ds, m in val_summary.items():
                            if ds != "macro":
                                wb_run.summary[f"best_{ds}_beat_f70_active"] = m["beat_f70_active"]
                                wb_run.summary[f"best_{ds}_downbeat_f70_active"] = m["downbeat_f70_active"]

                wb_epoch_logs.update(
                    flatten_val_metrics_for_wandb(val_summary, epoch=epoch, best_score=best_score)
                )

            if wb_run is not None:
                wb_run.log(wb_epoch_logs, step=global_step)

            save_btl_checkpoint(
                export_model,
                clean_hparams,
                out_dir / "last.ckpt",
                extra_meta={"epoch": epoch, "global_step": global_step},
            )
            save_train_state(
                out_dir / "train_state.pt",
                model, ema_obj, optimizer, scheduler, epoch, global_step, best_score,
                wandb_run_id=wb_run_id,
            )
            if save_every > 0 and epoch % save_every == 0:
                save_btl_checkpoint(
                    export_model,
                    clean_hparams,
                    out_dir / f"epoch_{epoch:03d}.ckpt",
                    extra_meta={"epoch": epoch, "global_step": global_step},
                )

            history.append(ep_rec)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(f"[INFO] Fine-tuning finished. Best macro active beat F70 = {best_score:.4f} ({out_dir / 'best.ckpt'})")
    finally:
        if wb_run is not None:
            wb_run.finish()


if __name__ == "__main__":
    main()