"""
Async output worker for RT-DETRv4 training.

This module keeps expensive output updates (history plot export, overlay image
generation) off the main training loop by dispatching jobs to a multiprocessing
worker through a queue.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AsyncReportSettings:
    enabled: bool = False
    queue_size: int = 1024

    repo_src: str = ""
    output_dir: str = ""
    stdout_log: str = ""
    official_root: str = ""
    official_config: str = ""

    train_image_root: str = ""
    train_annotations: str = ""
    val_image_root: str = ""
    val_annotations: str = ""

    overlay_check: bool = False
    overlay_every: int = 1
    overlay_max_train_images: int = 0
    overlay_max_val_images: int = 0
    overlay_score_thr: float = 0.3
    overlay_top_k: int = 0
    overlay_seed: int = 0
    overlay_device: str = "cpu"
    overlay_ref_train_annotations: Optional[str] = None
    overlay_ref_val_annotations: Optional[str] = None

    @classmethod
    def from_mapping(
        cls,
        cfg: Dict[str, Any] | None,
        *,
        output_dir: Optional[Path] = None,
    ) -> "AsyncReportSettings":
        data = dict(cfg or {})
        out = cls()

        out.enabled = bool(data.get("enabled", False))
        out.queue_size = max(1, int(data.get("queue_size", out.queue_size)))

        out.repo_src = str(data.get("repo_src", out.repo_src) or "")
        out.output_dir = str(data.get("output_dir", out.output_dir) or "")
        out.stdout_log = str(data.get("stdout_log", out.stdout_log) or "")
        out.official_root = str(data.get("official_root", out.official_root) or "")
        out.official_config = str(data.get("official_config", out.official_config) or "")

        out.train_image_root = str(data.get("train_image_root", out.train_image_root) or "")
        out.train_annotations = str(data.get("train_annotations", out.train_annotations) or "")
        out.val_image_root = str(data.get("val_image_root", out.val_image_root) or "")
        out.val_annotations = str(data.get("val_annotations", out.val_annotations) or "")

        out.overlay_check = bool(data.get("overlay_check", out.overlay_check))
        out.overlay_every = max(1, int(data.get("overlay_every", out.overlay_every)))
        out.overlay_max_train_images = max(0, int(data.get("overlay_max_train_images", out.overlay_max_train_images)))
        out.overlay_max_val_images = max(0, int(data.get("overlay_max_val_images", out.overlay_max_val_images)))
        out.overlay_score_thr = float(data.get("overlay_score_thr", out.overlay_score_thr))
        out.overlay_top_k = max(0, int(data.get("overlay_top_k", out.overlay_top_k)))
        out.overlay_seed = int(data.get("overlay_seed", out.overlay_seed))
        out.overlay_device = str(data.get("overlay_device", out.overlay_device) or "cpu")

        ref_train = data.get("overlay_ref_train_annotations", None)
        ref_val = data.get("overlay_ref_val_annotations", None)
        out.overlay_ref_train_annotations = str(ref_train) if ref_train else None
        out.overlay_ref_val_annotations = str(ref_val) if ref_val else None

        if output_dir is not None and not out.output_dir:
            out.output_dir = str(output_dir.resolve())
        if out.output_dir and not out.stdout_log:
            out.stdout_log = str((Path(out.output_dir) / "train_stdout.log").resolve())

        return out


def _resolve_repo_src(settings: AsyncReportSettings) -> Path:
    if settings.repo_src:
        return Path(settings.repo_src).resolve()
    # .../src/RT-DERTv4/engine/solver/async_reports.py -> .../src
    return Path(__file__).resolve().parents[4]


def _run_one_epoch_overlay(
    *,
    settings: AsyncReportSettings,
    output_dir: Path,
    epoch: int,
    checkpoint_path: Optional[Path],
) -> None:
    if not settings.overlay_check:
        return
    if settings.overlay_max_train_images <= 0 and settings.overlay_max_val_images <= 0:
        return
    if (int(epoch) % int(settings.overlay_every)) != 0:
        return

    from official_overlay import find_checkpoint_for_epoch, run_overlay_for_checkpoint

    root_dirname = (Path("epoch_overlay") / f"epoch_{int(epoch):04d}").as_posix()
    sparse_summary = output_dir / root_dirname / "summary.json"
    full_summary = output_dir / root_dirname / "summary_full_gt.json"

    run_sparse = not sparse_summary.exists()
    run_full = bool(
        settings.overlay_ref_train_annotations and settings.overlay_ref_val_annotations and (not full_summary.exists())
    )
    if not run_sparse and not run_full:
        return

    resolved_ckpt = checkpoint_path
    if resolved_ckpt is None or (not resolved_ckpt.exists()):
        resolved_ckpt = find_checkpoint_for_epoch(output_dir=output_dir, epoch=int(epoch))
    if resolved_ckpt is None:
        print(
            f"[async-output] overlay skipped: checkpoint missing (epoch={epoch})",
            flush=True,
        )
        return

    base_kwargs: Dict[str, Any] = {
        "official_root": Path(settings.official_root).resolve(),
        "config_path": Path(settings.official_config).resolve(),
        "checkpoint_path": resolved_ckpt.resolve(),
        "output_dir": output_dir.resolve(),
        "root_dirname": root_dirname,
        "train_image_root": Path(settings.train_image_root).resolve(),
        "train_annotations": Path(settings.train_annotations).resolve(),
        "val_image_root": Path(settings.val_image_root).resolve(),
        "val_annotations": Path(settings.val_annotations).resolve(),
        "max_train_images": int(settings.overlay_max_train_images),
        "max_val_images": int(settings.overlay_max_val_images),
        "score_thr": float(settings.overlay_score_thr),
        "top_k": int(settings.overlay_top_k),
        "seed": int(settings.overlay_seed) + int(epoch) * 17,
        "device_mode": str(settings.overlay_device),
    }

    if run_sparse:
        summary_path = run_overlay_for_checkpoint(
            **base_kwargs,
            train_overlay_annotations=None,
            val_overlay_annotations=None,
            name_suffix="",
            extra_summary={
                "epoch": int(epoch),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "epoch_overlay_check",
                "generated_by": "async_worker",
            },
        )
        print(
            f"[async-output] epoch overlay saved: epoch={epoch} -> {summary_path}",
            flush=True,
        )

    if run_full:
        summary_path = run_overlay_for_checkpoint(
            **base_kwargs,
            train_overlay_annotations=Path(settings.overlay_ref_train_annotations).resolve(),
            val_overlay_annotations=Path(settings.overlay_ref_val_annotations).resolve(),
            name_suffix="_full_gt",
            extra_summary={
                "epoch": int(epoch),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "epoch_overlay_check_full_gt",
                "generated_by": "async_worker",
            },
        )
        print(
            f"[async-output] epoch full-gt overlay saved: epoch={epoch} -> {summary_path}",
            flush=True,
        )


def _worker_entry(job_queue: mp.Queue, raw_settings: Dict[str, Any]) -> None:
    settings = AsyncReportSettings.from_mapping(raw_settings)
    if not settings.enabled:
        return

    repo_src = _resolve_repo_src(settings)
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from train_viz import update_official_history_from_log

    output_dir = Path(settings.output_dir).resolve()
    stdout_log = Path(settings.stdout_log).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    last_count = -1
    while True:
        job = job_queue.get()
        if not isinstance(job, dict):
            continue
        if job.get("kind") == "stop":
            break
        if job.get("kind") != "epoch_done":
            continue

        epoch = int(job.get("epoch", -1))
        checkpoint_text = job.get("checkpoint_path", None)
        checkpoint_path = Path(str(checkpoint_text)).resolve() if checkpoint_text else None

        try:
            next_count, history_outputs = update_official_history_from_log(
                stdout_log_path=stdout_log,
                output_dir=output_dir,
                previous_count=last_count,
            )
            last_count = int(next_count)
            if history_outputs is not None:
                json_path, csv_path, png_path, html_path = history_outputs
                print(
                    "[async-output] history updated: " f"{json_path} | {csv_path} | {png_path} | {html_path}",
                    flush=True,
                )
        except Exception as exc:  # pragma: no cover - safety for long runs
            print(
                f"[async-output] history update failed at epoch={epoch}: {repr(exc)}",
                flush=True,
            )
            traceback.print_exc()

        try:
            _run_one_epoch_overlay(
                settings=settings,
                output_dir=output_dir,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:  # pragma: no cover - safety for long runs
            print(
                f"[async-output] overlay failed at epoch={epoch}: {repr(exc)}",
                flush=True,
            )
            traceback.print_exc()

    # Final history flush.
    try:
        update_official_history_from_log(
            stdout_log_path=stdout_log,
            output_dir=output_dir,
            previous_count=last_count,
        )
    except Exception:
        pass


class AsyncReportDispatcher:
    """Main-process wrapper for async output worker."""

    def __init__(self, settings: AsyncReportSettings) -> None:
        self.settings = settings
        self._ctx = mp.get_context("spawn")
        self._queue: Optional[mp.Queue] = None
        self._proc: Optional[mp.Process] = None
        self._started = False

    @classmethod
    def from_mapping(
        cls,
        cfg: Dict[str, Any] | None,
        *,
        output_dir: Optional[Path] = None,
    ) -> "AsyncReportDispatcher":
        settings = AsyncReportSettings.from_mapping(cfg, output_dir=output_dir)
        return cls(settings)

    def start(self) -> None:
        if self._started or not self.settings.enabled:
            return
        self._queue = self._ctx.Queue(maxsize=int(self.settings.queue_size))
        self._proc = self._ctx.Process(
            target=_worker_entry,
            args=(self._queue, asdict(self.settings)),
            daemon=True,
            name="rtv4_async_reports",
        )
        self._proc.start()
        self._started = True
        print(
            f"[async-output] worker started (pid={self._proc.pid}, queue={self.settings.queue_size})",
            flush=True,
        )

    def requires_epoch_checkpoint(self, epoch: int) -> bool:
        if not self.settings.enabled:
            return False
        if not self.settings.overlay_check:
            return False
        if self.settings.overlay_max_train_images <= 0 and self.settings.overlay_max_val_images <= 0:
            return False
        return (int(epoch) % int(self.settings.overlay_every)) == 0

    def submit_epoch(self, *, epoch: int, checkpoint_path: Optional[Path]) -> None:
        if not self._started or self._queue is None:
            return
        payload = {
            "kind": "epoch_done",
            "epoch": int(epoch),
            "checkpoint_path": (str(checkpoint_path.resolve()) if checkpoint_path is not None else None),
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            print(
                f"[async-output] queue full, dropped epoch job: epoch={epoch}",
                flush=True,
            )
        except Exception as exc:
            print(f"[async-output] enqueue failed: {repr(exc)}", flush=True)

    def close(self, timeout_seconds: float = 300.0) -> None:
        if not self._started:
            return
        if self._queue is not None:
            try:
                self._queue.put({"kind": "stop"}, timeout=2.0)
            except Exception:
                pass

        if self._proc is not None:
            self._proc.join(timeout=max(0.0, float(timeout_seconds)))
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=10.0)
            print("[async-output] worker stopped", flush=True)

        self._started = False
        self._proc = None
        if self._queue is not None:
            try:
                self._queue.close()
            except Exception:
                pass
        self._queue = None


# Backward-compatible aliases
AsyncReportSettings = AsyncReportSettings
AsyncReportDispatcher = AsyncReportDispatcher
