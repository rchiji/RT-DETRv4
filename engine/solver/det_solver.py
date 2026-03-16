"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import datetime
import json
import math
import random
import time

import numpy as np
import torch

from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from async_reports import AsyncReportDispatcher
from .det_engine import evaluate, train_one_epoch


def _metric_to_scalar(metric) -> float:
    if isinstance(metric, (list, tuple)):
        if len(metric) == 0:
            return 0.0
        return float(metric[0])
    if isinstance(metric, torch.Tensor):
        if metric.numel() == 0:
            return 0.0
        return float(metric.reshape(-1)[0].item())
    return float(metric)


def _disable_writer(writer, reason: str) -> None:
    """Best-effort disable for TensorBoard writer failures."""
    if writer is None:
        return
    try:
        setattr(writer, "_rtv4_disabled", True)
    except Exception:
        pass
    # Keep the object alive but disabled; calling close() on a broken TB writer
    # can trigger secondary worker-thread exceptions.
    print(f"[warn] TensorBoard writer disabled in DetSolver: {reason}")


def _append_json_line_with_retry(path, payload, retries: int = 30, wait_seconds: float = 0.2) -> None:
    """Append one JSON line with retry; never raises to avoid aborting training on transient file locks."""
    line = json.dumps(payload) + "\n"
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                # Best effort durability. On some platforms/filesystems this can still fail transiently.
                try:
                    import os
                    os.fsync(f.fileno())
                except OSError:
                    pass
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(wait_seconds)
                continue
            break
        except Exception as exc:
            # Unexpected errors are also swallowed so training can continue.
            last_exc = exc
            break

    # Final fallback: write into a sidecar file and continue training.
    fallback = path.with_name(f"{path.stem}.fallback.log")
    try:
        with fallback.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    print(f"[warn] failed to append {path}; continued without crashing. error={repr(last_exc)}")


class _EarlyStopping:
    """Inline early stopping tracker."""

    def __init__(self, cfg_dict: dict):
        self.enabled = bool(cfg_dict.get("enabled", False))
        self.metric = str(cfg_dict.get("metric", "eval_bbox_ap"))
        self.mode = str(cfg_dict.get("mode", "max"))
        self.patience = int(cfg_dict.get("patience", 15))
        self.min_delta = float(cfg_dict.get("min_delta", 0.001))
        self.min_epochs = int(cfg_dict.get("min_epochs", 0))
        self.best_value: float = float("-inf") if self.mode == "max" else float("inf")
        self.best_epoch: int = -1
        self.no_improve_count: int = 0

    def step(self, epoch: int, metric_value: float) -> bool:
        """Return True if training should stop."""
        if not self.enabled:
            return False
        improved = False
        if self.mode == "max":
            improved = metric_value > self.best_value + self.min_delta
        else:
            improved = metric_value < self.best_value - self.min_delta
        if improved:
            self.best_value = metric_value
            self.best_epoch = epoch
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        if epoch < self.min_epochs:
            return False
        return self.no_improve_count >= self.patience

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "no_improve_count": self.no_improve_count,
        }


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg
        checkpoint_dir = None
        if self.output_dir:
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def _checkpoint_target(name: str):
            if checkpoint_dir is not None:
                return checkpoint_dir / name
            return name

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        # Early stopping setup
        early_stop_cfg = {}
        if isinstance(getattr(args, "yaml_cfg", None), dict):
            early_stop_cfg = args.yaml_cfg.get("early_stop", {}) or {}
        early_stopper = _EarlyStopping(early_stop_cfg)
        if early_stopper.enabled:
            print(f"[early-stop] enabled: metric={early_stopper.metric}, mode={early_stopper.mode}, "
                  f"patience={early_stopper.patience}, min_epochs={early_stopper.min_epochs}")

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                metric_scalar = _metric_to_scalar(test_stats[k])
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = metric_scalar
                top1 = metric_scalar
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        async_report_dispatcher = None
        if self.output_dir and dist_utils.is_main_process():
            async_report_cfg = {}
            if isinstance(getattr(self.cfg, "yaml_cfg", None), dict):
                yaml_cfg = self.cfg.yaml_cfg
                async_report_cfg = yaml_cfg.get("async_reports") or yaml_cfg.get("async_reports") or {}
            async_report_dispatcher = AsyncReportDispatcher.from_mapping(
                async_report_cfg,
                output_dir=self.output_dir,
            )
            async_report_dispatcher.start()

        try:
            for epoch in range(start_epoch, args.epoches):

                self.train_dataloader.set_epoch(epoch)
                # self.train_dataloader.dataset.set_epoch(epoch)
                if dist_utils.is_dist_available_and_initialized():
                    self.train_dataloader.sampler.set_epoch(epoch)

                # Epoch-based seeding for reproducibility (non-distributed)
                base_seed = int(args.seed) if args.seed is not None else 0
                epoch_seed = base_seed + epoch
                random.seed(epoch_seed)
                np.random.seed(epoch_seed % (2**31))
                torch.manual_seed(epoch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(epoch_seed)
                # Re-seed the dataloader's RandomSampler for shuffle reproducibility
                sampler = getattr(self.train_dataloader, 'sampler', None)
                if sampler is not None and hasattr(sampler, 'generator'):
                    if sampler.generator is None:
                        sampler.generator = torch.Generator()
                    sampler.generator.manual_seed(epoch_seed)

                if epoch == self.train_dataloader.collate_fn.stop_epoch:
                    self.load_resume_state(str(_checkpoint_target('best_stg1.pth')))
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

                train_stats, grad_percentages = train_one_epoch(
                    self.self_lr_scheduler,
                    self.lr_scheduler,
                    self.model,
                    self.criterion,
                    self.train_dataloader,
                    self.optimizer,
                    self.device,
                    epoch,
                    max_norm=args.clip_max_norm,
                    print_freq=args.print_freq,
                    ema=self.ema,
                    scaler=self.scaler,
                    lr_warmup_scheduler=self.lr_warmup_scheduler,
                    writer=self.writer,
                    teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
                )

                if not self.self_lr_scheduler:  # update by epoch
                    if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                        self.lr_scheduler.step()

                self.last_epoch += 1
                if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                    self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                    params = self.criterion.distill_adaptive_params
                    default_weight = params.get('default_weight')

                    avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                    current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                    new_weight = current_weight
                    reason = 'unchanged'

                    if avg_percentage < 1e-6:
                        if default_weight is not None:
                            new_weight = default_weight
                            reason = 'reset_to_default_zero_grad'
                    elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if default_weight is not None:
                            new_weight = default_weight
                            reason = 'ema_phase_default'
                    else:
                        rho = params['rho']
                        delta = params['delta']
                        lower_bound = rho - delta
                        upper_bound = rho + delta
                        if not (lower_bound <= avg_percentage <= upper_bound):
                            target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                            if current_weight > 1e-6:
                                p_current = avg_percentage / 100.0
                                p_target = target_percentage / 100.0
                                numerator = p_target * (1.0 - p_current)
                                denominator = p_current * (1.0 - p_target)
                                if abs(denominator) >= 1e-9:
                                    ratio = numerator / denominator
                                    ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                    new_weight = current_weight * ratio
                                    new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                    reason = f'adjusted_to_{target_percentage:.2f}%'

                    if abs(new_weight - current_weight) > 0:
                        self.criterion.weight_dict['loss_distill'] = new_weight
                    print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

                overlay_checkpoint_path = None
                overlay_checkpoint_is_temporary = False
                if self.output_dir:
                    checkpoint_paths = [_checkpoint_target('last.pth')]
                    # Save per-epoch checkpoint only when checkpoint_freq matches
                    if (epoch + 1) % args.checkpoint_freq == 0:
                        checkpoint_paths.append(_checkpoint_target(f'checkpoint{epoch:04}.pth'))

                    needs_overlay_ckpt = (
                        async_report_dispatcher is not None
                        and async_report_dispatcher.requires_epoch_checkpoint(epoch)
                    )
                    if needs_overlay_ckpt:
                        overlay_checkpoint_path = _checkpoint_target(f'checkpoint{epoch:04}.pth')
                        if overlay_checkpoint_path not in checkpoint_paths:
                            checkpoint_paths.append(overlay_checkpoint_path)
                            overlay_checkpoint_is_temporary = True

                    for checkpoint_path in checkpoint_paths:
                        dist_utils.save_on_master(self.state_dict(), checkpoint_path)

                    if overlay_checkpoint_path is None:
                        overlay_checkpoint_path = _checkpoint_target('last.pth')

                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device
                )

                # TODO
                for k in test_stats:
                    metric_value = test_stats[k]
                    metric_scalar = _metric_to_scalar(metric_value)
                    if self.writer and dist_utils.is_main_process() and not getattr(self.writer, "_rtv4_disabled", False):
                        try:
                            if isinstance(metric_value, (list, tuple)):
                                for i, v in enumerate(metric_value):
                                    self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
                            else:
                                self.writer.add_scalar(f'Test/{k}'.format(k), metric_scalar, epoch)
                        except Exception as exc:
                            _disable_writer(self.writer, repr(exc))
                            self.writer = None

                    if k in best_stat:
                        best_stat['epoch'] = epoch if metric_scalar > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], metric_scalar)
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = metric_scalar

                    if best_stat[k] > top1:
                        best_stat_print['epoch'] = epoch
                        top1 = best_stat[k]
                        if self.output_dir:
                            if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                                dist_utils.save_on_master(self.state_dict(), _checkpoint_target('best_stg2.pth'))
                            else:
                                dist_utils.save_on_master(self.state_dict(), _checkpoint_target('best_stg1.pth'))

                    best_stat_print[k] = max(best_stat[k], top1)
                    print(f'best_stat: {best_stat_print}')  # global best

                    if best_stat['epoch'] == epoch and self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            if metric_scalar > top1:
                                top1 = metric_scalar
                                dist_utils.save_on_master(self.state_dict(), _checkpoint_target('best_stg2.pth'))
                        else:
                            top1 = max(metric_scalar, top1)
                            dist_utils.save_on_master(self.state_dict(), _checkpoint_target('best_stg1.pth'))

                    elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        best_stat = {'epoch': -1, }
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(_checkpoint_target('best_stg1.pth')))
                        print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }

                if self.output_dir and dist_utils.is_main_process():
                    _append_json_line_with_retry(self.output_dir / "log.txt", log_stats)

                    # for evaluation logs
                    if coco_evaluator is not None:
                        (self.output_dir / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                dist_utils.save_on_master(
                                    coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name
                                )

                    if async_report_dispatcher is not None:
                        async_report_dispatcher.submit_epoch(
                            epoch=epoch,
                            checkpoint_path=overlay_checkpoint_path,
                            cleanup_checkpoint=overlay_checkpoint_is_temporary,
                        )

                # Early stopping check
                if early_stopper.enabled and dist_utils.is_main_process():
                    es_metric_key = f"test_{early_stopper.metric}"
                    es_value = None
                    if es_metric_key in log_stats:
                        raw = log_stats[es_metric_key]
                        es_value = _metric_to_scalar(raw) if not isinstance(raw, (int, float)) else float(raw)
                    if es_value is not None:
                        should_stop = early_stopper.step(epoch, es_value)
                        print(f"[early-stop] epoch={epoch}, {early_stopper.metric}={es_value:.6f}, "
                              f"best={early_stopper.best_value:.6f}@{early_stopper.best_epoch}, "
                              f"no_improve={early_stopper.no_improve_count}/{early_stopper.patience}")
                        if should_stop:
                            es_info = {
                                "triggered_at": datetime.datetime.now().isoformat(),
                                "stopped_epoch": epoch,
                                "stopped_metric_value": es_value,
                                **early_stopper.to_dict(),
                            }
                            es_path = self.output_dir / "early_stop.json"
                            with open(es_path, "w", encoding="utf-8") as f:
                                json.dump(es_info, f, indent=2, ensure_ascii=False)
                            print(f"[early-stop] TRIGGERED at epoch {epoch}. "
                                  f"Best was {early_stopper.best_value:.6f} at epoch {early_stopper.best_epoch}. "
                                  f"Saved: {es_path}")
                            break
        finally:
            if async_report_dispatcher is not None:
                async_report_dispatcher.close()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return

    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state
