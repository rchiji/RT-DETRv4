"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import copy
import logging
from typing import Any, ClassVar, Mapping, Sequence

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance

_logger = logging.getLogger(__name__)

Tensor = torch.Tensor
IndexPair = tuple[Tensor, Tensor]
MatchIndices = list[IndexPair]
TargetDict = dict[str, Tensor]
LossDict = dict[str, Tensor]
OutputsDict = dict[str, Any]
# 以降の型エイリアスは、shapeを読む際のノイズを減らすために定義している。


@register()
class RTv4Criterion(nn.Module):
    """Loss computation module for RT-DETRv4.

    Shape legend:
    - `B`: batch size
    - `Q`: number of queries (`num_queries`)
    - `C`: number of classes (`num_classes`)
    - `D`: feature/channel dimension
    - `H`, `W`: spatial height/width
    - `N`: number of matched pairs in the current loss call
    - `Ni`: number of GT boxes in image `i`
    - `Mi`: number of ignore boxes in image `i`
    - `num_bins`: `reg_max + 1` (distribution bins per bbox side)

    Expected main model outputs:
    - `pred_logits`: (B, Q, C) classification logits.
    - `pred_boxes`: (B, Q, 4) boxes in normalized `cxcywh`.

    Optional outputs for local/distribution losses:
    - `pred_corners`: (B, Q, 4 * (reg_max + 1)).
    - `ref_points`: (B, Q, 4), decoder reference points/anchors.
    - `teacher_corners`: same shape as `pred_corners`.
    - `teacher_logits`: (B, Q, C), used for DDF weighting.

    Optional outputs for distillation:
    - `student_distill_output`: (B, D, Hs, Ws).
    - `teacher_encoder_output`: (B, D, Ht, Wt).

    `targets` is a list of dicts per image with at least:
    - `labels`: (Ni,), class ids.
    - `boxes`: (Ni, 4), normalized `cxcywh`.
    - optional `ignore_boxes`: (Mi, 4), normalized `cxcywh`.
    """
    __share__: ClassVar[list[str]] = ["num_classes"]
    __inject__: ClassVar[list[str]] = ["matcher"]

    def __init__(
        self,
        matcher: Any,
        weight_dict: Mapping[str, float],
        losses: Sequence[str],
        alpha: float = 0.2,
        gamma: float = 2.0,
        num_classes: int = 80,
        reg_max: int = 32,
        boxes_weight_format: str | None = None,
        share_matched_indices: bool = False,
        mal_alpha: float | None = None,
        use_uni_set: bool = True,
        distill_adaptive_params: Mapping[str, Any] | None = None,
        sparse_label: bool = False,
        sparse_ignore_iou_threshold: float = 0.5,
        sparse_ignore_negative_weight: float = 1.0,
        sparse_unmatched_negative_weight: float = 1.0,
    ) -> None:
        """Initialize criterion configuration.

        Args:
            matcher: Hungarian matcher module. Returns
                `{"indices": list[(src_idx, tgt_idx)]}`.
            weight_dict: Loss weights keyed by loss names.
            losses: Enabled loss groups (`mal`, `boxes`, `local`, `distill`, ...).
            alpha: Focal/VFL alpha.
            gamma: Focal/VFL/MAL gamma.
            num_classes: Number of foreground classes.
            reg_max: Number of distance bins minus one (`num_bins = reg_max + 1`).
            boxes_weight_format: Optional weighting mode for bbox losses (`iou` or `giou`).
            share_matched_indices: Legacy compatibility option (kept for config parity).
            mal_alpha: Optional alpha used only by MAL.
            use_uni_set: If True, uses union matching across decoder layers for
                `boxes` and `local`.
            distill_adaptive_params: Reserved for adaptive distillation weighting.
            sparse_label: Enables sparse-label query weighting.
            sparse_ignore_iou_threshold: IoU threshold to mark a query as overlapping
                an ignore box.
            sparse_ignore_negative_weight: Penalty weight for unmatched queries that
                overlap ignore boxes.
            sparse_unmatched_negative_weight: Base weight for unmatched queries.
        """
        super().__init__()
        # 基本設定
        self.num_classes: int = num_classes
        self.matcher: Any = matcher
        self.weight_dict: Mapping[str, float] = weight_dict
        self.losses: Sequence[str] = losses
        self.boxes_weight_format: str | None = boxes_weight_format
        self.share_matched_indices: bool = share_matched_indices
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.fgl_targets: tuple[Tensor, Tensor, Tensor] | None = None
        self.fgl_targets_dn: tuple[Tensor, Tensor, Tensor] | None = None
        self.own_targets: Any | None = None
        self.own_targets_dn: Any | None = None
        self.reg_max: int = reg_max
        self.num_pos: Tensor | None = None
        self.num_neg: Tensor | None = None
        self.mal_alpha: float | None = mal_alpha
        self.use_uni_set: bool = use_uni_set

        # sparse label 関連設定
        self.distill_adaptive_params: Mapping[str, Any] | None = distill_adaptive_params
        self.sparse_label: bool = sparse_label
        self.sparse_ignore_iou_threshold: float = sparse_ignore_iou_threshold
        self.sparse_ignore_negative_weight: float = sparse_ignore_negative_weight
        self.sparse_unmatched_negative_weight: float = sparse_unmatched_negative_weight
        self._sparse_debug_stats: dict[str, Tensor] = {}

    def loss_distillation(
        self,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        indices: MatchIndices | None,
        num_boxes: float | None,
        **kwargs: Any,
    ) -> LossDict:
        """Cosine feature distillation loss.

        Shapes:
        - `student_distill_output`: (B, D, Hs, Ws)
        - `teacher_encoder_output`: (B, D, Ht, Wt)

        Returns:
            `{"loss_distill": scalar_tensor}`. If required tensors are missing,
            returns a differentiable zero scalar.
        """
        del targets, indices, num_boxes, kwargs

        student_feature_map: Tensor | None = outputs.get("student_distill_output")
        teacher_feature_map: Tensor | None = outputs.get("teacher_encoder_output")

        # distill用の特徴が無い場合は 0 を返して他lossだけで学習を継続する。
        if student_feature_map is None or teacher_feature_map is None:
            # Keep graph-compatible scalar for stable caller code.
            device: torch.device = (
                student_feature_map.device if student_feature_map is not None else torch.device("cuda")
            )
            return {"loss_distill": torch.tensor(0.0, device=device, requires_grad=True)}

        # チャネル次元が一致しないと cosine を取れないためここで停止する。
        if student_feature_map.shape[1] != teacher_feature_map.shape[1]:
            _logger.error(
                f"[RTv4Criterion] Feature dimension mismatch! Student: {student_feature_map.shape[1]}, Teacher: {teacher_feature_map.shape[1]}")
            raise ValueError("Feature dimension mismatch between student and teacher for distillation loss.")

        H_s: int
        W_s: int
        H_t: int
        W_t: int
        H_s, W_s = student_feature_map.shape[2:]  # student spatial size
        H_t, W_t = teacher_feature_map.shape[2:]  # teacher spatial size

        # 空間解像度が違う場合は teacher を student 側へ合わせる。
        if (H_s, W_s) != (H_t, W_t):
            _logger.warning(
                f"[RTv4Criterion] Resizing teacher feature map from {H_t}x{W_t} to student's {H_s}x{W_s} for distillation.")
            teacher_feature_map = F.interpolate(
                teacher_feature_map,
                size=(H_s, W_s),
                mode="bilinear",
                align_corners=False,
            )

        # (B, D, H, W) -> (B, H*W, D)
        student_output_flat: Tensor = student_feature_map.flatten(2).permute(0, 2, 1)
        teacher_output_flat: Tensor = teacher_feature_map.flatten(2).permute(0, 2, 1)

        # Normalize per token, then cosine similarity along D.
        student_output_norm: Tensor = F.normalize(student_output_flat, p=2, dim=-1)
        teacher_output_norm: Tensor = F.normalize(teacher_output_flat, p=2, dim=-1)

        cos_sim: Tensor = F.cosine_similarity(student_output_norm, teacher_output_norm, dim=-1)
        loss_distill: Tensor = (1 - cos_sim).mean()

        return {'loss_distill': loss_distill}

    def _get_distillation_weight_for_epoch(self) -> float:
        """Return distillation weight used at the current epoch.

        Current behavior is fixed weight from `weight_dict`.
        """
        fixed_weight: float = self.weight_dict.get("loss_distill", 0.0)
        return fixed_weight

    def loss_labels_focal(
        self,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        indices: MatchIndices,
        num_boxes: float,
        query_weights: Tensor | None = None,
    ) -> LossDict:
        """Sigmoid focal classification loss.

        Shapes:
        - `pred_logits`: (B, Q, C)
        - `target`: (B, Q, C)
        - `query_weights` (optional): (B, Q)
        """
        assert 'pred_logits' in outputs
        src_logits: Tensor = outputs["pred_logits"]
        idx: IndexPair = self._get_src_permutation_idx(indices)
        # Hungarian対応から「queryごとの正解クラス」を埋める。
        target_classes_o: Tensor = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes: Tensor = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        # 未マッチqueryは `self.num_classes` (= no-object) のまま残る。
        target_classes[idx] = target_classes_o
        target: Tensor = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        # one-hotの最終列(no-object)は focal実装に合わせて削除する。
        loss: Tensor = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction="none")
        if query_weights is not None:
            # sparse label 時は query単位重みで分類損失を調整する。
            loss = loss * query_weights.unsqueeze(-1)
        # DETR系実装に合わせ、query数と num_boxes で正規化。
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(
        self,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        indices: MatchIndices,
        num_boxes: float,
        values: Tensor | None = None,
        query_weights: Tensor | None = None,
    ) -> LossDict:
        """Varifocal loss with IoU target score.

        Shapes:
        - `pred_logits`: (B, Q, C)
        - `pred_boxes`: (B, Q, 4), `cxcywh`
        - `values` (optional): (num_matched,), IoU-alike quality for matches
        """
        assert 'pred_boxes' in outputs
        idx: IndexPair = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes: Tensor = outputs["pred_boxes"][idx]
            target_boxes: Tensor = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious: Tensor
            # values未指定時は matched pair の IoU を品質スコアとして使う。
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            # get_loss_meta_info から渡された品質スコアを優先。
            ious = values

        src_logits: Tensor = outputs["pred_logits"]
        target_classes_o: Tensor = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes: Tensor = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target: Tensor = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        # matched位置にだけ IoUターゲットを埋める。
        target_score_o: Tensor = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score: Tensor = target_score_o.unsqueeze(-1) * target

        # VFLの重み: 負例は予測スコア依存、正例はtarget_score依存。
        pred_score: Tensor = F.sigmoid(src_logits).detach()
        weight: Tensor = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss: Tensor = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        if query_weights is not None:
            # sparse label で unmatched/ignore の効きを制御する。
            loss = loss * query_weights.unsqueeze(-1)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(
        self,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        indices: MatchIndices,
        num_boxes: float,
        values: Tensor | None = None,
        query_weights: Tensor | None = None,
    ) -> LossDict:
        """Matching-Aware Loss (MAL) for classification branch.

        Compared with VFL, MAL uses a stronger separation between matched and
        unmatched targets through target shaping and weight design.
        """
        assert 'pred_boxes' in outputs
        idx: IndexPair = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes: Tensor = outputs["pred_boxes"][idx]
            target_boxes: Tensor = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious: Tensor
            # MALでも matched pair の IoU を教師信号に使う。
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits: Tensor = outputs["pred_logits"]
        target_classes_o: Tensor = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes: Tensor = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target: Tensor = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        # まず IoU を class target に投影する。
        target_score_o: Tensor = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score: Tensor = target_score_o.unsqueeze(-1) * target

        pred_score: Tensor = F.sigmoid(src_logits).detach()
        # MALの設計に合わせて正例ターゲットを gamma で変形。
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha is not None:
            weight: Tensor = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss: Tensor = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        if query_weights is not None:
            loss = loss * query_weights.unsqueeze(-1)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(
        self,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        indices: MatchIndices,
        num_boxes: float,
        boxes_weight: Tensor | None = None,
    ) -> LossDict:
        """BBox regression losses.

        Shapes:
        - `pred_boxes[idx]`: (num_matched, 4), normalized `cxcywh`
        - `target_boxes`: (num_matched, 4), normalized `cxcywh`

        Returns:
            Dict with:
            - `loss_bbox`: L1 sum / num_boxes
            - `loss_giou`: (1 - GIoU) sum / num_boxes
        """
        assert 'pred_boxes' in outputs
        idx: IndexPair = self._get_src_permutation_idx(indices)
        src_boxes: Tensor = outputs["pred_boxes"][idx]
        target_boxes: Tensor = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses: LossDict = {}
        # bbox L1: cx, cy, w, h の絶対誤差の総和。
        loss_bbox: Tensor = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou: 重なりが無いケースでも距離関係を反映できる。
        loss_giou: Tensor = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        # 追加の重み（iou/giouベース）を使う設定ではここで乗算。
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(
        self,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        indices: MatchIndices,
        num_boxes: float,
        T: float = 5,
    ) -> LossDict:
        """Compute local distribution losses: FGL and DDF.

        FGL:
        - uses matched queries only,
        - applies 2-bin interpolated CE on corner-distance distributions,
        - and multiplies each matched sample by its IoU.

        DDF:
        - KL between student and teacher corner distributions,
        - weighted by teacher confidence / matched IoU,
        - balanced between matched vs unmatched tokens.
        """
        losses: LossDict = {}
        if "pred_corners" not in outputs:
            # local headが無いモデル設定では FGL/DDF は計算しない。
            return losses

        # matched query と対応GTを一列に並べる。
        idx: IndexPair = self._get_src_permutation_idx(indices)
        target_boxes: Tensor = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)],
            dim=0,
        )  # (num_matched, 4) in normalized cxcywh

        # 4辺ぶんを独立サンプルとして扱える形へ変形。
        # (num_matched, 4 * (reg_max + 1)) -> (num_matched * 4, reg_max + 1)
        pred_corners: Tensor = outputs["pred_corners"][idx].reshape(-1, self.reg_max + 1)
        ref_points: Tensor = outputs["ref_points"][idx].detach()  # (num_matched, 4)

        # bbox2distance はコストが高いので、forward内で再利用できるようキャッシュする。
        with torch.no_grad():
            if self.fgl_targets_dn is None and "is_dn" in outputs:
                self.fgl_targets_dn = bbox2distance(
                    ref_points,
                    box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max,
                    outputs["reg_scale"],
                    outputs["up"],
                )
            if self.fgl_targets is None and "is_dn" not in outputs:
                self.fgl_targets = bbox2distance(
                    ref_points,
                    box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max,
                    outputs["reg_scale"],
                    outputs["up"],
                )

        target_corners: Tensor
        weight_right: Tensor
        weight_left: Tensor
        target_corners, weight_right, weight_left = self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets

        # matched pairごとの IoU（FGLの重み）を算出。
        ious: Tensor = torch.diag(
            box_iou(
                box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]),
                box_cxcywh_to_xyxy(target_boxes),
            )[0]
        )  # (num_matched,)

        # FGLは4辺それぞれにCEがあるため、IoUも4辺ぶんへ展開する。
        # (num_matched,) -> (num_matched, 4) -> (num_matched * 4,)
        weight_targets: Tensor = ious.unsqueeze(-1).repeat(1, 4).reshape(-1).detach()

        # FGL本体: 2bin補間CE + IoU重み + num_boxes正規化
        losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
            pred_corners,
            target_corners,
            weight_right,
            weight_left,
            weight_targets,
            avg_factor=num_boxes,
        )

        if "teacher_corners" in outputs:
            # DDFは全query（matched/unmatched）を対象に teacher 分布へ蒸留する。
            pred_corners_all: Tensor = outputs["pred_corners"].reshape(-1, self.reg_max + 1)
            target_corners_all: Tensor = outputs["teacher_corners"].reshape(-1, self.reg_max + 1)
            if not torch.equal(pred_corners_all, target_corners_all):
                # teacher分類スコアを、未マッチquery側の基本重みに使う。
                weight_targets_local: Tensor = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                # matched判定マスクを作り、corner単位へ展開する。
                mask: Tensor = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                # matched側は teacher信頼度ではなく IoU を重みとして採用する。
                weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                # KLDivLoss は (N, num_bins) なので corner軸を畳んだ入力で計算する。
                loss_match_local: Tensor = (
                    weight_targets_local
                    * (T**2)
                    * (
                        nn.KLDivLoss(reduction="none")(
                            F.log_softmax(pred_corners_all / T, dim=1),
                            F.softmax(target_corners_all.detach() / T, dim=1),
                        )
                    ).sum(-1)
                )

                if "is_dn" not in outputs:
                    # GPUごとのbatch差で比率が崩れないように matched/unmatched 重みを調整。
                    batch_scale: float = 8 / outputs["pred_boxes"].shape[0]
                    self.num_pos = (mask.sum() * batch_scale) ** 0.5
                    self.num_neg = ((~mask).sum() * batch_scale) ** 0.5

                # matched / unmatched の2群を別平均してから重み付き合成する。
                loss_match_local1: Tensor | int = loss_match_local[mask].mean() if mask.any() else 0
                loss_match_local2: Tensor | int = loss_match_local[~mask].mean() if (~mask).any() else 0
                losses["loss_ddf"] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (
                    self.num_pos + self.num_neg
                )

        return losses

    def _get_src_permutation_idx(
        self,
        indices: MatchIndices,
    ) -> IndexPair:
        """Build flattened index tensors for gathering matched predictions.

        Args:
            indices: Per-image match list where each entry is `(src_idx, tgt_idx)`.

        Returns:
            `(batch_idx, src_idx)`:
            - `batch_idx`: (num_matched,)
            - `src_idx`: (num_matched,)
            so `pred[batch_idx, src_idx]` gathers all matched query predictions.
        """
        # imageごとのsrc_idxに対応する batch番号を作る。
        batch_idx: Tensor = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx: Tensor = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: MatchIndices) -> IndexPair:
        """Build flattened target indices aligned with `_get_src_permutation_idx`."""
        # target側も同じ並び順で平坦化する。
        batch_idx: Tensor = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx: Tensor = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(
        self,
        indices: MatchIndices,
        indices_aux_list: Sequence[MatchIndices],
    ) -> MatchIndices:
        """Create union matching set across main/aux decoder layers.

        For each image, candidates from all layers are merged. If duplicate source
        query ids map to different targets, the most frequent mapping is preferred.
        """
        results: MatchIndices = []
        # まず main と aux の対応を連結し、候補集合を作る。
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        # 同一srcが複数targetへ紐づく場合は出現頻度が高い対応を採択する。
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique: Tensor
            counts: Tensor
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices: Tensor = torch.argsort(counts, descending=True)
            unique_sorted: Tensor = unique[count_sort_indices]
            column_to_row: dict[int, int] = {}
            for pair in unique_sorted:
                row_idx, col_idx = pair[0].item(), pair[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows: Tensor = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols: Tensor = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self) -> None:
        """Clear per-forward cached tensors/statistics."""
        # forward呼び出し単位でキャッシュを初期化する。
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None
        self._sparse_debug_stats = {}

    def pop_sparse_debug_stats(self) -> dict[str, Tensor]:
        """Return sparse-label debug metrics and clear internal buffer."""
        stats: dict[str, Tensor] = self._sparse_debug_stats
        self._sparse_debug_stats = {}
        return stats

    def _build_sparse_query_weights(
        self,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        indices: MatchIndices,
        record_debug: bool = False,
    ) -> Tensor | None:
        """Build per-query classification weights for sparse-label training.

        Policy implemented in this project:
        - Matched queries: weight = 1.0
        - Unmatched + overlap ignore: weight >= sparse_ignore_negative_weight
        - Other unmatched: sparse_unmatched_negative_weight (possibly 0.0)

        Returns:
            `(B, Q)` weights or `None` (means all ones).
        """
        # sparse学習を使わない設定では重みを返さない（= 全query重み1）。
        if not self.sparse_label:
            if record_debug:
                self._sparse_debug_stats = {}
            return None
        if "pred_boxes" not in outputs or "pred_logits" not in outputs:
            if record_debug:
                self._sparse_debug_stats = {}
            return None

        pred_boxes: Tensor = outputs["pred_boxes"]  # (B, Q, 4), cxcywh
        pred_logits: Tensor = outputs["pred_logits"]  # (B, Q, C)
        bs, num_queries = pred_logits.shape[:2]
        device: torch.device = pred_logits.device
        dtype: torch.dtype = pred_logits.dtype
        pred_boxes_xyxy: Tensor = box_cxcywh_to_xyxy(pred_boxes)

        # unmatchedの基本重み（0なら無罪化、1なら従来通り）。
        query_weights: Tensor | None = None
        if self.sparse_unmatched_negative_weight != 1.0:
            query_weights = torch.full(
                (bs, num_queries), self.sparse_unmatched_negative_weight, dtype=dtype, device=device
            )
            for b_idx, (src_idx, _) in enumerate(indices):
                if src_idx.numel() > 0:
                    query_weights[b_idx, src_idx] = 1.0

        ignore_mask_list: list[Tensor] = [torch.zeros(num_queries, dtype=torch.bool, device=device) for _ in range(bs)]

        for b_idx, (target, (src_idx, _)) in enumerate(zip(targets, indices)):
            ignore_boxes: Tensor | None = target.get("ignore_boxes", None)
            if ignore_boxes is None or ignore_boxes.numel() == 0:
                continue

            ignore_boxes_xyxy: Tensor = box_cxcywh_to_xyxy(ignore_boxes)
            ious: Tensor
            ious, _ = box_iou(pred_boxes_xyxy[b_idx], ignore_boxes_xyxy)
            if ious.numel() == 0:
                continue
            # ignore box と重なる未マッチqueryを抽出する。
            ignore_mask: Tensor = ious.max(dim=1).values >= self.sparse_ignore_iou_threshold
            if src_idx.numel() > 0:
                # matched query は常に正例扱いなので ignore 判定から除外。
                ignore_mask[src_idx] = False

            ignore_mask_list[b_idx] = ignore_mask

            if ignore_mask.any():
                if query_weights is None:
                    query_weights = torch.ones((bs, num_queries), dtype=dtype, device=device)
                # ignore重なり未マッチには下限ペナルティを与える。
                ignore_penalty: Tensor = torch.full_like(
                    query_weights[b_idx, ignore_mask], self.sparse_ignore_negative_weight
                )
                query_weights[b_idx, ignore_mask] = torch.maximum(query_weights[b_idx, ignore_mask], ignore_penalty)

        if record_debug:
            metric_weights: Tensor | None = query_weights
            if metric_weights is None:
                metric_weights = torch.ones((bs, num_queries), dtype=dtype, device=device)

            # ログ可視化用の統計を集計する。
            total_queries: int = int(bs * num_queries)
            total_unmatched: Tensor = torch.zeros((), dtype=torch.float32, device=device)
            total_ignore_unmatched: Tensor = torch.zeros((), dtype=torch.float32, device=device)
            sum_unmatched_weight: Tensor = torch.zeros((), dtype=torch.float32, device=device)
            sum_ignore_weight: Tensor = torch.zeros((), dtype=torch.float32, device=device)

            for b_idx, (src_idx, _) in enumerate(indices):
                matched_mask: Tensor = torch.zeros(num_queries, dtype=torch.bool, device=device)
                if src_idx.numel() > 0:
                    matched_mask[src_idx] = True
                unmatched_mask: Tensor = ~matched_mask
                ignore_unmatched_mask: Tensor = ignore_mask_list[b_idx] & unmatched_mask

                total_unmatched += unmatched_mask.float().sum()
                total_ignore_unmatched += ignore_unmatched_mask.float().sum()
                if unmatched_mask.any():
                    sum_unmatched_weight += metric_weights[b_idx, unmatched_mask].float().sum()
                if ignore_unmatched_mask.any():
                    sum_ignore_weight += metric_weights[b_idx, ignore_unmatched_mask].float().sum()

            unmatched_denom: Tensor = torch.clamp(total_unmatched, min=1.0)
            ignore_denom: Tensor = torch.clamp(total_ignore_unmatched, min=1.0)
            query_denom: float = float(max(total_queries, 1))

            sparse_debug: dict[str, Tensor] = {
                "loss_sparse_unmatched_penalty": sum_unmatched_weight / unmatched_denom,
                "loss_sparse_ignore_penalty": (
                    sum_ignore_weight / ignore_denom
                    if float(total_ignore_unmatched.item()) > 0
                    else torch.zeros((), dtype=torch.float32, device=device)
                ),
                "loss_sparse_unmatched_ratio": total_unmatched / query_denom,
                "loss_sparse_ignore_ratio": total_ignore_unmatched / unmatched_denom,
            }
            self._sparse_debug_stats = sparse_debug

        return query_weights

    def get_loss(
        self,
        loss: str,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        indices: MatchIndices | None,
        num_boxes: float | None,
        **kwargs: Any,
    ) -> LossDict:
        """Dispatch loss computation by loss-group key."""
        # 損失名 -> 実装関数の対応表。
        loss_map: dict[str, Any] = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
            "local": self.loss_local,  # FGL + DDF
            "distill": self.loss_distillation,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(
        self,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        **kwargs: Any,
    ) -> LossDict:
        """Compute all configured losses for main and auxiliary branches.

        Steps:
        1. Match final decoder outputs to GT (Hungarian).
        2. Optionally build union matching set across aux branches.
        3. Compute configured losses on main output.
        4. Repeat for aux / pre / encoder aux / denoising outputs.
        5. Apply per-loss weights from `weight_dict`.
        """
        # `aux` を除いた最終出力（main branch）を先に取り出す。
        outputs_without_aux: OutputsDict = {k: v for k, v in outputs.items() if "aux" not in k}

        # 最終decoder出力で Hungarian matching を計算。
        indices: MatchIndices = self.matcher(outputs_without_aux, targets)["indices"]
        self._clear_cache()
        # debug用の sparse 統計をこのタイミングで更新しておく。
        self._build_sparse_query_weights(outputs_without_aux, targets, indices, record_debug=True)

        # boxes/local は union matching を使えるように準備。
        indices_go: MatchIndices = indices
        num_boxes_go: float | None = None
        cached_indices: list[MatchIndices] = []
        cached_indices_enc: list[MatchIndices] = []
        if 'aux_outputs' in outputs:
            indices_aux_list: list[MatchIndices] = []
            aux_outputs_list: list[OutputsDict] = outputs["aux_outputs"]
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                # 各aux層の matching を保持（aux loss 計算時に再利用）。
                indices_aux: MatchIndices = self.matcher(aux_outputs, targets)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc: MatchIndices = self.matcher(aux_outputs, targets)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go_count: int = sum(len(x[0]) for x in indices_go)
            num_boxes_go_tensor: Tensor = torch.as_tensor(
                [num_boxes_go_count], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go_tensor)
            # 分散学習時は全rank平均の num_boxes を使う。
            num_boxes_go = torch.clamp(num_boxes_go_tensor / get_world_size(), min=1).item()
        else:
            if 'enc_aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                    indices_enc: MatchIndices = self.matcher(aux_outputs, targets)["indices"]
                    cached_indices_enc.append(indices_enc)

        # 損失正規化用の num_boxes（全rank平均）を計算。
        num_boxes_count: int = sum(len(t["labels"]) for t in targets)
        num_boxes_tensor: Tensor = torch.as_tensor(
            [num_boxes_count], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes_tensor)
        num_boxes: float = torch.clamp(num_boxes_tensor / get_world_size(), min=1).item()
        if num_boxes_go is None:
            num_boxes_go = num_boxes

        # main出力の損失を計算。
        losses: LossDict = {}
        for loss_name in self.losses:
            if loss_name == 'distill':
                l_dict: LossDict = self.get_loss(loss_name, outputs, targets, None, None, **kwargs)
                if 'loss_distill' in l_dict and l_dict['loss_distill'] != 0:
                    dynamic_weight: float = self._get_distillation_weight_for_epoch()
                    l_dict['loss_distill'] = l_dict['loss_distill'] * dynamic_weight
                losses.update(l_dict)
            else:
                # boxes/local は union matching、それ以外はmain matchingを使う。
                use_uni_set: bool = self.use_uni_set and (loss_name in ["boxes", "local"])
                indices_in: MatchIndices = indices_go if use_uni_set else indices
                num_boxes_in: float = num_boxes_go if use_uni_set else num_boxes
                meta: dict[str, Tensor] = self.get_loss_meta_info(loss_name, outputs, targets, indices_in)
                l_dict: LossDict = self.get_loss(loss_name, outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                losses.update(l_dict)

        # Decoder auxiliary layers.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if "local" in self.losses:
                    # local loss で必要なパラメータを main から引き継ぐ。
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    use_uni_set: bool = self.use_uni_set and (loss in ["boxes", "local"])
                    indices_in: MatchIndices = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in: float = num_boxes_go if use_uni_set else num_boxes
                    meta: dict[str, Tensor] = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict: LossDict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Traditional head output at first decoder layer (D-FINE only).
        if 'pre_outputs' in outputs:
            aux_outputs: OutputsDict = outputs["pre_outputs"]
            for loss in self.losses:
                use_uni_set: bool = self.use_uni_set and (loss in ["boxes", "local"])
                indices_in: MatchIndices = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in: float = num_boxes_go if use_uni_set else num_boxes
                meta: dict[str, Tensor] = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict: LossDict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # Encoder auxiliary outputs.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic: bool = outputs["enc_meta"]["class_agnostic"]
            enc_targets: Sequence[TargetDict]
            if class_agnostic:
                # class agnostic時は全ラベルを0へ寄せてからencoder lossを計算。
                orig_num_classes: int = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set: bool = self.use_uni_set and (loss == "boxes")
                    indices_in: MatchIndices = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in: float = num_boxes_go if use_uni_set else num_boxes
                    meta: dict[str, Tensor] = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict: LossDict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # Denoising branch outputs.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn: MatchIndices = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes: float = num_boxes * outputs["dn_meta"]["dn_num_group"]

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if "local" in self.losses:
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta: dict[str, Tensor] = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict: LossDict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # Denoising pre-head output (D-FINE only).
            if 'dn_pre_outputs' in outputs:
                aux_outputs: OutputsDict = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta: dict[str, Tensor] = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict: LossDict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 学習ログが壊れないよう NaN を 0 に置換。
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(
        self,
        loss: str,
        outputs: OutputsDict,
        targets: Sequence[TargetDict],
        indices: MatchIndices,
    ) -> dict[str, Tensor]:
        """Build auxiliary tensors needed by selected losses.

        Returns possible keys:
        - `query_weights`: (B, Q), for sparse-label classification losses.
        - `boxes_weight`: (num_matched,), for bbox losses.
        - `values`: (num_matched,), IoU/GIoU quality for VFL/MAL.
        """
        meta: dict[str, Tensor] = {}

        # 分類系lossのみ query重み（sparse設定）を受け取る。
        if loss in ("focal", "vfl", "mal"):
            query_weights: Tensor | None = self._build_sparse_query_weights(outputs, targets, indices)
            if query_weights is not None:
                meta["query_weights"] = query_weights

        # bbox重み設定が無ければここで終了。
        if self.boxes_weight_format is None:
            return meta

        src_boxes: Tensor = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes: Tensor = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        iou: Tensor
        if self.boxes_weight_format == 'iou':
            # detached pred を使い、重み経由で勾配が流れないようにする。
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou( \
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError(f"Unsupported boxes_weight_format: {self.boxes_weight_format}")

        if loss in ('boxes',):
            meta["boxes_weight"] = iou
        elif loss in ('vfl', 'mal'):
            meta["values"] = iou

        return meta

    @staticmethod
    def get_cdn_matched_indices(
        dn_meta: Mapping[str, Any],
        targets: Sequence[TargetDict],
    ) -> MatchIndices:
        """Create denoising branch match indices.

        Args:
            dn_meta:
                - `dn_positive_idx`: list[(num_gt_i * dn_num_group,)]
                - `dn_num_group`: int
            targets: batch targets.
        """
        # dn側は「GT index を dn_num_group 回複製した対応」を使う。
        dn_positive_idx: Sequence[Tensor] = dn_meta["dn_positive_idx"]
        dn_num_group: int = dn_meta["dn_num_group"]
        num_gts: list[int] = [len(t["labels"]) for t in targets]
        device: torch.device = targets[0]["labels"].device

        dn_match_indices: MatchIndices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx: Tensor = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                                         torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices

    def feature_loss_function(self, fea: Tensor, target_fea: Tensor) -> Tensor:
        """Legacy feature loss helper (not used in current forward path)."""
        loss: Tensor = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self,
        pred: Tensor,
        label: Tensor,
        weight_right: Tensor,
        weight_left: Tensor,
        weight: Tensor | None = None,
        reduction: str = "sum",
        avg_factor: float | None = None,
    ) -> Tensor:
        """2-bin interpolated CE used by FGL.

        Shapes:
        - `pred`: (N, reg_max + 1), per-side logits.
        - `label`: (N,), left-bin float index from `bbox2distance`.
        - `weight_left/right`: (N,), interpolation weights.
        - `weight` (optional): (N,), sample-level scaling (IoU in FGL).
        """
        # labelは左binのfloat値。longにして左bin indexにする。
        dis_left: Tensor = label.long()
        # 右binは左bin + 1。
        dis_right: Tensor = dis_left + 1

        # 左右2binのCEを補間係数で合成。
        loss: Tensor = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1) + F.cross_entropy(
            pred, dis_right, reduction="none"
        ) * weight_right.reshape(-1)

        if weight is not None:
            # FGLではここに IoU重みが入る。
            weight = weight.float()
            loss = loss * weight

        # avg_factor 指定がある場合はそれを最優先で使う。
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs: OutputsDict) -> list[float]:
        """Return layer-wise coefficients in [0.5, 1.0] for gradual schedule."""
        # aux層数に応じて 0.5 -> 1.0 を線形に割り当てる。
        num_layers: int = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step: float = 0.5 / (num_layers - 1)
        opt_list: list[float] = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
