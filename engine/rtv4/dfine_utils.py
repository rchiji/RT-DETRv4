"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch

from .box_ops import box_xyxy_to_cxcywh

Tensor = torch.Tensor


def weighting_function(
    reg_max: int,
    up: Tensor,
    reg_scale: Tensor | float,
    deploy: bool = False,
) -> Tensor:
    """D-FINE の非一様重み関数 `W(n)` を生成する。

    形状の前提:
    - `up`: 通常は `(1,)`
    - 返り値: `(reg_max + 1,)`

    Args:
        reg_max: 距離分布の最大bin index（bin数は `reg_max + 1`）。
        up: 重み関数の上下限を決める係数。
        reg_scale: 曲率を制御する係数。
        deploy: True の場合は `.item()` ベースで固定値列を構築。

    Returns:
        重み関数の値列（`W(0)...W(reg_max)`）。
    """
    # deploy時は Python float で値を組み立て、最後に Tensor 化する。
    if deploy:
        upper_bound1: float = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2: float = (abs(up[0]) * abs(reg_scale) * 2).item()
        step: float = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values: list[float] = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values: list[float] = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values: list[Tensor | float] = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)

    # 学習時は Tensor のまま構築して autograd / dtype / device を自然に合わせる。
    upper_bound1: Tensor = abs(up[0]) * abs(reg_scale)
    upper_bound2: Tensor = abs(up[0]) * abs(reg_scale) * 2
    step: Tensor = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values_t: list[Tensor] = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values_t: list[Tensor] = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values_t: list[Tensor] = (
        [-upper_bound2]
        + left_values_t
        + [torch.zeros_like(up[0][None])]
        + right_values_t
        + [upper_bound2]
    )
    return torch.cat(values_t, 0)


def translate_gt(
    gt: Tensor,
    reg_max: int,
    reg_scale: Tensor | float,
    up: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """連続値GTを離散bin + 左右補間重みに変換する。

    Args:
        gt: 連続距離のGT。形状は任意だが内部で `(N,)` に平坦化する。
        reg_max: 最大bin index。
        reg_scale: 重み関数の曲率制御パラメータ。
        up: 重み関数の上下限制御パラメータ。

    Returns:
        - indices: 左bin index（float, `(N,)`）
        - weight_right: 右bin重み（`(N,)`）
        - weight_left: 左bin重み（`(N,)`）
    """
    gt_flat: Tensor = gt.reshape(-1)
    function_values: Tensor = weighting_function(reg_max, up, reg_scale)

    # 各GT値に対して「直近の左bin index」を探す。
    diffs: Tensor = function_values.unsqueeze(0) - gt_flat.unsqueeze(1)
    mask: Tensor = diffs <= 0
    closest_left_indices: Tensor = torch.sum(mask, dim=1) - 1

    # CE教師として使う左bin index（float）と左右補間重みを初期化する。
    indices: Tensor = closest_left_indices.float()
    weight_right: Tensor = torch.zeros_like(indices)
    weight_left: Tensor = torch.zeros_like(indices)

    # 有効範囲 [0, reg_max) の要素だけ補間重みを計算する。
    valid_idx_mask: Tensor = (indices >= 0) & (indices < reg_max)
    valid_indices: Tensor = indices[valid_idx_mask].long()

    left_values: Tensor = function_values[valid_indices]
    right_values: Tensor = function_values[valid_indices + 1]

    left_diffs: Tensor = torch.abs(gt_flat[valid_idx_mask] - left_values)
    right_diffs: Tensor = torch.abs(right_values - gt_flat[valid_idx_mask])

    # GTが右bin寄りなら weight_right が大きくなる。
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # 下限外: 最左binへ寄せる（left=1, right=0）。
    invalid_idx_mask_neg: Tensor = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    # 上限外: 最右bin側へ寄せる（left=0, right=1）。
    # `reg_max - 0.1` にして `long()` 時に `reg_max - 1` へ落とす。
    invalid_idx_mask_pos: Tensor = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def distance2bbox(
    points: Tensor,
    distance: Tensor,
    reg_scale: Tensor | float,
) -> Tensor:
    """参照点 + 4辺距離から bbox (`cxcywh`) を復元する。

    Args:
        points: `(B, N, 4)` または `(N, 4)`。 `[x, y, w, h]`。
        distance: `points` と同じ先頭shapeで4辺距離 `[l, t, r, b]`。
        reg_scale: 重み関数由来のスケール。

    Returns:
        復元した bbox（`cxcywh`）。
    """
    reg_scale_abs: Tensor | float = abs(reg_scale)

    # 点と距離表現から xyxy を復元。
    x1: Tensor = points[..., 0] - (0.5 * reg_scale_abs + distance[..., 0]) * (points[..., 2] / reg_scale_abs)
    y1: Tensor = points[..., 1] - (0.5 * reg_scale_abs + distance[..., 1]) * (points[..., 3] / reg_scale_abs)
    x2: Tensor = points[..., 0] + (0.5 * reg_scale_abs + distance[..., 2]) * (points[..., 2] / reg_scale_abs)
    y2: Tensor = points[..., 1] + (0.5 * reg_scale_abs + distance[..., 3]) * (points[..., 3] / reg_scale_abs)

    bboxes_xyxy: Tensor = torch.stack([x1, y1, x2, y2], -1)
    return box_xyxy_to_cxcywh(bboxes_xyxy)


def bbox2distance(
    points: Tensor,
    bbox: Tensor,
    reg_max: int | None,
    reg_scale: Tensor | float,
    up: Tensor,
    eps: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """bbox (`xyxy`) を参照点からの4辺距離表現へ変換する。

    Args:
        points: `(N, 4)` の `[x, y, w, h]`。
        bbox: `(N, 4)` の `xyxy`。
        reg_max: 最大bin index。`None` で clamp 無効。
        reg_scale: 重み関数の曲率制御パラメータ。
        up: 重み関数の上下限制御パラメータ。
        eps: `target < reg_max` を保つための余白。

    Returns:
        - target_corners: `(N*4,)` 左bin index（float）
        - weight_right: `(N*4,)` 右bin補間重み
        - weight_left: `(N*4,)` 左bin補間重み
    """
    reg_scale_abs: Tensor | float = abs(reg_scale)

    # 参照点から bbox の4辺までの正規化距離を計算。
    left: Tensor = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale_abs + 1e-16) - 0.5 * reg_scale_abs
    top: Tensor = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale_abs + 1e-16) - 0.5 * reg_scale_abs
    right: Tensor = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale_abs + 1e-16) - 0.5 * reg_scale_abs
    bottom: Tensor = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale_abs + 1e-16) - 0.5 * reg_scale_abs

    # (N, 4) にまとめて離散binへ変換。
    four_lens: Tensor = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale_abs, up)

    # CEのlabelとして範囲外を防ぐ（right index = left+1 が壊れないようにする）。
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)

    # criterion 側の実装に合わせて平坦化 + detach して返す。
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()
