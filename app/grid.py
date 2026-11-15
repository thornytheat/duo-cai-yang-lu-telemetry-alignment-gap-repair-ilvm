"""统一网格重采样：所有序列落到 1 分钟桶，**按连续采样档位分段解释**。

分段口径（见 app/segments.py）：
- 高频档（interval < 60s，如 1s/5s）→ 桶内 mean/max/min + 计数，source='agg'；
- 低频档（interval >= 60s，如 60s/120s/180s）→ 取桶内最后一点，source='sample'，
  该档内上报间隔被拉长（降级）时 hold 沿用，carried=True。
- hold 只在「前后原始格相距 <= 3 格」且「前后采样点间距 > 1.5 桶」时发生；
  跨档位边界的 hold 会额外打 cross_segment 标记，便于页面核对边界行为；
  更长空缺留给缺口模块按分段判定，1s 档偶尔丢一格也不会被静默补值。
- 首格之前没有前值 → 不反向外推（保持空）。

每个单元格带 segment_index / interval_s（所属分段及其档位），缺口与摘要据此按分段重算。
"""
from __future__ import annotations

from . import segments as seg_mod
from . import settings


def bucket_of(ts: float) -> int:
    return int(ts // settings.BUCKET_SECONDS)


def bucket_start_iso(bucket: int) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(bucket * settings.BUCKET_SECONDS, tz=_tz()).isoformat(timespec="seconds")


def _tz():
    from datetime import timedelta, timezone

    return timezone(timedelta(hours=8))


def _cell(bucket: int, **kwargs) -> dict:
    base = {
        "bucket": bucket,
        "bucket_start": bucket_start_iso(bucket),
        "value": None,
        "mean": None,
        "max": None,
        "min": None,
        "samples": 0,
        "source": "empty",   # agg | sample | hold | interpolated | empty
        "carried": False,
        "filled": False,     # 缺口插补或沿用后是否为「非原始」值
        "first_ts": None,
        "last_ts": None,
        "segment_index": None,   # 所属采样档位分段
        "interval_s": None,      # 所属分段的标称档位
        "cross_segment": False,  # 该格的 hold/插补是否跨越了分段边界
    }
    base.update(kwargs)
    return base


def plan_segments(
    points: list[tuple[float, float]],
    start_bucket: int,
    end_bucket: int,
    fallback_interval: int = settings.BUCKET_SECONDS,
):
    """识别档位分段并完成桶归属，返回 Segment 列表（纯函数，可独立单测）。"""
    parts = seg_mod.detect_segments(points, fallback_interval=fallback_interval)
    seg_mod.assign_buckets(parts, points, start_bucket, end_bucket)
    return parts


def resample(
    points: list[tuple[float, float]],
    interval_s: int,
    start_bucket: int,
    end_bucket: int,
) -> list[dict]:
    """把一条序列重采样到 [start_bucket, end_bucket] 的 1 分钟格（按档位分段）。"""
    parts = plan_segments(points, start_bucket, end_bucket, fallback_interval=interval_s)

    grouped: dict[int, list[tuple[float, float]]] = {}
    for ts, value in points:
        b = bucket_of(ts)
        if b < start_bucket or b > end_bucket:
            continue
        grouped.setdefault(b, []).append((ts, value))

    # 桶 → 所属分段
    owner: dict[int, int] = {}
    seg_by_index: dict[int, seg_mod.Segment] = {}
    for part in parts:
        seg_by_index[part.index] = part
        for b in range(part.start_bucket, part.end_bucket + 1):
            owner[b] = part.index

    cells: list[dict] = []
    for bucket in range(start_bucket, end_bucket + 1):
        part = seg_by_index.get(owner.get(bucket, -1), parts[0]) if parts else None
        samples = grouped.get(bucket)
        if not samples:
            cells.append(
                _cell(
                    bucket,
                    segment_index=part.index if part else None,
                    interval_s=part.interval_s if part else interval_s,
                )
            )
            continue
        values = [v for _, v in samples]
        high_rate = part is not None and seg_mod.is_high_rate(part.interval_s)
        first_ts = min(t for t, _ in samples)
        last_ts = max(t for t, _ in samples)
        if high_rate:
            cells.append(
                _cell(
                    bucket,
                    first_ts=first_ts,
                    last_ts=last_ts,
                    value=sum(values) / len(values),
                    mean=sum(values) / len(values),
                    max=max(values),
                    min=min(values),
                    samples=len(values),
                    source="agg",
                    segment_index=part.index,
                    interval_s=part.interval_s,
                )
            )
        else:
            last_ts, last_value = max(samples, key=lambda p: p[0])
            cells.append(
                _cell(
                    bucket,
                    first_ts=first_ts,
                    last_ts=last_ts,
                    value=last_value,
                    mean=last_value,
                    max=last_value,
                    min=last_value,
                    samples=len(values),
                    source="sample",
                    segment_index=part.index,
                    interval_s=part.interval_s,
                )
            )
    _apply_hold(cells)
    return cells


def segment_plans(
    points: list[tuple[float, float]],
    interval_s: int,
    start_bucket: int,
    end_bucket: int,
) -> list[dict]:
    """对外的分段描述（含桶区间与档位标签），供流水线/页面/导出复用。"""
    parts = plan_segments(points, start_bucket, end_bucket, fallback_interval=interval_s)
    return [p.to_dict() for p in parts]


def _apply_hold(cells: list[dict]) -> None:
    """低频档内 hold 沿用前值并计 carried；分段边界处的 hold 打 cross_segment。

    触发条件（两者同时满足）：
    - 前后原始格相距 <= HOLD_MAX_BRACKET_BUCKETS 格（更长的空缺交给缺口模块判定）；
    - 前后两个原始采样点的时间间距 > 1.5 桶（说明该段上报间隔确实稀疏，
      高频档偶尔丢一格时不静默补值）。

    边界语义：缺口插补绝不跨分段；hold 是「沿用最后一个已观测值」的弱补值，
    允许在边界处沿用，但显式标记 cross_segment=True，质量报告单独说明。
    """
    anchor_idx = [i for i, c in enumerate(cells) if c["source"] in ("agg", "sample")]
    if len(anchor_idx) < 2:
        return
    min_spacing_s = 1.5 * settings.BUCKET_SECONDS
    for prev, nxt in zip(anchor_idx, anchor_idx[1:]):
        distance = cells[nxt]["bucket"] - cells[prev]["bucket"]
        if distance <= 1 or distance > settings.HOLD_MAX_BRACKET_BUCKETS:
            continue
        prev_ts = cells[prev]["last_ts"]
        next_ts = cells[nxt]["first_ts"]
        if prev_ts is None or next_ts is None:
            continue
        if next_ts - prev_ts <= min_spacing_s:
            continue
        cross = cells[prev].get("segment_index") != cells[nxt].get("segment_index")
        hold_value = cells[prev]["value"]
        for i in range(prev + 1, nxt):
            cells[i]["value"] = hold_value
            cells[i]["mean"] = hold_value
            cells[i]["max"] = hold_value
            cells[i]["min"] = hold_value
            cells[i]["source"] = "hold"
            cells[i]["carried"] = True
            cells[i]["filled"] = True
            # 沿用跨越了采样档位分段边界：值仍可沿用，但显式标记以便页面/质量报告核对。
            if cross:
                cells[i]["cross_segment"] = True


def stats(cells: list[dict]) -> dict:
    counts = {"agg": 0, "sample": 0, "hold": 0, "empty": 0}
    for cell in cells:
        counts[cell["source"]] = counts.get(cell["source"], 0) + 1
    raw = counts["agg"] + counts["sample"]
    cross = sum(1 for c in cells if c.get("cross_segment"))
    return {
        "cells": len(cells),
        "agg_cells": counts["agg"],
        "sample_cells": counts["sample"],
        "held_cells": counts["hold"],
        "empty_cells": counts["empty"],
        "raw_cells": raw,
        "carried_cells": counts["hold"],
        "cross_segment_cells": cross,
    }
